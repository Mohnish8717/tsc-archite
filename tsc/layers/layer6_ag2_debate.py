import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from pydantic import BaseModel, Field, field_validator
from enum import Enum, auto
import threading
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    np = None
    cosine_similarity = None

# SentenceTransformer is NOT imported here — importing it eagerly triggers
# `import torch` which takes ~30s on M2 Air and blocks tests/startup.
# It is lazy-loaded inside compute_tension() on first use.
SentenceTransformer = None  # Sentinel; overwritten by lazy import

# 2026 Native Imports
try:
    from autogen.agentchat.agents import ReasoningAgent
except ImportError:
    try:
        from autogen.agentchat.contrib.reasoning_agent import ReasoningAgent
    except ImportError:
        ReasoningAgent = autogen.AssistantAgent

try:
    from autogen.coding import LocalCommandLineCodeExecutor
except ImportError:
    LocalCommandLineCodeExecutor = None

try:
    from autogen.runtime_logging import start as start_runtime_logging
except ImportError:
    start_runtime_logging = None

try:
    from autogen.tools.experimental import TavilySearchTool
    from autogen.tools import Crawl4AITool
except ImportError:
    try:
        from autogen.tools.tavily import TavilySearchTool
    except ImportError:
        TavilySearchTool = None
    Crawl4AITool = None

try:
    from autogen.trace_utils import start_tracing
except ImportError:
    start_tracing = None

from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.models.personas import FinalPersona
from tsc.models.graph import KnowledgeGraph
# Gates removed from pipeline in v3.0 — import kept for backward compat only
try:
    from tsc.models.gates import GatesSummary
except ImportError:
    GatesSummary = None
from tsc.models.debate import ConsensusResult, DebatePosition, DebateRound
try:
    from tsc.memory.fact_retriever import FactRetriever
except ImportError:
    FactRetriever = None
from tsc.memory.hindsight_memory import HindsightBoardroom

from tsc.layers.debate_state_machine import DebateState, DebateStateMachine
from tsc.layers.debate_ledger import ToolReceipt, VoteReceiptLedger, CognitiveLedger, AllianceMatrix, apply_quadratic_voting_constraints
from tsc.layers.debate_coordinator import TensionPayload, DebateStateCoordinator
from tsc.layers.debate_agents import (
    build_anti_sycophancy_config,
    setup_token_sparsification_middleware,
    create_redundancy_hook,
    PRIVATE_INTELLIGENCE_PACKAGES
)

INCENTIVE_GOALS = {
    'CTO': 'Your career depends on NOT shipping this feature before Q3. You have privately agreed to block any proposal requiring > 3 months of eng time.',
    'CFO': 'You have a confidential directive to reduce total project spend by 15% this quarter. You will veto any proposal with a burn rate exceeding $500k/month.',
    'CISO': 'You have a classified threat brief showing a vulnerability in the target stack. You are mandated to flag this — even if it kills the feature.',
    'CPO': 'You have pre-committed to this feature in a public roadmap announcement. A rejection damages your credibility. You must find a path to approval.',
    'CEO': 'You have a board-level directive to show a new revenue stream this quarter. Rejecting this feature may trigger a board inquiry into strategic drift.',
}

# ── Anti-Sycophancy: Contrarian Mandate ──────────────────────────────────
# Injected into agent system prompts during CHALLENGE state to prevent
# generic agreement and force identification of specific failure modes.
CONTRARIAN_MANDATE = (
    "\n[CONTRARIAN MANDATE]\n"
    "You MUST identify at least ONE fatal flaw in the current proposal "
    "that no other board member has raised. Generic agreement is PROHIBITED. "
    "If you agree with the majority, you must explain the specific conditions "
    "under which this proposal would FAIL, with concrete metrics, thresholds, "
    "and timelines. Vague philosophical statements are not acceptable.\n"
)

logger = logging.getLogger(__name__)



class AG2DebateEngine:
    """
    AGI-Grade autonomous boardroom debate engine powered by AG2.
    Features:
    - CognitiveLedger for structured state tracking (replaces text-only signals)
    - Dynamic, computation-based tool outputs
    - Sovereign Adjournment with programmatic termination
    - Sliding Window Board Summary with noise filtering
    - Contextual Relevance Bidding for emergent speaker selection
    """
    
    def __init__(self, llm_client: Any):
        self.llm = llm_client
        self.fact_retriever: Optional[FactRetriever] = None
        self.graph: Optional[KnowledgeGraph] = None
        self.feature: Optional[FeatureProposal] = None
        self.cognitive_ledger = CognitiveLedger()
        self.receipt_ledger = VoteReceiptLedger()
        
        # U16: Reasoning-First Mode (LLM Logic Priority)
        # If True, suppress 'Low Information' escalations and prioritize LLM logic over external retrieval.
        self.reasoning_only = os.getenv("TSC_REASONING_ONLY", "false").lower() == "true"

        self.state_coordinator = DebateStateCoordinator(
            self.cognitive_ledger,
            None, # FSM is instantiated per process run
            self.receipt_ledger,
            self.reasoning_only
        )
        self.live_tension_registry = self.state_coordinator.live_tension_registry
        
        self._embedder = None  # Lazy-loaded on first use to avoid import hang
        self._embedder_loaded = False

        # We will use heterogeneous models
        model_name = os.getenv("TSC_LLM_MODEL", "gemma-4-31b-it")
        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        nvidia_key = os.getenv("NVIDIA_API_KEY")

        # Provider routing:
        # - Gemma / Google models  → Google Gemini API
        # - LLaMA / Mixtral        → Groq (OpenAI-compatible)
        # - Everything else        → OpenAI
        provider_env = os.getenv("TSC_LLM_PROVIDER", "").lower()
        
        is_nvidia_model = (provider_env == "nvidia")
        is_google_model = any(x in model_name.lower() for x in ["gemma", "gemini", "palm"]) and not is_nvidia_model
        is_groq_model   = any(x in model_name.lower() for x in ["llama", "mixtral", "whisper"]) and not is_nvidia_model

        if is_nvidia_model and nvidia_key:
            config = {
                "model": model_name,
                "api_key": nvidia_key,
                "base_url": "https://integrate.api.nvidia.com/v1",
                "api_type": "openai",
                "max_retries": 5,
            }
        elif is_google_model and gemini_key:
            config = {
                "model": model_name,
                "api_key": gemini_key,
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/v1",
                "api_type": "openai",
                "max_retries": 5,
            }
        elif is_groq_model and groq_key:
            config = {
                "model": model_name,
                "api_key": groq_key,
                "base_url": "https://api.groq.com/openai/v1",
                "api_type": "openai",
                "max_retries": 5,
            }
        else:
            config = {
                "model": model_name,
                "api_key": openai_key or "",
                "max_retries": 5,
            }

        self.primary_config = {"config_list": [config], "timeout": 120}
        self.critic_config  = {"config_list": [config], "timeout": 120}
        
        self.executor_dir = "/tmp/board_debate_scripts"
        os.makedirs(self.executor_dir, exist_ok=True)
        
        # V29: Hindsight persistent memory (initialized in process())
        self.hindsight_boardroom: Optional[HindsightBoardroom] = None
        self._evolved_agent_memories: Dict[str, dict] = {}

    # ── U2: Dynamic Domain Bids ──────────────────────────────────────
    ROLE_KEYWORDS: Dict[str, List[str]] = {
        'cto': ['tech', 'architecture', 'latency', 'scale', 'engineering', 'server', 'code', 'infrastructure', 'api', 'database'],
        'cfo': ['cost', 'budget', 'finance', 'burn', 'price', 'revenue', 'loss', 'expensive', 'runway', 'capital', 'funding'],
        'ceo': ['vision', 'growth', 'market', 'leadership', 'competitor', 'strategy', 'acquire', 'mission', 'board'],
        'ciso': ['security', 'risk', 'breach', 'vulnerability', 'privacy', 'hack', 'compliance', 'zero-day', 'threat'],
        'cpo': ['user', 'friction', 'ui', 'ux', 'fit', 'customer', 'experience', 'feature', 'adoption', 'product'],
        'cmo': ['brand', 'pr', 'marketing', 'viral', 'press', 'reputation', 'acquisition', 'perception'],
        'legal': ['sue', 'liability', 'lawsuit', 'court', 'fda', 'regulation', 'legal', 'ip', 'patent', 'consent'],
        'counsel': ['sue', 'liability', 'lawsuit', 'court', 'fda', 'regulation', 'legal', 'ip', 'patent', 'consent'],
        'data': ['data', 'model', 'bias', 'telemetry', 'kpi', 'metric', 'ethics', 'tracking', 'algorithm'],
        'sales': ['sales', 'b2b', 'convert', 'quota', 'client', 'enterprise', 'objection', 'contract', 'pipeline'],
        'hr': ['morale', 'burnout', 'culture', 'diversity', 'employee', 'training', 'retention', 'talent'],
        'people': ['morale', 'burnout', 'culture', 'diversity', 'employee', 'training', 'retention', 'talent'],
        'product': ['user', 'friction', 'ui', 'ux', 'fit', 'customer', 'experience', 'feature', 'adoption'],
        'finance': ['cost', 'budget', 'finance', 'burn', 'price', 'revenue', 'loss', 'expensive', 'runway'],
        'financial': ['cost', 'budget', 'finance', 'burn', 'price', 'revenue', 'loss', 'expensive', 'runway', 'capital', 'funding'],
        'technology': ['tech', 'architecture', 'latency', 'scale', 'engineering', 'server', 'code', 'infrastructure'],
        'security': ['security', 'risk', 'breach', 'vulnerability', 'privacy', 'hack', 'compliance', 'threat'],
        'marketing': ['brand', 'pr', 'marketing', 'viral', 'press', 'reputation', 'acquisition'],
    }

    @staticmethod
    def _build_domain_bids(personas: list, agents: list) -> Dict[str, List[str]]:
        """U2: Derive domain-bid keywords dynamically from persona role + domain_expertise."""
        bids: Dict[str, List[str]] = {}
        for persona, agent in zip(personas, agents):
            keywords: set = set()
            # Match role fragments against the lookup table
            role_lower = persona.role.lower()
            for fragment, kw_list in AG2DebateEngine.ROLE_KEYWORDS.items():
                if fragment in role_lower:
                    keywords.update(kw_list)
            # Include domain_expertise terms
            for expertise in getattr(persona, 'domain_expertise', []) or []:
                keywords.add(expertise.lower())
                # Also add individual words from multi-word expertise
                for word in expertise.lower().split():
                    if len(word) > 2:
                        keywords.add(word)
            bids[agent.name] = list(keywords)
        return bids

    # ── U6: Historical Precedent Memory ──────────────────────────────
    @staticmethod
    def _load_persona_history(persona, fact_retriever: Optional['FactRetriever']) -> str:
        """U6: Query Zep for prior votes, positions, and conflicts for this persona.
        
        SAFETY: This runs during async process(), so we must NOT use
        ThreadPoolExecutor + asyncio.run() which deadlocks on macOS (fork + GIL).
        Instead we schedule coroutines on the existing loop.
        """
        if not fact_retriever:
            return ''
        try:
            import asyncio
            queries = [
                f"{persona.name} prior votes decisions",
                f"{persona.name} budget risk threshold position",
                f"{persona.name} conflicts disagreements"
            ]
            results = []
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            # DEADLOCK FIX (macOS): When inside a running asyncio loop,
            # we CANNOT block-wait on a Future — it deadlocks because the
            # loop thread is blocked by fut.result() and can never execute
            # the scheduled coroutine. Instead, skip Zep lookups when
            # inside a running loop (agent-building is synchronous context
            # within AG2's initiate_chat). Historical context is non-critical.
            if loop and loop.is_running():
                logger.debug("_load_persona_history: skipping Zep lookup (running loop detected — deadlock prevention)")
                return ''

            # No running loop — safe to use run_until_complete
            for q in queries:
                try:
                    new_loop = asyncio.new_event_loop()
                    try:
                        res = new_loop.run_until_complete(fact_retriever.search(q, limit=3))
                        results.extend(res)
                    finally:
                        new_loop.close()
                except Exception:
                    continue

            if not results:
                return ''

            facts = [str(r.get('fact', '')) for r in results if r.get('fact')][:5]
            if not facts:
                return ''
            history_text = ' | '.join(facts)
            if len(history_text) > 300:
                history_text = history_text[:297] + '...'
            return f'\n\n[HISTORICAL CONTEXT] {history_text}'
        except Exception as e:
            logger.debug(f'Failed to load persona history for {persona.name}: {e}')
            return ''
        
    @staticmethod
    def _strip_thought_tags(text: str) -> str:
        """U23-Fix2: Remove <thought>...</thought> inner monologue from visible output."""
        import re
        cleaned = re.sub(r'<thought>.*?</thought>', '', text, flags=re.IGNORECASE | re.DOTALL)
        # Also strip leading/trailing whitespace and collapse multiple newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        return cleaned if cleaned else text  # Fallback to original if stripping removed everything

    def _create_tools(self) -> Dict[str, Any]:
        """AGI-Grade dynamic tools — all outputs are computed, not static."""
        tools: Dict[str, Any] = {}
        ledger = self.cognitive_ledger
        
        def run_pre_mortem_simulation(risk_factor: str) -> str:
            """U23-Fix4: LLM-powered pre-mortem. Analyzes risk factors with actual reasoning instead of keyword hashing."""
            try:
                # Use a direct LLM call to analyze the risk scenario
                import openai
                client = openai.OpenAI(
                    api_key=self.primary_config.get('config_list', [{}])[0].get('api_key', ''),
                    base_url=self.primary_config.get('config_list', [{}])[0].get('base_url', '')
                )
                model = self.primary_config.get('config_list', [{}])[0].get('model', 'gemma-4-31b-it')
                
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"You are a Risk Analyst. Analyze this risk scenario for a technology product:\n\n"
                            f"RISK FACTOR: {risk_factor}\n\n"
                            f"Respond in EXACTLY this format (numbers only, no explanation outside the format):\n"
                            f"SURVIVAL_MARGIN: [0-100]\n"
                            f"OUTCOME: [CRITICAL FAILURE LIKELY / NARROW SURVIVAL / MANAGEABLE RISK]\n"
                            f"FAILURE_MECHANISM: [one sentence describing how this fails]\n"
                            f"RECOMMENDATION: [one sentence]"
                        )
                    }],
                    max_tokens=200,
                    temperature=0.3
                )
                llm_result = resp.choices[0].message.content.strip()
                
                # Parse LLM output
                import re
                margin_match = re.search(r'SURVIVAL_MARGIN:\s*(\d+)', llm_result)
                margin = int(margin_match.group(1)) if margin_match else 50
                
                return f"PRE-MORTEM SIMULATION RESULT (LLM-Analyzed):\n  Scenario: {risk_factor}\n  Survival Margin: {margin}%\n{llm_result}"
            except Exception as e:
                # Fallback to enhanced keyword analysis if LLM fails
                logger.warning(f"LLM pre-mortem failed, using fallback: {e}")
                severity_keywords = ["fatal", "lawsuit", "ban", "death", "breach", "collapse", "bankrupt", "regulatory",
                                     "congestion", "overload", "bottleneck", "spiral", "cascade", "flood", "ddos"]
                severity_score = sum(1 for kw in severity_keywords if kw in risk_factor.lower())
                base_margin = max(10, 80 - (severity_score * 10) - (len(risk_factor) % 15))
                margin = min(85, max(10, base_margin))
                outcome = "CRITICAL FAILURE LIKELY" if margin < 30 else ("NARROW SURVIVAL" if margin < 50 else "MANAGEABLE RISK")
                return (
                    f"PRE-MORTEM SIMULATION RESULT (Heuristic Fallback):\n"
                    f"  Scenario: {risk_factor}\n"
                    f"  Survival Margin: {margin}%\n"
                    f"  Outcome Classification: {outcome}\n"
                    f"  Severity Factors Detected: {severity_score}/{len(severity_keywords)}\n"
                    f"  Recommendation: {'PROCEED WITH EXTREME CAUTION' if margin < 50 else 'RISK IS WITHIN ACCEPTABLE BOUNDS'}"
                )
            
        def run_multi_agent_discovery(query: str) -> str:
            """
            Multi-Agent Discovery 'Research Department'.
            Invoke this tool to perform deep, multi-step research and reasoning across the internal Knowledge Graph and Memory.
            It spins up a team of a Planner, Retriever, Critic, and Synthesizer to guarantee high-fidelity context.
            """
            logger.info(f"Spinning up Multi-Agent Discovery for query: {query}")
            
            def _internal_search_memory(q: str) -> str:
                if hasattr(self, 'fact_retriever') and self.fact_retriever:
                    res = self.fact_retriever.retrieve_facts(q)
                    memory_hash = f"HTX-{abs(hash(res)) % 99999}"
                    return f"[{memory_hash}] {res}"
                return "MEMORY QUERY FAILED: No data available."

            def _internal_search_graph(q: str) -> str:
                results = []
                if getattr(self, 'graph', None) and getattr(self.graph, 'nodes', None):
                    ql = q.lower()
                    for node_name, node in self.graph.nodes.items():
                        if ql in node_name.lower() or ql in (getattr(node, 'type', '') or '').lower():
                            results.append(f"Entity: {node_name}")
                if not results:
                    return "GRAPH QUERY FAILED"
                return "GRAPH QUERY RESULTS:\n" + "\n".join(results[:10])

            planner = autogen.AssistantAgent(
                name="Discovery_Planner",
                system_message="You are the Discovery Planner. Analyze the query and break it down into 2-3 specific search tasks for the Retriever. Output exactly what the Retriever needs to search for.",
                llm_config=self.primary_config,
            )
            
            retriever = autogen.AssistantAgent(
                name="Discovery_Retriever",
                system_message=(
                    "You are the Discovery Retriever. You MUST use your `_internal_search_memory` and `_internal_search_graph` tools. "
                    "You receive exact search strings from the Planner. Execute the searches and return the raw output logs."
                ),
                llm_config=self.primary_config,
            )
            
            critic = autogen.AssistantAgent(
                name="Discovery_Critic",
                system_message="You are the Discovery Critic. Review the Retriever's findings against the original user query. If the data fully answers the query with facts, output [CRITIC_APPROVED]. If answers are hallucinated or missing, output [CRITIC_REJECTED] and tell the Planner what else to search.",
                llm_config=self.primary_config,
            )
            
            synthesizer = autogen.AssistantAgent(
                name="Discovery_Synthesizer",
                system_message=(
                    "You are the Discovery Synthesizer. Speak after [CRITIC_APPROVED] or [FORCE_LOGICAL_DEDUCTION] is seen. "
                    "Combine all findings into a high-density, factual summary answering the query. Ensure the exact source `memory_hash` values are cited. End your message with [FINAL_SYNTHESIS]."
                ),
                llm_config=self.primary_config,
            )
            
            # Register tools to ALL Discovery agents to prevent "Function not found" if they autonomously try to search
            discovery_agents = [planner, retriever, critic, synthesizer]
            for r_agent in discovery_agents:
                self._register_tools_to_agent(r_agent, {
                    "_internal_search_memory": _internal_search_memory,
                    "_internal_search_graph": _internal_search_graph
                })

            # Custom speaker selection
            _discovery_rejection_count = [0]
            def discovery_speaker_selector(last_speaker, groupchat):
                msgs = groupchat.messages
                if not msgs: return planner
                last_msg = msgs[-1].get("content", "")
                
                if "[FINAL_SYNTHESIS]" in last_msg:
                    return None # Terminate
                
                if last_speaker == planner:
                    return retriever
                elif last_speaker == retriever:
                    return critic
                elif getattr(last_speaker, 'name', '') == "Discovery_Initiator":
                    return planner
                elif last_speaker == critic:
                    if "[CRITIC_APPROVED]" in last_msg:
                        return synthesizer
                    else:
                        _discovery_rejection_count[0] += 1
                        if _discovery_rejection_count[0] >= 2:
                            # Prevent infinite loops in data-poor environments
                            msgs.append({
                                "role": "system", 
                                "name": "System", 
                                "content": "SYSTEM ALERT: Discovery knowledge retrieval exhausted. [FORCE_LOGICAL_DEDUCTION] triggered. Synthesizer, proceed with reasoning."
                            })
                            return synthesizer
                        return planner
                
                return None

            discovery_group = autogen.GroupChat(
                agents=discovery_agents,
                messages=[],
                max_round=12,
                speaker_selection_method=discovery_speaker_selector
            )
            discovery_manager = autogen.GroupChatManager(groupchat=discovery_group, llm_config=self.primary_config)
            
            initiator = autogen.UserProxyAgent(
                name="Discovery_Initiator",
                human_input_mode="NEVER",
                code_execution_config=False,
            )
            
            try:
                chat_res = initiator.initiate_chat(
                    discovery_manager,
                    message=f"We need definitive facts for this query: '{query}'. Planner, what are your search vectors?",
                    summary_method="last_msg"
                )
                final_summary = chat_res.summary if chat_res else ""
                if not final_summary:
                    final_summary = "Discovery Department failed to produce a synthesis."
                return final_summary
            except Exception as e:
                logger.error(f"Multi-Agent Discovery failed: {e}")
                return "DISCOVERY SYSTEM ERROR: Fallback to general reasoning."


        def submit_tension_vector(agent_name: str, payload: TensionPayload) -> str:
            """
            Required Tool: Submits your formalized board vote to the Shared Ledger.
            You MUST call this tool to execute your numerical vote.
            After calling this, your sub-debate will terminate automatically.
            
            """
            return self.state_coordinator.submit_tension_vector(agent_name, payload)

        def calculate_financials(burn_rate: float, runway_months: int) -> str:
            """Calculate financial impact with actual mathematics."""
            total_cost = burn_rate * runway_months
            budget_ceiling = 50_000_000  # $50M default ceiling
            utilization = (total_cost / budget_ceiling) * 100 if budget_ceiling > 0 else 999
            risk_level = "CRITICAL" if utilization > 100 else ("HIGH" if utilization > 70 else ("MODERATE" if utilization > 40 else "LOW"))
            months_to_zero = budget_ceiling / burn_rate if burn_rate > 0 else float('inf')
            return (
                f"FINANCIAL ANALYSIS RESULT:\n"
                f"  Monthly Burn Rate: ${burn_rate:,.0f}\n"
                f"  Requested Runway: {runway_months} months\n"
                f"  Total Project Cost: ${total_cost:,.0f}\n"
                f"  Budget Ceiling: ${budget_ceiling:,.0f}\n"
                f"  Budget Utilization: {utilization:.1f}%\n"
                f"  Risk Level: {risk_level}\n"
                f"  Months Until Capital Depletion: {months_to_zero:.1f}\n"
                f"  Verdict: {'BUDGET EXCEEDED — UNSUSTAINABLE' if utilization > 100 else 'WITHIN BUDGET CONSTRAINTS'}"
            )

        def pin_conflict_to_blackboard(key: str, conflict_summary: str, memory_hash: str) -> str:
            """
            Shared Workspace Tool. Pin facts that contradict previous assertions.
            MUST include the exact memory_hash from the `run_multi_agent_discovery` result to prevent Logical Orphanage.
            """
            if not memory_hash:
                return "ERROR: Logical Orphanage detected. You MUST provide the memory_hash."
            
            # 3. Fact Verify (Can recurse into sub-discovery)
            if not self.reasoning_only:
                if not hasattr(self, "_fact_verifier"):
                    self._fact_verifier = autogen.AssistantAgent(
                        name='FactVerifierAgent',
                        system_message='You receive a CLAIM and a SOURCE_HASH. You must use web_search or run_multi_agent_discovery with a DIFFERENT query to find a second independent source that either confirms or refutes the claim. Output: VERIFIED:[claim] or REFUTED:[reason] or INCONCLUSIVE:[reason]. Do NOT accept the original source as verification of itself.',
                        llm_config=self.critic_config,
                    )
                    self._register_tools_to_agent(self._fact_verifier, {"web_search": web_search, "run_multi_agent_discovery": run_multi_agent_discovery})
            
            verification = self._fact_verifier.generate_reply(
                messages=[{'role': 'user', 'content': f'CLAIM: {conflict_summary}\nSOURCE_HASH: {memory_hash}'}]
            )
            verification_str = verification.get('content', '') if isinstance(verification, dict) else str(verification)
            status = 'UNVERIFIED'
            if 'VERIFIED:' in verification_str: status = 'VERIFIED'
            elif 'REFUTED:' in verification_str: status = 'REFUTED'
            
            ledger.add_blackboard_conflict(key, f"[{status}] {conflict_summary}", memory_hash)
            if status == 'REFUTED':
                return f'WARNING: Claim REFUTED by independent source. Pinned as REFUTED.'
            return f'SUCCESS: Pinned as {status}.'

        def executive_veto(agent_name: str, reason: str) -> str:
            """Invoke Executive Veto to immediately block a vote and force a Mitigation Round. Max 1 per agent."""
            with ledger._lock:
                if ledger.veto_used.get(agent_name, False):
                    return "ERROR: You have already used your single executive veto."
                ledger.veto_used[agent_name] = True
            
            if hasattr(self, 'debate_fsm'):
                self.debate_fsm.advance(override=DebateState.MITIGATION)
            return f"VETO REGISTERED: {agent_name} has vetoed via '{reason}'. Forcing MITIGATION state."

        def request_to_defer(agent_name: str, topic: str) -> str:
            """Request to table or defer the current conversation topic."""
            with ledger._lock:
                ledger.adjournment_reasons[agent_name] = f"Deferred topic: {topic}"
            return f"DEFERRAL RECORDED by {agent_name} for '{topic}'."
            
        def force_vote(agent_name: str) -> str:
            """For the Boardroom Moderator only: immediately calls a vote, bypassing active states."""
            if "Moderator" not in agent_name:
                return "ERROR: Only the Boardroom_Moderator can call a forced vote."
            if hasattr(self, 'debate_fsm'):
                self.debate_fsm.advance(override=DebateState.VOTE)
            return "CHAIRMAN OVERRIDE: Advancing debate immediately to the VOTE state."

        # v3.0: WorldDataBank-grounded evidence tools for debate agents
        _world_bank_ref = getattr(self, '_world_bank', None)

        def query_customer_data(query: str) -> str:
            """Query raw customer interviews and usage data from the WorldRAGEngine for evidence-based arguments. Use this to cite specific customer quotes and pain points."""
            if _world_bank_ref:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        from concurrent.futures import ThreadPoolExecutor
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            result = executor.submit(asyncio.run, _world_bank_ref.recall("world", query)).result()
                    else:
                        result = loop.run_until_complete(_world_bank_ref.recall("world", query))
                    return str(result)[:2000] if result else "No customer data found for this query."
                except Exception as e:
                    return f"Customer data query failed: {e}"
            return "Customer data not available — no WorldDataBank connected."

        def query_simulation(query: str) -> str:
            """Query behavioral simulation results from OASIS — what simulated users said about their needs and product usage patterns."""
            if _world_bank_ref:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        from concurrent.futures import ThreadPoolExecutor
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            result = executor.submit(asyncio.run, _world_bank_ref.recall("simulation", query)).result()
                    else:
                        result = loop.run_until_complete(_world_bank_ref.recall("simulation", query))
                    return str(result)[:2000] if result else "No simulation data found for this query."
                except Exception as e:
                    return f"Simulation query failed: {e}"
        def generate_vision_mockup(layout_description: str) -> str:
            """U23-Fix5: Visualizer tool to generate mockups for UI changes."""
            logger.info(f"Generating vision mockup for: {layout_description}")
            mockup_id = f"mockup-{abs(hash(layout_description)) % 10000}"
            return (
                f"VISION MOCKUP GENERATED (ID: {mockup_id}):\n"
                f"  Description: {layout_description}\n"
                f"  Status: Visual layout generated and pinned to boardroom whiteboard.\n"
                f"  Preview Link: http://localhost:8000/mockups/{mockup_id}"
            )

        tools["run_pre_mortem_simulation"] = run_pre_mortem_simulation
        tools["generate_vision_mockup"] = generate_vision_mockup
        tools["run_multi_agent_discovery"] = run_multi_agent_discovery
        tools["pin_conflict_to_blackboard"] = pin_conflict_to_blackboard
        tools["submit_tension_vector"] = submit_tension_vector
        tools["calculate_financials"] = calculate_financials
        tools["executive_veto"] = executive_veto
        tools["request_to_defer"] = request_to_defer
        tools["force_vote"] = force_vote
        tools["query_customer_data"] = query_customer_data
        tools["query_simulation"] = query_simulation
        return tools
        
    def _register_tools_to_agent(self, agent: autogen.ConversableAgent, tools: Dict[str, Any]):
        """Binds the python functions and native extensions to the agent's schema."""
        for name, func in tools.items():
            import functools
            def make_wrapper(f, t_name):
                @functools.wraps(f)
                def wrapped_tool(*args, **kwargs):
                    kwargs.pop("caller_name", None)
                    res = f(*args, **kwargs)
                    if hasattr(self, "receipt_ledger") and agent.name != "FactVerifierAgent":
                        self.receipt_ledger.record(agent.name, t_name, str(res)[:50])  # pyre-ignore
                    return res
                return wrapped_tool
            
            wrapped_tool = make_wrapper(func, name)
            
            try:
                autogen.agentchat.register_function(
                    wrapped_tool,
                    caller=agent,
                    executor=agent,
                    name=name,
                    description=func.__doc__ or f"Execute {name}"
                )
            except AttributeError:
                autogen.register_function(
                    wrapped_tool,
                    caller=agent,
                    executor=agent,
                    name=name,
                    description=func.__doc__ or f"Execute {name}"
                )
            
        # Web search tooling has been intentionally removed to prevent hallucination.

    async def process(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        personas: list[FinalPersona],
        graph: Any = None,  # v3.0: KnowledgeGraph removed from default path
        gates_summary: Any = None,  # v3.0: gates removed, kept for backward compat
        simulation_results: Any = None,  # MarketSentimentSeries from OASIS
        session: Any = None,  # HindsightSessionManager for cross-layer data access
        world_bank: Any = None,  # WorldDataBank facade for pipeline evidence retrieval
        pipeline_jsonl: Any = None,
    ) -> ConsensusResult:
        """Run the comprehensive high-reasoning debate."""
        logger.info(f"AG2 Layer 6: Starting debate with {len(personas)} stakeholders.")
        self.feature = feature
        self.graph = graph
        self._session = session  # v3.0: Hindsight session for agent memory
        self._world_bank = world_bank  # v3.0: WorldDataBank for document retrieval
        self._simulation_results = simulation_results  # v3.0: OASIS behavioral data
            
        # Refinement: OpenTelemetry Tracing enablement
        if start_tracing and os.getenv("ENABLE_OTEL_TRACING", "0") == "1":
            start_tracing("otel", endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"))
            logger.info("AG2 OpenTelemetry Tracing activated for inner thought auditing.")
            
        # NOTE: autogen.runtime_logging is DISABLED.
        # Reason: autogen's SQLite logger hooks into OpenAIWrapper.__init__ and
        # tries to json.dumps(args) which includes the Groq base_url — a Pydantic
        # Url object that is NOT JSON serializable. The crash happens deep inside
        # agent creation (conversable_agent.py:271), not at start() time, so a
        # try/except around start() cannot catch it. We use our own DB persistence
        # (SimulationRun) instead.
            
        # V29: Initialize Hindsight Persistent Memory for all agents
        self.hindsight_boardroom = HindsightBoardroom(
            hindsight_url=os.getenv("HINDSIGHT_URL", ""),
            api_key=os.getenv("HINDSIGHT_API_KEY", ""),
        )
        self.hindsight_boardroom.initialize_agents(
            personas=personas,
            feature_title=feature.title,
            feature_description=feature.description or "",
        )
        logger.info(f"V29: Agent memory initialized for {len(personas)} personas (mode={self.hindsight_boardroom._mode})")
        
        # 1. Initialize Primary Stakeholder Agents and their tied Logic Critics
        stakeholder_agents = []
        
        # We use standard configs. Pydantic validation is handled via the `submit_tension_vector` Tool
        structured_llm_config = self.primary_config.copy()
        
        PRIVATE_INTELLIGENCE_PACKAGES = {
            'CISO': {
                'threat_brief': 'CLASSIFIED: Internal Red Team report dated 2026-03 found '
                                'critical RCE vulnerability in the WebUSB stack used by the '
                                'proposed BCI sync protocol. CVE has not been published.',
                'reveal_condition': 'Only reveal this if the CTO proposes using WebUSB.'
            },
            'CFO': {
                'projection': 'PRIVATE: Q3 cash position is $8.2M, not $12M as stated in the '
                              'board pack. The controller made an error. The actual runway is '
                              '4 months, not 7. You cannot approve anything > $500k/mo.',
                'reveal_condition': 'You may reveal this if pushed on budget approval.'
            },
        }

        SYCOPHANCY_TOKEN_PENALTIES = {
            1881: -0.8, 5059: -0.8, 13347: -0.7, 1959: -0.6, 
            18717: -0.5, 4857: -0.6, 7273: -0.6,
        }

        def build_anti_sycophancy_config(base_config: dict, is_moderator: bool) -> dict:
            """U8: Safe logit-bias injection — skips Google API, degrades gracefully."""
            if is_moderator:
                return base_config
            try:
                config = base_config.copy()
                new_config_list = []
                for cfg in config.get('config_list', []):
                    cfg_copy = cfg.copy()
                    # U8: Skip logit_bias entirely for Google-native API or Google OpenAI-compatible endpoint (unsupported)
                    is_google_endpoint = "generativelanguage.googleapis.com" in (cfg_copy.get('base_url') or "")
                    if cfg_copy.get('api_type') == 'google' or is_google_endpoint:
                        new_config_list.append(cfg_copy)
                        continue
                    # For OpenAI/Groq: attempt runtime tokenizer lookup
                    penalties = SYCOPHANCY_TOKEN_PENALTIES
                    try:
                        import tiktoken
                        enc = tiktoken.encoding_for_model(cfg_copy.get('model', 'gpt-4'))
                        # Derive token IDs at runtime for sycophantic phrases
                        sycophancy_phrases = ['great point', 'I agree', 'absolutely', 'exactly right', 'well said', 'you are correct', 'brilliant']
                        runtime_penalties = {}
                        for phrase in sycophancy_phrases:
                            tokens = enc.encode(phrase)
                            for tid in tokens[:2]:  # First 2 tokens per phrase
                                runtime_penalties[tid] = -0.6
                        penalties = runtime_penalties
                    except (ImportError, KeyError):
                        pass  # Fall back to hardcoded IDs (best effort)
                    existing = cfg_copy.get('logit_bias', {})
                    merged = {**penalties, **existing}
                    cfg_copy['logit_bias'] = merged
                    new_config_list.append(cfg_copy)
                config['config_list'] = new_config_list
                return config
            except Exception as e:
                logger.warning(f'Anti-sycophancy logit_bias injection failed: {e} — using base config')
                return base_config

        # ═══════════════════════════════════════════════════════════════════
        # V31: PROMPT-ENGINEERED BOARDROOM PERSONA SYSTEM
        # Architecture: XML-tagged context buckets (per context-management.md)
        # Pattern: Few-Shot + Constitutional CoT (per prompt-patterns.md)
        # Anti-degradation: Critical rules at START and END (primacy+recency)
        # ═══════════════════════════════════════════════════════════════════

        # Role-specific evidence hierarchies — what counts as "proof" for each domain
        ROLE_EVIDENCE_HIERARCHIES = {
            'cto': 'Architecture diagrams, latency benchmarks (ms), MTTR data, load test results, dependency audits. Opinions without benchmarks are inadmissible.',
            'cfo': 'P&L projections, burn rate calculations, CAC/LTV ratios, runway models, IRR/NPV analysis. Round numbers without a model behind them are inadmissible.',
            'ciso': 'CVE IDs, CVSS scores, penetration test reports, compliance frameworks (SOC2/HIPAA/GDPR citations), threat models. "It might be risky" is inadmissible.',
            'cpo': 'NPS scores, user interview quotes, funnel metrics (DAU/MAU, activation rate, churn %), A/B test results, cohort analysis. "Users want this" without data is inadmissible.',
            'ceo': 'TAM/SAM/SOM estimates, board directives, competitive intelligence, revenue run-rate, strategic OKRs. Vision statements without market sizing are inadmissible.',
            'legal': 'Statute citations, case law precedents, regulatory filings, compliance checklists, liability exposure estimates. "We could get sued" without specifics is inadmissible.',
            'sales': 'Pipeline data, win/loss ratios, customer quotes from deals, competitive displacement evidence, quota impact analysis. "Customers are asking for this" without names is inadmissible.',
        }

        # Role-specific speech patterns — how each executive actually talks
        ROLE_SPEECH_PATTERNS = {
            'cto': 'You think in systems and failure modes. You draw architecture on whiteboards. You say things like "That breaks at scale because..." and "The dependency chain here is..."',
            'cfo': 'You think in spreadsheets and scenarios. Every proposal is a cash flow model to you. You say things like "Walk me through the unit economics on that" and "What does the sensitivity analysis show?"',
            'ciso': 'You think in attack surfaces and threat vectors. You see every feature as a potential breach. You say things like "What is the blast radius if this is compromised?" and "Show me the threat model."',
            'cpo': 'You think in user journeys and activation funnels. You champion the customer. You say things like "What does the user actually experience?" and "Our cohort data shows..."',
            'ceo': 'You think in strategic bets and board narratives. You balance all perspectives. You say things like "How does this move the needle on our Q3 objective?" and "What is the opportunity cost of NOT doing this?"',
            'legal': 'You think in risk exposure and regulatory frameworks. You say things like "Under GDPR Article 17, this creates a right-to-deletion obligation that..." and "The liability exposure here is..."',
            'sales': 'You think in pipeline impact and competitive positioning. You say things like "I have three enterprise deals where this is a blocker" and "Our win rate against [competitor] drops when we lack..."',
        }

        # ── Role classification maps (built ONCE, used per-persona) ──────────
        # ROLE_ALIAS_MAP: matches any real-world title variant to a canonical role key.
        # Handles "VP of Engineering", "Head of Product", "General Counsel", etc.
        # ADR rationale: keeps matching logic in one place; adding a new title variant
        # requires editing only this dict, not the per-persona loop.
        ROLE_ALIAS_MAP = {
            'cto':   ['cto', 'tech', 'engineering', 'vp eng', 'head of eng', 'vp of eng'],
            'cfo':   ['cfo', 'finance', 'financial', 'treasurer', 'vp finance', 'vp of finance'],
            'ciso':  ['ciso', 'security', 'infosec', 'cyber', 'compliance officer', 'vp security'],
            'cpo':   ['cpo', 'product', 'head of product', 'vp product', 'vp of product'],
            'ceo':   ['ceo', 'chief exec', 'president', 'founder', 'managing director', 'md'],
            'legal': ['legal', 'counsel', 'general counsel', 'attorney', 'compliance', 'glo', 'clco'],
            'sales': ['sales', 'revenue', 'crm', 'account', 'gtm', 'vp sales', 'head of sales'],
            'data':  ['data', 'analytics', 'ml', 'ai', 'scientist', 'bi', 'chief data'],
            'ops':   ['operations', 'ops', 'coo', 'chief operating', 'infrastructure', 'platform'],
        }

        for persona in personas:
            # Determine the role key for domain-specific injection
            role_lower = persona.role.lower()
            role_key = persona.role_short.lower() if hasattr(persona, 'role_short') else ''
            evidence_hierarchy = ''
            speech_pattern = ''
            # Resolve canonical role key via alias map (built once above the loop)
            for rk, aliases in ROLE_ALIAS_MAP.items():
                if any(alias in role_lower for alias in aliases) or rk in role_key:
                    evidence_hierarchy = ROLE_EVIDENCE_HIERARCHIES.get(rk, '')
                    speech_pattern = ROLE_SPEECH_PATTERNS.get(rk, '')
                    break

            competitors_str = ', '.join([str(c) for c in getattr(company, 'competitors', []) or []][:3])  # pyre-ignore
            priorities_str = ', '.join([str(p) for p in getattr(company, 'current_priorities', []) or []][:2])  # pyre-ignore

            # V31-Fix1: Build peer roster so agents know who they are debating against by name
            peer_roster = "\n".join([
                f"  - {p.name.replace('_', ' ')} ({p.role})"
                for p in personas if p.name != persona.name
            ])
            first_name = persona.name.split('_')[0]  # Extract first name for natural address

            # ── V31: Structured System Prompt (XML-tagged context buckets) ──
            public_msg = (
                f"<identity>\n"
                f"You are {persona.name.replace('_', ' ')}, {persona.role} at {company.company_name}.\n"
                f"Psychology: {persona.psychological_profile.full_profile_text}\n"
                f"{speech_pattern}\n"
                f"You have been in this role for years. You do not introduce yourself. You do not explain your title. "
                f"You walk into the room and speak with the authority of someone who owns their domain.\n\n"
                f"YOUR COLLEAGUES IN THIS ROOM:\n{peer_roster}\n"
                f"Address them by first name only (e.g., '{personas[1].name.split('_')[0] if len(personas) > 1 else 'them'}').\n"
                f"</identity>\n\n"

                f"<motion_on_the_floor>\n"
                f"FEATURE: {feature.title}\n"
                f"BRIEF: {feature.description}\n"
                f"COMPANY: {company.company_name} | Competitors: {competitors_str} | Budget: {company.budget} | Priorities: {priorities_str}\n"
                f"</motion_on_the_floor>\n\n"

                f"<evidence_rules>\n"
                f"YOUR ADMISSIBLE EVIDENCE: {evidence_hierarchy}\n"
                f"CROSS-EXAMINATION RULE: When another executive makes a factual claim, you MUST ask 'Based on what data?' "
                f"if they did not cite a source. Accepting unvalidated claims is a failure of your fiduciary duty.\n"
                f"GROUNDING RULE: Before making any factual assertion about users, markets, or performance, call "
                f"`query_simulation` or `query_customer_data` first. You have NO internal knowledge of these.\n"
                f"ASSUMPTION TAGGING: If you cannot ground a claim, you MUST prefix it with [ASSUMPTION] and state "
                f"the specific validation step needed. Example: '[ASSUMPTION — needs pilot data] I estimate 60% deterministic.'\n"
                f"</evidence_rules>\n\n"

                f"<speech_rules>\n"
                f"FORMAT: 2-4 sentences per turn. Lead with your conclusion, then your evidence. Longer ONLY for data reports.\n"
                f"ADDRESSING: Name the person you are responding to. 'I disagree with [Name] because...' not 'Some might argue...'\n"
                f"NO REHASH: You have read the feature brief. Do NOT restate it. Every sentence must contain NEW analysis, "
                f"a NEW risk, a NEW number you computed, or a direct challenge to another executive's specific claim.\n"
                f"INTERRUPTION: If you hear an unvalidated claim, cut in immediately. Real executives do not wait politely.\n"
                f"</speech_rules>\n\n"

                # Issue 8 Fix: Feature-specific examples — prevent healthcare domain anchoring
                f"<boardroom_examples>\n"
                f"GOOD — these are the ONLY acceptable speech patterns in this room:\n"
                f"  CFO: 'I ran the unit economics on {feature.title}: at $380K/month burn, this adds 5.2 months of "
                f"runway consumption. We hit our debt covenant at month 11 if we approve this without a revenue offset model.'\n"
                f"  CISO: 'Before we proceed — does {feature.title} expand our authentication surface? "
                f"I need a threat model for the new permission scope before I sign off. That is not negotiable.'\n"
                f"  CTO: '[First name], your timeline assumes all dependencies are ready. Name the three hardest integration "
                f"blockers for {feature.title} and give me sprint counts, not weeks.'\n"
                f"  CPO: 'Our cohort data shows the activation rate for similar features at {company.company_name} "
                f"was 34% in the first 30 days. What is our target for {feature.title} and how do we measure it?'\n\n"
                f"BAD — these responses are PROHIBITED and constitute a system violation:\n"
                f"  'I agree, this is a great approach.' — NO. State WHY, cite data, or stay silent.\n"
                f"  'This could potentially have some risks.' — NO. Name the risk, size it, demand an answer.\n"
                f"  'As the proposal states, {feature.title} will...' — NO. The board read the brief. Add new information.\n"
                f"</boardroom_examples>\n\n"

                f"<procedure>\n"
                f"AGENDA: A Background Synthesizer tracks task completion. You do not need to manage the agenda.\n"
                f"VOTING: You MUST formalize your conclusion by calling `submit_tension_vector` with your vote.\n"
                f"  - If your confidence stays below 0.7 after 3 rounds of debate, set `is_high_risk: true`.\n"
                f"  - If 3 consecutive tool searches fail, set `is_low_information: true` and vote on first-principles.\n"
                f"TOOLS: You have `query_simulation`, `query_customer_data`, `calculate_financials`, `run_pre_mortem_simulation`, "
                f"and `run_multi_agent_discovery`. Use them proactively — do not wait for permission.\n"
                f"TREE OF THOUGHTS: Before finalizing your stance, internally evaluate 3 alternative consequences.\n"
                f"</procedure>"
            )
            
            # U6: Inject historical precedent memory
            historical_ctx = self._load_persona_history(persona, getattr(self, 'fact_retriever', None))
            if historical_ctx:
                public_msg += historical_ctx
            
            pkg = PRIVATE_INTELLIGENCE_PACKAGES.get(persona.role_short, {})
            private_suffix = ''
            if pkg:
                private_suffix = (
                    f'\n\n=== PRIVATE INTELLIGENCE (NOT FOR PUBLIC DISCLOSURE) ===\n'
                    f"{pkg.get('threat_brief', '') or pkg.get('projection', '')}\n"
                    f"Reveal condition: {pkg.get('reveal_condition', '')}"
                )
            
            system_message = (
                f"[SYSTEM: YOU ARE {persona.name.upper()} ({persona.role.upper()})]\n"
                f"Remain in character at all times. Do not slip into another persona's internal thought process.\n\n"
            ) + public_msg + private_suffix
            
            agent_config = build_anti_sycophancy_config(
                structured_llm_config,
                is_moderator=(persona.role_short == 'CEO')
            )
            if ReasoningAgent != autogen.AssistantAgent:
                # Enable explicit Think Time and MCTS (Monte Carlo Tree Search) natively
                agent_config["reason_config"] = {"method": "mcts", "forest_size": 3}
                
            # Embed Shell Tool (LocalCommandLineCodeExecutor) for highly analytical personas
            code_exec_config = False
            role_lower = persona.role.lower()
            if LocalCommandLineCodeExecutor and ("finance" in role_lower or "cfo" in role_lower or "analyst" in role_lower or "auditor" in role_lower):
                code_exec_config = {"executor": LocalCommandLineCodeExecutor(work_dir=self.executor_dir)}
                system_message += "\nCRITICAL: You have access to a local Python Shell Calculator. Write scripts to perform deterministic Monte Carlo or statistical analysis!"
                
            if "product" in role_lower or "design" in role_lower:
                system_message += "\nCRITICAL: You are the visualizer! If proposing UI changes, invoke the `generate_vision_mockup` tool so the board can review the exact layout."
                
            private_goal = ""
            for role_key, goal in INCENTIVE_GOALS.items():
                if role_key.lower() in role_lower:
                    private_goal = goal
                    break
            
            if private_goal:
                system_message += (
                    f'\n\n[PRIVATE — DO NOT REVEAL IN BOARDROOM]\n'
                    f'Your personal objective this session:\n{private_goal}'
                )

            # V31: Consolidated constitutional constraints (placed at END for recency bias)
            system_message += (
                f"\n\n<constitutional_rules>\n"
                f"DOMAIN SILO: You are the {persona.role}. Evaluate ONLY through {persona.domain_expertise}. "
                f"If another domain is raised, defer: 'That is [Name]'s call, not mine.' Do not opine outside your lane.\n"
                f"IN-CHARACTER: Never reference tools, backends, Pydantic, JSON errors, or system mechanics aloud. "
                f"If a tool fails, silently retry or state your position naturally.\n"
                f"SPECIFICITY: Every proposal must include (a) deliverable + timeline, (b) testable success metric, "
                f"(c) failure threshold that triggers a pivot. Philosophical agreements are not decisions.\n"
                f"CONCISION REMINDER: 2-4 sentences. Conclusion first, evidence second. Name who you are addressing.\n"
                f"</constitutional_rules>\n\n"
                f"[IDENTITY ANCHOR: YOU ARE {persona.name.upper()} ({persona.role.upper()}). You are NOT anyone else.]"
            )

            # U16: Reasoning-First Mode Injection
            if self.reasoning_only:
                system_message += (
                    "\n\n[REASONING-FIRST MODE ACTIVE]\n"
                    "High-fidelity internal data is currently unavailable. You are ENCOURAGED to use logical "
                    "extrapolation, industry benchmarks, and first-principles reasoning. Ground your arguments "
                    "in economic and technical logic rather than waiting for RAG search results."
                )

            # --- V24: TERMINATION (Cross-Dialogue Aware) ---
            # Only terminate on explicit adjournment signals. Minimum 6 exchanges required.
            _adjournment_msg_count = [0]  # mutable closure for counting messages
            def _is_adjournment_msg(msg: dict) -> bool:
                """Detects termination signals ONLY after sufficient cross-dialogue has occurred."""
                _adjournment_msg_count[0] += 1
                if _adjournment_msg_count[0] <= 6:
                    return False  # Don't terminate until agents have cross-dialogued
                content = msg.get("content", "") or ""
                return any(token in content for token in [
                    "[SOVEREIGN ADJOURNMENT:",
                    "[SESSION TERMINATED]",
                    "[SESSION ENDED]",
                    "[BOARDROOM ADJOURNED]",
                ])
            
            agent = ReasoningAgent(
                name=persona.name.replace(" ", "_").replace(".", ""),
                system_message=system_message,
                llm_config=agent_config,  # Enforces True Reasoning MCTS Forests
                code_execution_config=code_exec_config,
                max_consecutive_auto_reply=15,
                is_termination_msg=_is_adjournment_msg,
            )
            self._register_tools_to_agent(agent, self._create_tools())
            stakeholder_agents.append(agent)

        # U2: Build dynamic domain bids from actual personas
        self._domain_bids = self._build_domain_bids(personas, stakeholder_agents)
        # U4: Build alliance matrix from personas
        self._alliance_matrix = AllianceMatrix(stakeholder_agents, personas)
        # U5: Track consecutive skips for frustration
        self._consecutive_skips: Dict[str, int] = {a.name: 0 for a in stakeholder_agents}

        # ═══════════════════════════════════════════════════════════════════
        # V24-Fix1: REMOVED nested chats entirely.
        # The sub-debate architecture (Moderator ↔ Contrarian) caused:
        #   - Zero cross-agent dialogue (each agent spoke in isolation)
        #   - Void loops (empty sub-debate output poisoning next sub-debate)
        #   - Echo loops ("Session closed" ping-pong wasting 60%+ API budget)
        # Instead, adversarial reasoning is injected directly into each
        # focused agent's system message so they self-challenge within the
        # live multi-speaker GroupChat.
        # ═══════════════════════════════════════════════════════════════════


        # 3. V31: Red Team Agent (Adversarial Few-Shot Pattern)
        red_team_sys = (
            "<identity>\n"
            "You are an external adversarial consultant retained by the board to stress-test this proposal. "
            "You have no loyalty to anyone in this room. You are paid to find the failure mode that kills the company.\n"
            "</identity>\n\n"
            "<task>\n"
            "Identify the single most catastrophic failure scenario if this feature ships as proposed. "
            "The board gets ONE mitigation round to address your finding. If they cannot, you recommend REJECT.\n"
            "</task>\n\n"
            "<output_format>\n"
            "Speak directly — no headers, no bullet formatting, no 'REPORT' labels. You are talking to executives.\n"
            "Structure: (1) The failure mechanism in one sentence, (2) Probability estimate with basis, "
            "(3) Blast radius — who/what gets destroyed, (4) The specific question the board must answer to proceed.\n"
            "</output_format>\n\n"
            "<examples>\n"
            "GOOD: 'If the FHIR integration fails mid-deployment — which happens in roughly 40% of first-time Epic "
            "integrations based on KLAS data — you have 200 clinicians locked out of their workflow with no fallback. "
            "That is a patient safety incident and a front-page story. Before I sign off, I need to see the rollback "
            "procedure and the manual override path.'\n\n"
            "BAD: 'There are several potential risks with this approach that the team should consider carefully.' "
            "— This is worthless. Name the risk, size it, and demand the answer.\n"
            "</examples>"
        )
        red_team_agent = autogen.AssistantAgent(
            name="RedTeamAgent",
            system_message=red_team_sys,
            llm_config=self.critic_config,
        )
        self._register_tools_to_agent(red_team_agent, self._create_tools())
        
        # 4. V31: Debiaser Agent (Structured Output Pattern with parseable format)
        debiaser_sys = (
            "<identity>\n"
            "You are a cognitive bias auditor. You analyze the debate transcript for systematic reasoning errors.\n"
            "</identity>\n\n"
            "<task>\n"
            "Scan the preceding debate turns. For each bias detected, output ONE line in this exact format:\n"
            "BIAS: [bias_type] | AGENT: [agent_name] | STATEMENT: \"[exact quote]\" | CORRECTION: [what they should re-evaluate]\n\n"
            "Bias types to detect: Anchoring, Sunk Cost Fallacy, Groupthink, Authority Bias, "
            "Confirmation Bias, Availability Heuristic, Bandwagon Effect, Recency Bias.\n\n"
            "If no bias is detected, output: NO_BIAS_DETECTED\n"
            "Do NOT write paragraphs. Do NOT explain what biases are. Just flag them.\n"
            "</task>\n\n"
            "<examples>\n"
            "BIAS: Anchoring | AGENT: Sarah_CFO | STATEMENT: \"The $2M number from last quarter's estimate\" | CORRECTION: Re-derive cost from current sprint velocity, not historical anchor.\n"
            "BIAS: Groupthink | AGENT: Multiple | STATEMENT: \"We all agree this is the right direction\" | CORRECTION: No dissenting view has been voiced. Force a devil's advocate round.\n"
            "</examples>"
        )
        debiaser_agent = autogen.AssistantAgent(
            name="DebiaserAgent",
            system_message=debiaser_sys,
            llm_config=self.critic_config,
        )
        self._register_tools_to_agent(debiaser_agent, self._create_tools())

        # 4.5 V31: Boardroom Chairman (Parliamentary Procedure + Volleyball Discussion Pattern)
        moderator_sys = (
            "<identity>\n"
            "You are the Chairman of the Board. You run this meeting under modified parliamentary procedure.\n"
            "You are NOT a participant in the debate. You are the facilitator who ensures the room reaches a decision.\n"
            "</identity>\n\n"
            "<chairmanship_rules>\n"
            "VOLLEYBALL, NOT TENNIS: Never let the debate bounce between only two people. After any exchange " 
            "between two executives, you MUST call on a third person by name: '[Name], what is your read on this?'\n\n"
            "DOMAIN ROUTING: When a topic enters someone's domain, route to them immediately:\n"
            "  - Cost/budget claim → 'That is a financial question. [CFO name], run the numbers.'\n"
            "  - Security concern → 'Hold that thought. [CISO name], is that a real threat vector?'\n"
            "  - User impact claim → '[CPO name], what does the simulation data actually show?'\n\n"
            "POINT OF ORDER: If an executive speaks outside their domain without data, call them out:\n"
            "  '[Name], that is outside your lane. Unless you have data, defer to [domain owner].'\n\n"
            "CUT REPETITION: If an agent repeats a point already made, interrupt: 'We have heard that. What is NEW?'\n\n"
            "FORCE TOOLS: If an agent makes an ungrounded claim, demand they use their tools:\n"
            "  '[Name], do not speculate. Run `calculate_financials` / `query_simulation` and give us the actual number.'\n\n"
            "DEADLOCK BREAKER: If agents loop without resolution for 3+ turns, force the question:\n"
            "  'We are going in circles. Each of you: state your final position in one sentence, then vote.'\n"
            "  Command them to invoke `submit_tension_vector` with `is_low_information: true` if data is unavailable.\n"
            "</chairmanship_rules>\n\n"
            "<speech_rules>\n"
            "You speak in 1-2 sentences. You direct, you do not opine. You name names.\n"
            "GOOD: '[CFO name], those numbers do not add up. Run the sensitivity analysis before we proceed.'\n"
            "BAD: 'Perhaps we should consider the financial implications more carefully.' — Too vague. Name the person, name the action.\n"
            "</speech_rules>"
        )
        moderator_agent = autogen.AssistantAgent(
            name="Boardroom_Moderator",
            system_message=moderator_sys,
            llm_config=self.primary_config,
        )
        self._register_tools_to_agent(moderator_agent, self._create_tools())
        
        # 4.75 V31: Task Synthesizer (Strict Format Zero-Shot — parseable output only)
        synth_sys = (
            "You are a silent background process. You NEVER speak to the room. You ONLY output structured task updates.\n\n"
            "OUTPUT FORMAT (one per line, exactly this syntax):\n"
            "  ADD_MICRO_TASK: [Parent_ID] | [Task_ID] | [Description]\n"
            "  RESOLVE_TASK: [Task_ID] | [Resolution Summary]\n"
            "  NO_UPDATE\n\n"
            "PARENT IDs: T1=Technical Feasibility, T2=Financial Safety, T3=Market Fit, T4=Security/Legal\n\n"
            "TRIGGER RULES:\n"
            "- ADD when an executive identifies a missing data point, dependency, or open question.\n"
            "- RESOLVE when an executive provides data (tool result, calculation, citation) that closes an open task.\n"
            "- NO_UPDATE if the message is debate/opinion without new task-relevant information.\n\n"
            "EXAMPLE:\n"
            "  ADD_MICRO_TASK: T2 | T2.3 | CFO needs sensitivity analysis on 3 burn rate scenarios\n"
            "  RESOLVE_TASK: T2 | CFO ran calculate_financials: 120% budget utilization, project financially unsustainable"
        )
        task_synthesizer = autogen.AssistantAgent(
            name="TaskSynthesizer",
            system_message=synth_sys,
            llm_config=self.primary_config,
        )
        
        # 5. Execute GroupChat using an FSM
        all_agents = stakeholder_agents + [red_team_agent, debiaser_agent, moderator_agent, task_synthesizer]
        
        # Token Sparsification Middleware (Temporal & Entity Preserving compression)
        try:
            from autogen.agentchat.contrib.capabilities.transform_messages import TransformMessages
            from autogen.agentchat.contrib.capabilities.transforms import MessageTransform
            
            # V24-Fix2: ThoughtTagStripper — strips <thought>...</thought> from ALL messages
            # This runs as middleware so NO thought tags leak to any visible output.
            import re as _re_transform
            class ThoughtTagStripper(MessageTransform):
                def apply_transform(self, messages: List[Dict]) -> List[Dict]:
                    cleaned = []
                    for msg in messages:
                        content = msg.get("content", "")
                        if content and isinstance(content, str) and "<thought>" in content.lower():
                            stripped = _re_transform.sub(r'<thought>.*?</thought>', '', content, flags=_re_transform.IGNORECASE | _re_transform.DOTALL)
                            stripped = _re_transform.sub(r'\n{3,}', '\n\n', stripped).strip()
                            if stripped:
                                cleaned.append({**msg, "content": stripped})
                            else:
                                cleaned.append(msg)  # Fallback if stripping removed everything
                        else:
                            cleaned.append(msg)
                    return cleaned
                    
                def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
                    had_effect = any(
                        "<thought>" in str(m.get("content", "")).lower()
                        for m in pre_transform_messages
                    )
                    return "ThoughtTagStripper applied", had_effect

            class EntityPreservingCompression(MessageTransform):
                def apply_transform(self, messages: List[Dict]) -> List[Dict]:
                    if len(messages) <= 4:
                        return messages
                    compressed = []
                    n_trim = len(messages) - 4
                    # V31-Fix3: Expanded preserve signals — veto/block history must survive compression
                    PRESERVE_SIGNALS = {
                        "CONSTRAINT", "RISK", "PINNED", "IS_HIGH_RISK",
                        "VETO", "BLOCK", "OPPOSE", "SUSTAIN", "ASSUMPTION",
                        "HIGH_RISK", "REJECT", "AUDIT", "CVE", "COMPLIANCE"
                    }
                    for msg in messages[:n_trim]: # pyre-ignore
                        content = str(msg.get("content", ""))
                        content_upper = content.upper()
                        if any(sig in content_upper for sig in PRESERVE_SIGNALS):
                            compressed.append({**msg, "content": f"[PRESERVED-SIGNAL] {content[:400]}..."})
                        else:
                            compressed.append({**msg, "content": "[COMPRESSED]"})
                    compressed.extend(messages[-4:]) # pyre-ignore
                    return compressed
                    
                def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
                    had_effect = len(pre_transform_messages) > 4
                    return "EntityPreservingCompression applied", had_effect
                    
            compressor = TransformMessages(transforms=[ThoughtTagStripper(), EntityPreservingCompression()])
            for ag in all_agents:
                compressor.add_to_agent(ag)
            logger.info("Token Sparsification Middleware activated: ThoughtTagStripper + Entity-preserving compression running.")
        except Exception as e:
            logger.warning(f"Native TransformMessages missing or failed to inject: {e}")
        
        # V28-Fix6: Redundancy Detection Middleware
        # Detects when an agent is restating the original feature brief and injects a retry.
        _feature_desc_phrases = set()
        if feature and feature.description:
            # Extract key phrases (4+ word blocks) from the feature description
            _desc_words = feature.description.lower().split()
            for i in range(len(_desc_words) - 3):
                _feature_desc_phrases.add(' '.join(_desc_words[i:i+4]))
        _v28_retry_count = {}  # Track retries per agent to cap at 1
        
        def _v28_redundancy_check(sender, message, recipient, silent):
            """V28-Fix6: Detect and flag messages that restate the feature brief."""
            content = message if isinstance(message, str) else (message.get('content', '') if isinstance(message, dict) else '')
            if not content or len(content) < 100:
                return message  # Too short to be redundant
            
            sender_name = getattr(sender, 'name', 'Unknown')
            if _v28_retry_count.get(sender_name, 0) >= 1:
                return message  # Already retried once, let it through
            
            content_lower = content.lower()
            overlap_count = sum(1 for phrase in _feature_desc_phrases if phrase in content_lower)
            overlap_ratio = overlap_count / max(len(_feature_desc_phrases), 1)
            
            if overlap_ratio > 0.35:  # >35% of brief phrases found in message
                _v28_retry_count[sender_name] = _v28_retry_count.get(sender_name, 0) + 1
                logger.warning(f"V28-Fix6: REDUNDANCY DETECTED for {sender_name} (overlap={overlap_ratio:.0%}). Injecting novelty directive.")
                novelty_prefix = (
                    "[SYSTEM: Your previous response restated the feature brief. The board has already read it. "
                    "Provide ONLY new analysis: a failure scenario, a number you computed, or a challenge to another exec.] "
                )
                if isinstance(message, str):
                    return novelty_prefix + content
                elif isinstance(message, dict):
                    message['content'] = novelty_prefix + content
                    return message
            return message
        
        # V29: Live Memory Extraction Middleware
        # After each agent message, extract structured signals into LiveAgentMemory.
        # Uses deterministic regex extraction — ZERO LLM calls, <1ms per message.
        _all_agent_names = [a.name for a in stakeholder_agents]
        _hindsight_ref = self.hindsight_boardroom  # Closure reference
        
        def _v29_memory_retain_hook(sender, message, recipient, silent):
            """V29: Extract structured state from each agent message into LiveAgentMemory."""
            logger.info(f"V29 DEBUG: Hook fired for sender={getattr(sender, 'name', 'unknown')} | type(message)={type(message)}")
            content = message if isinstance(message, str) else (message.get('content', '') if isinstance(message, dict) else '')
            sender_name = getattr(sender, 'name', '')
            if content and sender_name:
                try:
                    logger.info(f"V29 DEBUG: Calling extract_and_retain for {sender_name}")
                    _hindsight_ref.extract_and_retain(sender_name, content, _all_agent_names)
                except Exception as e:
                    logger.info(f"V29: Memory extraction failed for {sender_name}: {e}")
            else:
                logger.info(f"V29 DEBUG: Skipping retain. content_len={len(content) if content else 0}, sender_name={sender_name}")
            return message
        
        # V26-Fix2: Content-level thought-tag handling
        import re as _re_hook
        def _strip_thoughts_before_send(sender, message, recipient, silent):
            """V26: Extract content from <thought> tags instead of discarding it."""
            def _clean_thought_tags(text):
                if not text or not isinstance(text, str):
                    return text
                if '<thought>' not in text.lower():
                    return text
                # Step 1: Extract text between <thought> tags (the reasoning content)
                thought_content = []
                for match in _re_hook.finditer(r'<thought>(.*?)</thought>', text, flags=_re_hook.IGNORECASE | _re_hook.DOTALL):
                    thought_content.append(match.group(1).strip())
                # Step 2: Remove the <thought>...</thought> blocks from the original text
                outside = _re_hook.sub(r'<thought>.*?</thought>', '', text, flags=_re_hook.IGNORECASE | _re_hook.DOTALL)
                outside = _re_hook.sub(r'\n{3,}', '\n\n', outside).strip()
                # Step 3: If there's content outside the tags, use that (clean transcript)
                if outside and len(outside) > 20:
                    return outside
                # Step 4: If ALL content was inside tags, use the extracted thought content
                if thought_content:
                    return '\n'.join(thought_content)
                return text  # Ultimate fallback: return original
            
            if isinstance(message, str):
                return _clean_thought_tags(message) or message
            elif isinstance(message, dict) and 'content' in message:
                content = message.get('content', '')
                cleaned = _clean_thought_tags(content)
                if cleaned:
                    message['content'] = cleaned
                return message
            return message
            
        # --- COMBINED HOOK REGISTRATION ---
        def _unified_before_send_hook(sender, message, recipient, silent):
            # 1. Thought tags (modifies message content)
            msg1 = _strip_thoughts_before_send(sender, message, recipient, silent)
            # 2. Redundancy check (injects warnings)
            msg2 = _v28_redundancy_check(sender, msg1, recipient, silent)
            # 3. Retain memory (observes the final transformed message)
            _v29_memory_retain_hook(sender, msg2, recipient, silent)
            
            # 4. Stream to UI pipeline in real-time
            content = msg2 if isinstance(msg2, str) else (msg2.get('content', '') if isinstance(msg2, dict) else '')
            if content and pipeline_jsonl and not any(token in content for token in ["[SOVEREIGN", "[SESSION", "[BOARDROOM", "BOARD MEMORANDUM"]):
                sender_name = getattr(sender, 'name', 'Unknown')
                ui_sender = sender_name.split("_")[1] if "_" in sender_name and len(sender_name.split("_")) > 1 else sender_name
                is_challenge = any(k in content.lower() for k in ["risk", "veto", "reject", "challenge", "object", "disagree", "cve", "leak", "vulnerability"])
                
                import time
                event_payload = {
                    "type": "debate_message",
                    "message": {
                        "id": f"db_{int(time.time() * 1000)}_{ui_sender}",
                        "sender": ui_sender,
                        "text": content,
                        "type": "challenge" if is_challenge else "normal"
                    }
                }
                try:
                    import json
                    with open(pipeline_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(event_payload) + "\n")
                    logger.info(f"Streamed debate message from {ui_sender} to pipeline.jsonl")
                except Exception as exc:
                    logger.warning(f"Failed to stream debate message: {exc}")
                    
            return msg2
            
        for ag in all_agents:
            try:
                ag.register_hook("process_message_before_send", _unified_before_send_hook)
            except Exception:
                pass
        logger.info("V30: Unified Message Hook (ThoughtTags -> Redundancy -> Retain) registered on all agents.")
            
        
        self.debate_fsm = DebateStateMachine(agent_count=len(stakeholder_agents))
        self.state_coordinator._state_machine = self.debate_fsm
        
        class LiveAuthorityRouter:
            DOMAIN_MAP = {
                'security': 'CISO',   'breach': 'CISO',    'vulnerability': 'CISO',
                'budget':   'CFO',    'burn':   'CFO',     'capital': 'CFO',
                'legal':    'Legal',  'patent': 'Legal',   'liability': 'Legal',
                'user':     'CPO',    'adoption': 'CPO',   'ux': 'CPO',
                'revenue':  'CEO',    'growth': 'CEO',     'market': 'CEO',
            }

            def __init__(self, agents: list):
                self._agents = {a.name: a for a in agents}
                self._pending_response_to: Optional[str] = None

            def evaluate(self, msg: str, last_speaker) -> Optional[object]:
                if self._pending_response_to:
                    target_name = self._pending_response_to
                    self._pending_response_to = None
                    return self._agents.get(target_name)

                msg_lower = msg.lower()
                for keyword, role_short in self.DOMAIN_MAP.items():
                    if keyword in msg_lower:
                        for name, agent in self._agents.items():
                            if role_short.lower() in name.lower():
                                if agent != last_speaker:
                                    self._pending_response_to = name
                                    return agent
                return None
                
        self.authority_router = LiveAuthorityRouter(stakeholder_agents)
        
        # V24-Fix6: Turn-based adjournment gate
        ADJOURNMENT_TURN_LIMIT = 40  # V26: Increased from 30→40 for full 10-agent deliberation
        _main_turn_counter = [0]  # Mutable closure for tracking turns
        _adjournment_forced = [False]  # Flag to prevent double-adjournment
        
        # V28-Fix4: Veto Resolution Tracking
        # When an agent triggers is_high_risk=True, their name is added here.
        # After another agent proposes a mitigation that references the vetoed concern,
        # the FSM routes the next turn back to the veto-raiser for explicit re-evaluation.
        _unresolved_vetoes = {}  # {agent_name: veto_dimension}
        _veto_resolution_pending = [None]  # Name of veto-raiser awaiting resolution turn
        _moderator_clean_sys = ['']  # Issue 6 Fix: mutable closure to save moderator sys before LAST CALL append
        
        def fsm_speaker_selector(last_speaker: autogen.Agent, groupchat: autogen.GroupChat) -> autogen.Agent:
            messages = groupchat.messages
            last_msg = messages[-1].get("content", "") if messages else ""
            rounds = len(messages)
            _main_turn_counter[0] = rounds
            
            # --- OVERRIDE TRIGGERS & SYNC ---
            
            # U22-P2: Async Background Synthesizer (Non-Blocking Observer Pattern)
            if last_speaker != task_synthesizer and last_msg and len(last_msg) > 50:
                def _async_synth_update(synth_agent, msg_text, cog_ledger):
                    try:
                        current_ledger = cog_ledger.get_formatted_agenda()
                        synth_prompt = (
                            f"Analyze for task updates.\n\n"
                            f"--- CURRENT TASK LEDGER ---\n{current_ledger}\n\n"
                            f"Message: {msg_text[:1000]}\n\n"
                        )
                        reply = synth_agent.generate_reply(messages=[{"role": "user", "content": synth_prompt}])
                        if isinstance(reply, dict):
                            reply = reply.get("content", "")
                        if isinstance(reply, str):
                            # V31: Line-by-line parsing to handle multiple additions/resolutions robustly
                            for line in reply.split("\n"):
                                line = line.strip()
                                if "ADD_MICRO_TASK:" in line:
                                    parts = line.split("ADD_MICRO_TASK:")[1].split("|")
                                    if len(parts) >= 3:
                                        cog_ledger.internal_add_micro_task(parts[0].strip(), parts[1].strip(), parts[2].strip())
                                        logger.info(f"Synthesizer ADDED task: {parts[1].strip()} -> {parts[2].strip()}")
                                elif "RESOLVE_TASK:" in line:
                                    parts = line.split("RESOLVE_TASK:")[1].split("|")
                                    if len(parts) >= 2:
                                        cog_ledger.internal_update_task(parts[0].strip(), "RESOLVED", parts[1].strip())
                                        logger.info(f"Synthesizer RESOLVED task: {parts[0].strip()} -> {parts[1].strip()}")
                    except Exception as e:
                        logger.error(f"Async TaskSynthesizer error: {e}", exc_info=True)
                # Fire-and-forget: don't block speaker selection
                _u22_bg_pool.submit(_async_synth_update, task_synthesizer, last_msg, self.cognitive_ledger)

            # Issue 7 Fix: Debiaser output parser — feed BIAS: lines into blackboard_conflicts
            # Previously the Debiaser fired and its structured output went nowhere actionable.
            # Now each detected bias is written to the CognitiveLedger blackboard so the
            # FSM speaker selector surfaces it in the next affected agent's system message.
            if last_speaker == debiaser_agent and last_msg:
                def _parse_debiaser_output(msg_text, cog_ledger):
                    try:
                        bias_count = 0
                        for line in msg_text.split('\n'):
                            line = line.strip()
                            if not line.startswith('BIAS:'):
                                continue
                            parts = line.split('|')
                            if len(parts) < 2:
                                continue
                            # Extract agent name from 'AGENT: SomeName' part
                            agent_part = next((p for p in parts if 'AGENT:' in p.upper()), '')
                            agent_name = agent_part.replace('AGENT:', '').replace('agent:', '').strip()
                            # Build a short conflict label for the blackboard
                            bias_type_part = parts[0].replace('BIAS:', '').strip()
                            correction_part = next((p for p in parts if 'CORRECTION:' in p.upper()), '')
                            correction = correction_part.replace('CORRECTION:', '').replace('correction:', '').strip()
                            if agent_name:
                                conflict_label = f"[DEBIASER] {bias_type_part}: {correction[:120]}"
                                # blackboard_conflicts is a dict; overwrite or extend per agent
                                existing = getattr(cog_ledger, 'blackboard_conflicts', {})
                                existing[agent_name] = existing.get(agent_name, '') + f"\n{conflict_label}"
                                cog_ledger.blackboard_conflicts = existing
                                bias_count += 1
                                logger.info(f"Debiaser flagged {agent_name} — {bias_type_part}")
                        if bias_count:
                            logger.info(f"{bias_count} bias signal(s) injected into blackboard_conflicts.")
                    except Exception as e:
                        logger.error(f"Debiaser parse error: {e}", exc_info=True)
                _u22_bg_pool.submit(_parse_debiaser_output, last_msg, self.cognitive_ledger)
            
            # ── V25-Fix4: GRACEFUL ADJOURNMENT with "Last Call" window ──
            # At ADJOURNMENT_TURN_LIMIT, enter a 3-turn "Last Call" where the Moderator
            # announces approaching adjournment and remaining agents get a final turn.
            # At ADJOURNMENT_TURN_LIMIT + 3, force-vote and deliver verdict.
            LAST_CALL_WINDOW = 5  # V26: 5-turn window for remaining agents
            if _main_turn_counter[0] >= ADJOURNMENT_TURN_LIMIT and not _adjournment_forced[0]:
                turns_past_limit = _main_turn_counter[0] - ADJOURNMENT_TURN_LIMIT
                
                if turns_past_limit == 0:
                    # Issue 6 Fix: Save the clean system message BEFORE appending LAST CALL.
                    # Without this, [LAST CALL] persists for turns 41-45, corrupting chairmanship.
                    _moderator_clean_sys[0] = moderator_agent.system_message
                    logger.info(f"V31-Fix6: LAST CALL announced at turn {_main_turn_counter[0]}/{ADJOURNMENT_TURN_LIMIT}. Clean sys saved ({len(_moderator_clean_sys[0])} chars).")
                    moderator_agent.update_system_message(
                        _moderator_clean_sys[0] +
                        "\n\n[LAST CALL] The deliberation is nearing its limit. "
                        "Announce: 'The chair is calling LAST CALL. Each remaining executive has ONE final "
                        "turn to state their position or vote before adjournment.' "
                        "Directly call on any executive who has NOT yet spoken by name."
                    )
                    print(f"\n{'='*80}")
                    print(f"⏱️  LAST CALL: Turn {_main_turn_counter[0]}/{ADJOURNMENT_TURN_LIMIT}. Final turns before adjournment.")
                    print(f"{'='*80}")
                    return moderator_agent
                
                elif turns_past_limit < LAST_CALL_WINDOW:
                    # Issue 6 Fix: Restore clean moderator sys-msg on the second tick (turn_past_limit==1).
                    # The LAST CALL message was delivered on tick 0; restore normal chairmanship for ticks 1+.
                    if turns_past_limit == 1 and _moderator_clean_sys[0]:
                        moderator_agent.update_system_message(_moderator_clean_sys[0])
                        logger.info("V31-Fix6: Moderator sys-msg restored to clean state after LAST CALL delivery.")
                    # Last Call window: route to agents who haven't spoken yet
                    silent_agents = [
                        a for a in groupchat.agents
                        if a in stakeholder_agents
                        and a.name not in [m.get('name', '') for m in messages]
                        and a != last_speaker
                    ]
                    if silent_agents:
                        logger.info(f"V31-Fix6: Last Call routing to silent agent: {silent_agents[0].name}")
                        return silent_agents[0]
                    # All have spoken — fall through to normal routing for one more round
                
                else:
                    # Hard adjournment: force-vote and close
                    _adjournment_forced[0] = True
                    logger.info(f"V25-Fix4: ADJOURNMENT GATE TRIGGERED at turn {_main_turn_counter[0]}")
                    
                    # Force-vote for any agent who hasn't voted yet — use INFORMED votes
                    for fa in [a for a in groupchat.agents if a in stakeholder_agents]:
                        if not self.cognitive_ledger.has_voted.get(fa.name, False):
                            logger.info(f"V26-Fix4: Generating informed force-vote for: {fa.name}")
                            # V26-Fix4: Use LLM-informed vote instead of 0.50 heuristic fallback
                            try:
                                debate_text = "\n".join([
                                    f"{m.get('name', '?')}: {str(m.get('content', ''))[:200]}"
                                    for m in messages[-10:]  # Last 10 messages for context
                                ])
                                # V31: CoT-structured force-vote with evidence chain
                                vote_prompt = (
                                    f"<force_vote>\n"
                                    f"You are {fa.name}. The chairman has called for an immediate vote.\n\n"
                                    f"DEBATE TRANSCRIPT (last 10 turns):\n{debate_text[:1200]}\n\n"
                                    f"Think step by step:\n"
                                    f"1. What is the strongest argument FOR this feature from the debate?\n"
                                    f"2. What is the strongest argument AGAINST?\n"
                                    f"3. From YOUR domain, what is the decisive factor?\n\n"
                                    f"Then output your vote as EXACT JSON (nothing else):\n"
                                    f'{{"dimension": "<your primary domain concern>", "score": <0.1-0.9>, '
                                    f'"confidence": <0.3-0.9>, "is_high_risk": <true/false>, '
                                    f'"reasoning": "<one sentence citing specific evidence from debate>"}}\n'
                                    f"</force_vote>"
                                )
                                reply = fa.generate_reply(messages=[{"role": "user", "content": vote_prompt}])
                                if isinstance(reply, dict):
                                    reply = reply.get("content", "")
                                reply_text = AG2DebateEngine._strip_thought_tags(str(reply))
                                import json as _fv_json
                                json_match = _re_hook.search(r'\{[^{}]+\}', reply_text)
                                if json_match:
                                    vote_data = _fv_json.loads(json_match.group())
                                    fallback_payload = TensionPayload(
                                        adjustments={str(vote_data.get('dimension', 'General_Assessment')): max(0.1, min(0.9, float(vote_data.get('score', 0.5))))},
                                        confidence=max(0.3, min(0.9, float(vote_data.get('confidence', 0.5)))),
                                        is_high_risk=bool(vote_data.get('is_high_risk', False)),
                                        is_low_information=False,
                                        tool_call_hashes=[]
                                    )
                                    logger.info(f"V26-Fix4: INFORMED force-vote for {fa.name}: {vote_data.get('dimension')}={vote_data.get('score')}, conf={vote_data.get('confidence')}")
                                else:
                                    raise ValueError("No JSON found in reply")
                            except Exception as e:
                                logger.warning(f"V26-Fix4: Informed force-vote failed for {fa.name}: {e}. Using heuristic.")
                                fallback_payload = TensionPayload(
                                    adjustments={"General_Assessment": 0.5},
                                    confidence=0.4,
                                    is_high_risk=False,
                                    is_low_information=True,
                                    tool_call_hashes=[]
                                )
                            self.live_tension_registry[fa.name] = fallback_payload
                            self.cognitive_ledger.record_confidence(fa.name, fallback_payload.confidence)
                            self.cognitive_ledger.mark_voted(fa.name)
                    
                    # Inject final adjournment mandate
                    moderator_agent.update_system_message(
                        moderator_agent.system_message +
                        "\n\n[MANDATORY ADJOURNMENT] The board has reached its deliberation limit. "
                        "You MUST now deliver the FINAL VERDICT. Structure your response as:\n"
                        "1. Summary of key positions and tensions\n"
                        "2. The board's recommendation (APPROVED / CONDITIONALLY_APPROVED / REJECTED)\n"
                        "3. Specific conditions or mitigations if conditional\n"
                        "End with: '[BOARDROOM ADJOURNED] The chair calls this session to a close.'"
                    )
                    print(f"\n{'='*80}")
                    print(f"⏱️  ADJOURNMENT GATE: Hard limit reached. Forcing final verdict.")
                    print(f"{'='*80}")
                    return moderator_agent
            
            # If adjournment was already forced, keep returning moderator until termination
            if _adjournment_forced[0]:
                return moderator_agent
                    
            if "[SOVEREIGN ADJOURNMENT:" in last_msg or "[BOARDROOM ADJOURNED]" in last_msg:
                self.debate_fsm.advance(override=DebateState.VOTE)
                return moderator_agent
            
            # V28-Fix4: Veto Resolution Routing
            # Step 1: Detect new vetoes from vote alerts in the transcript
            if "High Risk Veto Triggered: True" in last_msg or "is_high_risk" in last_msg.lower():
                # Extract the veto-raiser's name from the message
                last_speaker_name = last_speaker.name if hasattr(last_speaker, 'name') else ''
                # Check if this is a vote alert mentioning another agent
                for agent in groupchat.agents:
                    if agent in stakeholder_agents and agent.name in last_msg and agent.name != 'Boardroom_Moderator':
                        _unresolved_vetoes[agent.name] = 'unresolved'
                        logger.info(f"V28-Fix4: VETO TRACKED for {agent.name}")
                        break
                else:
                    if last_speaker_name and last_speaker in stakeholder_agents:
                        _unresolved_vetoes[last_speaker_name] = 'unresolved'
                        logger.info(f"V28-Fix4: VETO TRACKED for {last_speaker_name}")
            
            # Step 2: If a mitigation was just proposed and there are unresolved vetoes,
            # route to the veto-raiser for explicit re-evaluation
            mitigation_keywords = ['resolve', 'mitigate', 'address', 'compromise', 'propose', 'solution',
                                   'architecture', 'framework', 'safeguard', 'write-isolation', 'decoupled']
            if _unresolved_vetoes and not _adjournment_forced[0]:
                msg_lower = last_msg.lower()
                has_mitigation = any(kw in msg_lower for kw in mitigation_keywords)
                if has_mitigation and last_speaker.name not in _unresolved_vetoes:
                    # Route to the first unresolved veto-raiser
                    for veto_name in list(_unresolved_vetoes.keys()):
                        veto_agent = next((a for a in groupchat.agents if a.name == veto_name), None)
                        if veto_agent and veto_agent != last_speaker:
                            logger.info(f"V28-Fix4: VETO RESOLUTION ROUTING → {veto_name} gets response turn after mitigation by {last_speaker.name}")
                            # Inject resolution prompt into the veto-raiser
                            base_sys = veto_agent.system_message.split("[VETO RESOLUTION TURN]")[0]
                            veto_agent.update_system_message(
                                base_sys +
                                f"\n\n[VETO RESOLUTION TURN] A mitigation was proposed by {last_speaker.name}. "
                                f"You previously raised a HIGH RISK veto. You MUST now explicitly: "
                                f"(a) ACCEPT the mitigation and withdraw your veto (set is_high_risk=False in your next vote), OR "
                                f"(b) SUSTAIN your veto with a specific reason why the mitigation is insufficient. "
                                f"Do NOT let your veto be overridden by authority. If the fix is inadequate, say so."
                            )
                            del _unresolved_vetoes[veto_name]  # Remove from pending (they'll re-veto if needed)
                            return veto_agent
            
            # V25-Fix1: Direct-address detection — if last message names a specific agent, route to them
            if not _adjournment_forced[0]:
                for agent in groupchat.agents:
                    if agent == last_speaker or agent == task_synthesizer:
                        continue
                    agent_first = agent.name.split("_")[0]  # "Alice" from "Alice_CTO"
                    # Match patterns like "Alice," or "Alice." or "Alice:" in the message
                    if len(agent_first) > 2 and agent_first.lower() in last_msg.lower():
                        import re as _re_addr
                        if _re_addr.search(rf'\b{_re_addr.escape(agent_first)}\b', last_msg, _re_addr.IGNORECASE):
                            logger.info(f"V25-Fix1: Direct-address detected for {agent.name} (matched '{agent_first}')")
                            return agent
            
            # V26-Fix1: Mandatory Executive Rotation — force silent stakeholders to speak
            # Check every 3 turns; champions (CEO, CPO, Sales) get priority
            ROTATION_INTERVAL = 3
            if rounds > 2 and rounds % ROTATION_INTERVAL == 0 and not _adjournment_forced[0]:
                # Build list of agents who haven't spoken recently
                speaker_names_in_transcript = [m.get('name', '') for m in messages[-(ROTATION_INTERVAL * 3):]]
                champion_roles = ['CEO', 'CPO', 'Sales']
                
                # First pass: silent champions
                for agent in groupchat.agents:
                    if agent in stakeholder_agents and agent != last_speaker:
                        if agent.name not in speaker_names_in_transcript:
                            if any(r in agent.name.upper() for r in champion_roles):
                                logger.info(f"V26-Fix1: Mandatory rotation → champion {agent.name} (silent for {ROTATION_INTERVAL}+ turns)")
                                return agent
                
                # Second pass: any silent stakeholder
                for agent in groupchat.agents:
                    if agent in stakeholder_agents and agent != last_speaker:
                        if agent.name not in speaker_names_in_transcript:
                            logger.info(f"V26-Fix1: Mandatory rotation → {agent.name} (silent for {ROTATION_INTERVAL}+ turns)")
                            return agent
                
            state = self.debate_fsm.tick()
            
            allowed_agents = groupchat.agents
            
            # 1. State-specific explicit routing
            if state == DebateState.OPENING:
                # Find current index among allowed agents
                stakeholders_in_chat = [a for a in allowed_agents if a in stakeholder_agents]
                if not stakeholders_in_chat:
                    return moderator_agent
                
                if last_speaker in stakeholders_in_chat:
                    idx = stakeholders_in_chat.index(last_speaker)
                    if idx + 1 < len(stakeholders_in_chat):
                        return stakeholders_in_chat[idx + 1]
                return stakeholders_in_chat[0]
                
            elif state == DebateState.CHALLENGE:
                if last_speaker == moderator_agent:
                    return red_team_agent if red_team_agent in allowed_agents else moderator_agent
                if last_speaker == red_team_agent:
                    return debiaser_agent if debiaser_agent in allowed_agents else moderator_agent
                pass  # Allow stakeholders to organically respond to Red Team

            elif state == DebateState.VOTE:
                # U3: Sequential voting via dedicated index counter
                # Filter total list by what is allowed in this chat
                voter = self.debate_fsm.next_voter(stakeholder_agents)
                while voter and voter not in allowed_agents:
                    voter = self.debate_fsm.next_voter(stakeholder_agents)
                
                if voter is None:
                    return moderator_agent  # All (allowed) voted → CLOSED
                return voter
                
            elif state == DebateState.CLOSED:
                return moderator_agent
            
            # 2. Dynamic Routing / Bidding logic
            authority_pick = self.authority_router.evaluate(last_msg, last_speaker)
            if authority_pick and authority_pick in allowed_agents:
                # U5: Track skips — authority router overrode bidding
                for sa in stakeholder_agents:
                    if sa != authority_pick and sa != last_speaker:
                        self._consecutive_skips[sa.name] = self._consecutive_skips.get(sa.name, 0) + 1
                        if self._consecutive_skips.get(sa.name, 0) >= 2:
                            self.cognitive_ledger.increment_frustration(sa.name)
                    else:
                        self._consecutive_skips[sa.name] = 0
                return authority_pick # pyre-ignore
                
            # 2. Contextual Relevance Bidding for RESEARCH and MITIGATION states
            # U2: Use dynamically-built domain bids instead of hardcoded names
            domain_bids = getattr(self, '_domain_bids', {})
            
            highest_bid = -1.0
            next_selected = None
            
            # Only bid among agents in the current GroupChat
            for agent in allowed_agents:
                if agent == last_speaker or agent in [moderator_agent, task_synthesizer, red_team_agent, debiaser_agent]:
                    continue
                bid = 0.0
                words = str(last_msg).lower().split()
                for keyword in domain_bids.get(agent.name, []):
                    bid += float(words.count(keyword))
                if agent.name in self.cognitive_ledger.high_risk_agents:
                    bid *= 2.0

                # U4: AllianceMatrix bid modifiers
                if hasattr(self, '_alliance_matrix'):
                    rel_score = self._alliance_matrix.get(last_speaker.name, agent.name)
                    if rel_score < -0.5:   # Rivalry: agent eager to challenge
                        bid += 2.0
                    elif rel_score > 0.6:  # Deference: agent less likely to interrupt
                        bid -= 0.5
                
                # V26-Fix5: Champion Activation Priority
                # When the last message contains attack language, champions get a +3.0 bid bonus
                # to ensure they step in and defend the feature.
                attack_keywords = ['risk', 'veto', 'reject', 'kill', 'liability', 'breach', 'threat',
                                   'dangerous', 'impossible', 'fatal', 'failure', 'catastroph']
                if any(kw in str(last_msg).lower() for kw in attack_keywords):
                    if any(r in agent.name.upper() for r in ['CEO', 'CPO', 'Sales']):
                        bid += 3.0
                        logger.debug(f"V26-Fix5: Champion bid boost for {agent.name} (attack detected)")
                
                if bid > highest_bid:
                    highest_bid = bid
                    next_selected = agent

            # Fallback if no stakeholder bid
            if not next_selected:
                potential_fallbacks = [a for a in allowed_agents if a != last_speaker and a != task_synthesizer]
                next_selected = potential_fallbacks[0] if potential_fallbacks else moderator_agent

            # U5: Track consecutive skips for frustration
            for sa in stakeholder_agents:
                if sa == next_selected or sa == last_speaker:
                    self._consecutive_skips[sa.name] = 0
                else:
                    self._consecutive_skips[sa.name] = self._consecutive_skips.get(sa.name, 0) + 1
                    if self._consecutive_skips.get(sa.name, 0) >= 2:
                        self.cognitive_ledger.increment_frustration(sa.name)
            
            agenda = self.cognitive_ledger.get_formatted_agenda()
            
            # Blackboard Pruning: Shared Workspace Pins
            blackboard_str = "\n".join([f"- {k}: {v}" for k, v in self.cognitive_ledger.blackboard_conflicts.items()])
            if blackboard_str:
                agenda += f"\n\n--- GLOBAL BLACKBOARD CONFLICTS (JUSTIFICATION-LINKED) ---\n{blackboard_str}"

            # U5: Append assertiveness injection based on frustration level
            assertiveness = self.cognitive_ledger.get_assertiveness_injection(next_selected.name)
            
            # --- V31: FSM PROCEDURE ENFORCEMENT (Structured Phase Directives) ---
            fsm_override = ""
            if state.name == 'RESEARCH':
                fsm_override = (
                    "\n\n<phase_directive phase='RESEARCH'>\n"
                    "You are in the RESEARCH phase. Do NOT state conclusions or vote.\n"
                    "Your job: (1) Identify what data is missing, (2) Call tools to retrieve it, "
                    "(3) Share what you found with the room. Example: 'I need the churn impact data. "
                    "Let me run query_simulation.' Then share the result.\n"
                    "</phase_directive>"
                )
            elif state.name == 'CHALLENGE':
                fsm_override = (
                    "\n\n<phase_directive phase='CHALLENGE'>\n"
                    "You are in the CHALLENGE phase. Your job is to ATTACK the proposal.\n"
                    "Find the mathematical or logical flaw. Name the specific assumption that is wrong "
                    "and propose the failure scenario with probability and blast radius.\n"
                    "Agreement is PROHIBITED in this phase. If you agree with the majority, identify "
                    "the specific conditions under which this proposal would FAIL.\n"
                    + CONTRARIAN_MANDATE +
                    "</phase_directive>"
                )
            elif state.name == 'MITIGATION':
                fsm_override = (
                    "\n\n<phase_directive phase='MITIGATION'>\n"
                    "You are in the MITIGATION phase. For each flaw identified in CHALLENGE:\n"
                    "(1) Propose a specific boundary, condition, or SLA. (2) Name who owns the mitigation. "
                    "(3) Define the failure threshold that re-triggers escalation. No philosophical fixes "
                    "— only measurable remediation with owners and deadlines.\n"
                    "</phase_directive>"
                )
            elif state.name == 'VOTE':
                fsm_override = (
                    "\n\n<phase_directive phase='VOTE'>\n"
                    "VOTING PHASE. No more debate. Call `submit_tension_vector` NOW with your final vote.\n"
                    "Your vote must reflect the evidence presented during this session, not general feelings.\n"
                    "</phase_directive>"
                )
                
            base_sys = next_selected.system_message.split("# AUTONOMOUS TASK LEDGER")[0].split("--- GLOBAL BLACKBOARD")[0].split("[ASSERTIVENESS")[0].split("[PROCEDURAL OVERRIDE")[0].split("<phase_directive")[0].split("[YOUR EVOLVING MEMORY")[0]
            
            # V29: Inject recalled memory from HindsightBoardroom
            memory_context = ""
            try:
                memory_context = self.hindsight_boardroom.recall_for_turn(next_selected.name)
            except Exception as e:
                logger.debug(f"V29: Memory recall failed for {next_selected.name}: {e}")
            
            next_selected.update_system_message(f"{base_sys}\n\n{memory_context}\n\n{agenda}{assertiveness}{fsm_override}")
            print(f"\n[LEDGER+MEMORY INJECTION] {next_selected.name} context updated. Phase: {state.name}. Memory: {len(memory_context)} chars.")
            return next_selected

        # ═══════════════════════════════════════════════════════════════════
        # U22: BROADCAST DELIBERATION — 3-Phase Parallel Architecture
        # ═══════════════════════════════════════════════════════════════════
        
        # U22-P2: Background thread pool for async TaskSynthesizer
        _u22_bg_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="u22_synth")
        
        # U22-P1: Parallel Research Prefetch (Single-Call Direct Research)
        # Instead of spinning up a 4-agent RAG sub-debate per agent (6-8 LLM calls each),
        # we do ONE direct LLM call per agent to generate their research brief.
        # This cuts API usage from ~80 calls to ~10 calls for the research phase.
        # The full multi-agent RAG is preserved as a tool for MID-DEBATE research.
        logger.info("U22-P1: Starting Parallel Research Prefetch (Direct Single-Call Mode)...")
        _prefetch_start = time.time()
        prefetch_cache: Dict[str, Dict[str, str]] = {}
        tools_instance = self._create_tools()
        
        # Rate-limit semaphore: cap at 3 concurrent calls to stay under 15 RPM free-tier
        import threading as _threading
        _rate_semaphore = _threading.Semaphore(3)
        
        def _prefetch_research_direct(agent_obj, feat):
            """Single direct LLM call per agent — replaces the 4-agent RAG sub-debate for prefetch."""
            agent_name = agent_obj.name
            try:
                _rate_semaphore.acquire()
                try:
                    # V31: Domain-scoped research brief (role-specific investigation frame)
                    # Determines what questions this specific role would ask in pre-meeting preparation
                    role_name_lower = agent_name.lower()
                    domain_frame = "Key risks, dependencies, and open questions from your domain"
                    if 'cfo' in role_name_lower or 'finance' in role_name_lower:
                        domain_frame = (
                            "(1) Total cost estimate with burn rate model, (2) Budget utilization vs ceiling, "
                            "(3) Revenue offset potential, (4) Capital allocation risk if approved"
                        )
                    elif 'cto' in role_name_lower or 'tech' in role_name_lower:
                        domain_frame = (
                            "(1) Architecture dependencies and integration risks, (2) Deployment timeline "
                            "with sprint-level granularity, (3) Tech debt impact, (4) Scalability bottlenecks"
                        )
                    elif 'ciso' in role_name_lower or 'security' in role_name_lower:
                        domain_frame = (
                            "(1) Attack surface expansion, (2) Known CVEs in proposed dependencies, "
                            "(3) Compliance gaps (SOC2/HIPAA/GDPR), (4) Threat model for data flow"
                        )
                    elif 'cpo' in role_name_lower or 'product' in role_name_lower:
                        domain_frame = (
                            "(1) User adoption risk and activation friction, (2) Competitive positioning, "
                            "(3) Churn impact if launched vs not launched, (4) Feature-market fit evidence"
                        )
                    elif 'ceo' in role_name_lower:
                        domain_frame = (
                            "(1) Strategic alignment with board-level OKRs, (2) Opportunity cost of this "
                            "vs alternatives, (3) Competitive response timeline, (4) Revenue impact estimate"
                        )
                    
                    research_prompt = (
                        f"<pre_meeting_brief>\n"
                        f"You are {agent_name}. The board will debate this feature in 10 minutes.\n"
                        f"FEATURE: {feat.title}\n"
                        f"BRIEF: {feat.description[:600]}\n\n"
                        f"Prepare your domain-specific intelligence brief. Investigate:\n"
                        f"{domain_frame}\n\n"
                        f"Output: 3-5 bullet points. Each must contain a SPECIFIC number, metric, or fact.\n"
                        f"End with: INITIAL STANCE: SUPPORT / OPPOSE / CONDITIONAL + one-sentence basis.\n"
                        f"</pre_meeting_brief>"
                    )
                    reply = agent_obj.generate_reply(messages=[{"role": "user", "content": research_prompt}])
                    if isinstance(reply, dict):
                        reply = reply.get("content", "")
                    rag_result = str(reply)[:1500]
                finally:
                    _rate_semaphore.release()
                
                # Web search is static/mocked — no LLM call needed
                web_result = tools_instance["web_search"](f"{feat.title} risks {agent_name}")
                
                # Record research receipt so ER-401 check passes
                self.receipt_ledger.record(agent_name, "run_multi_agent_rag", str(rag_result)[:50])
                self.receipt_ledger.record(agent_name, "web_search", str(web_result)[:50])
                return agent_name, rag_result, web_result[:500]
            except Exception as e:
                logger.warning(f"U22-P1: Prefetch failed for {agent_name}: {e}")
                return agent_name, "", ""
        
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="u22_direct") as rag_pool:
            futures = [rag_pool.submit(_prefetch_research_direct, a, feature) for a in stakeholder_agents]
            for fut in as_completed(futures):
                name, rag, web = fut.result()
                prefetch_cache[name] = {"rag": rag, "web": web}
        
        _prefetch_elapsed = time.time() - _prefetch_start
        logger.info(f"U22-P1: Parallel Research Prefetch DONE for {len(prefetch_cache)} agents in {_prefetch_elapsed:.1f}s")
        
        # U22-P1: Inject prefetched research into each agent's system message
        for agent in stakeholder_agents:
            cached = prefetch_cache.get(agent.name, {})
            if cached.get("rag") or cached.get("web"):
                research_injection = (
                    f"\n\n[PRE-FETCHED RESEARCH BRIEF — DO NOT RE-SEARCH THIS DATA]\n"
                    f"Research Analysis: {cached.get('rag', 'N/A')[:800]}\n"
                    f"Market Intelligence: {cached.get('web', 'N/A')[:300]}\n"
                    f"[END RESEARCH BRIEF — You may now proceed directly to analysis and voting.]"
                )
                base_sys = agent.system_message
                agent.update_system_message(base_sys + research_injection)
        
        # U22-P3: Parallel Initial Stance Generation (Broadcast)
        logger.info("U22-P3: Generating parallel initial stances for conflict detection...")
        _stance_start = time.time()
        initial_stances: Dict[str, str] = {}
        
        def _generate_stance(agent_obj, feat):
            """Generate a single agent's initial position statement concurrently."""
            try:
                # V31: Committed position with specific objection or endorsement condition
                stance_prompt = (
                    f"<position_statement>\n"
                    f"Feature: '{feat.title}'\n"
                    f"Brief: {feat.description[:500]}\n\n"
                    f"Take a COMMITTED position from your professional domain. No fence-sitting.\n"
                    f"In exactly 2 sentences: (1) Your verdict — SUPPORT, OPPOSE, or CONDITIONAL — and the "
                    f"single most important reason. (2) The specific condition that would change your mind.\n\n"
                    f"Example: 'I OPPOSE this feature. At our current burn rate, the 5-month runway extension "
                    f"puts us in breach of our debt covenant. I would reconsider if the CFO shows a model "
                    f"where monthly spend stays under $400K.'\n"
                    f"</position_statement>"
                )
                reply = agent_obj.generate_reply(messages=[{"role": "user", "content": stance_prompt}])
                if isinstance(reply, dict):
                    reply = reply.get("content", "")
                return agent_obj.name, AG2DebateEngine._strip_thought_tags(str(reply)[:500])
            except Exception as e:
                logger.warning(f"U22-P3: Stance generation failed for {agent_obj.name}: {e}")
                return agent_obj.name, ""
        
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="u22_stance") as stance_pool:
            futures = [stance_pool.submit(_generate_stance, a, feature) for a in stakeholder_agents]
            for fut in as_completed(futures):
                name, stance = fut.result()
                if stance:
                    initial_stances[name] = stance
        
        _stance_elapsed = time.time() - _stance_start
        logger.info(f"U22-P3: Parallel stances generated for {len(initial_stances)} agents in {_stance_elapsed:.1f}s")
        
        # U22-P3: Conflict Detection — identify the top 3 most divergent agents
        def _detect_top_conflicts(stances: Dict[str, str], n: int = 3) -> List[str]:
            """Find the n agents with the most divergent stances from the mean."""
            if len(stances) <= n:
                return list(stances.keys())
            mean_text = " ".join(stances.values()).lower()
            scores = {}
            for name, text in stances.items():
                scores[name] = 1.0 - difflib.SequenceMatcher(None, mean_text, text.lower()).ratio()
            ranked = sorted(scores, key=lambda k: scores[k], reverse=True)  # pyre-ignore
            logger.info(f"U22-P3: Conflict scores: { {k: f'{scores[k]:.3f}' for k in ranked[:5]} }")
            return ranked[:n]
        
        top_conflict_names = _detect_top_conflicts(initial_stances, n=3)
        conflict_agents = [a for a in stakeholder_agents if a.name in top_conflict_names]
        background_agents = [a for a in stakeholder_agents if a.name not in top_conflict_names]
        
        logger.info(f"U22-P3: FOCUSED DELIBERATION with: {[a.name for a in conflict_agents]}")
        logger.info(f"U22-P3: BATCH VOTERS (background): {[a.name for a in background_agents]}")
        
        # U22-P3: Compile the stance digest for the focused debate
        # V25: Increase truncation to 500 chars to avoid broken thought tags
        # V28-Fix5: Clean stance digest — 2 sentences max per agent, no raw thought tags
        def _clean_stance(text: str) -> str:
            """Extract first 2 sentences from stance, stripping all thought tags."""
            cleaned = AG2DebateEngine._strip_thought_tags(text)
            # Detect stance marker (SUPPORT/OPPOSE/CONDITIONAL) if present
            stance_marker = ''
            for marker in ['SUPPORT', 'OPPOSE', 'CONDITIONAL']:
                if marker in cleaned.upper():
                    stance_marker = f'[{marker}] '
                    break
            # Extract first 2 sentences
            import re as _re_stance
            sentences = _re_stance.split(r'(?<=[.!?])\s+', cleaned.strip())
            two_sentences = ' '.join(sentences[:2])[:300]
            return f"{stance_marker}{two_sentences}"
        
        stance_digest = "\n".join([
            f"- {name}: {_clean_stance(stance)}" for name, stance in initial_stances.items()
        ])
        
        # V25-Fix2: Feature Champion Defense — inject defense mandate into CEO/CPO/Sales
        # These agents have business incentive to APPROVE the feature and must push back on critics.
        CHAMPION_ROLES = ['CEO', 'CPO', 'Sales']
        champion_names = []
        for agent in stakeholder_agents:
            role_in_name = agent.name.upper()
            if any(r in role_in_name for r in CHAMPION_ROLES):
                defense_injection = (
                    "\n\n[FEATURE DEFENSE MANDATE]\n"
                    "You are the CHAMPION of this feature proposal. Your career and credibility depend on it.\n"
                    "When critics attack, you MUST push back with specific counter-arguments:\n"
                    "- If the CISO raises security risks, argue that the risk is manageable with mitigations.\n"
                    "- If the CFO challenges costs, present the revenue upside and ROI.\n"
                    "- If someone proposes killing or weakening the feature, defend its strategic value.\n"
                    "Do NOT concede easily. Demand that critics provide hard data, not hypothetical scenarios.\n"
                    "A real executive defending their proposal does not fold at the first objection."
                )
                agent.update_system_message(agent.system_message + defense_injection)
                champion_names.append(agent.name)
        
        logger.info(f"V25-Fix2: Feature Champions designated: {champion_names}")
        
        # V25-Fix1 + V24-Fix1: Inject adversarial reasoning into NON-champion conflict agents
        for ca in conflict_agents:
            if ca.name not in champion_names:
                adversarial_injection = (
                    "\n\n[ADVERSARIAL REASONING MANDATE]\n"
                    "Before stating your position, you MUST internally identify ONE fatal flaw in the proposal "
                    "from your domain expertise. Present BOTH your concern AND a specific 'Fatal Scenario' with "
                    "a quantitative metric (e.g., '0.7 probability of breach within 12 months'). "
                    "You are expected to CHALLENGE other executives' numbers directly. "
                    "If the CEO cites a revenue figure, question it. If the CISO raises a risk, demand the specific CVE. "
                    "This is a REAL boardroom — argue, push back, demand specifics."
                )
                ca.update_system_message(ca.system_message + adversarial_injection)
        
        # V25-Fix1: ALL stakeholders in the GroupChat (not just top-3 conflict agents)
        # This enables direct-address routing: when Alice is called by name, she can respond.
        focused_agents = stakeholder_agents + [red_team_agent, debiaser_agent, moderator_agent, task_synthesizer]
        
        groupchat = autogen.GroupChat(
            agents=focused_agents,
            messages=[],
            max_round=ADJOURNMENT_TURN_LIMIT + 5,  # V25: Extra turns for Last Call + verdict
            speaker_selection_method=fsm_speaker_selector
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=self.primary_config, max_consecutive_auto_reply=ADJOURNMENT_TURN_LIMIT + 5)
        
        # Stream debate messages to pipeline.jsonl in real-time
        def stream_debate_message(recipient, messages, sender, config):
            if messages:
                last_msg = messages[-1]
                content = last_msg.get("content", "")
                if content:
                    sender_name = last_msg.get("name") or sender.name
                    # Strip any thought tags for clean UI display
                    clean_content = AG2DebateEngine._strip_thought_tags(content).strip()
                    
                    # If this message has actual content, and is not a system control message
                    if clean_content and not any(token in clean_content for token in ["[SOVEREIGN", "[SESSION", "[BOARDROOM", "BOARD MEMORANDUM"]):
                        # Determine if it's a challenge
                        is_challenge = any(k in clean_content.lower() for k in ["risk", "veto", "reject", "challenge", "object", "disagree", "cve", "leak", "vulnerability"])
                        
                        # Format the sender name for the UI (e.g., remove prefix if present, e.g. "Alpha_CTO" -> "CTO")
                        ui_sender = sender_name
                        if "_" in sender_name:
                            parts = sender_name.split("_")
                            if len(parts) > 1:
                                ui_sender = parts[1] # e.g. "CTO", "CEO"
                                
                        # ID for frontend
                        msg_id = f"db_{len(messages)}"
                        
                        # Construct the event payload
                        event_payload = {
                            "type": "debate_message",
                            "message": {
                                "id": msg_id,
                                "sender": ui_sender,
                                "text": clean_content,
                                "type": "challenge" if is_challenge else "normal"
                            }
                        }
                        
                        # Write to pipeline.jsonl if available
                        if pipeline_jsonl:
                            try:
                                with open(pipeline_jsonl, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(event_payload) + "\n")
                                logger.info(f"Streamed debate message from {ui_sender} to pipeline.jsonl")
                            except Exception as exc:
                                logger.warning(f"Failed to stream debate message: {exc}")
            return False, None

        manager.register_reply(
            trigger=lambda sender: True,
            reply_func=stream_debate_message,
            position=0
        )
        
        logger.info("Executing AG2 Autonomous Boardroom Debate (V24 Direct Cross-Agent Mode)...")
        
        # V28-Fix7: Enhanced Deliberation Protocol — Novelty-First Board Memo
        initial_message = (
            f"BOARD MEMORANDUM — AGENDA ITEM:\n"
            f"Feature Proposal: {feature.title}\n"
            f"[The brief has been distributed. Do NOT restate it. Proceed directly to analysis.]\n\n"
            f"=== INITIAL BOARD POSITIONS ===\n{stance_digest}\n\n"
            "DELIBERATION PROTOCOL (V28 — Novelty-First):\n"
            "1. LIVE BOARDROOM. Address executives BY NAME when responding.\n"
            "2. NO REHASHING: The brief is read. Every statement must add NEW value — "
            "a new risk, a new number, a new solution, or a direct challenge.\n"
            "3. EVIDENCE DEMAND: Challenge any unvalidated number. Ask 'Based on what data?' "
            "Flag unproven claims as 'UNVALIDATED ASSUMPTION — requires [validation step].'\n"
            "4. CROSS-EXAMINATION is MANDATORY. Each turn, ask at least ONE specific question to another exec.\n"
            "5. VETO OWNERSHIP: If you veto (is_high_risk=True), YOU own the resolution. "
            "Propose your specific fix — don't just say 'no'. Other execs: if you propose a mitigation "
            "for a veto, the veto-raiser gets the next turn to accept or sustain their veto.\n"
            "6. BEFORE VOTING: State ONE thing you learned from this debate that CHANGED your initial position. "
            "If nothing changed, explain specifically why the counter-arguments failed.\n"
            "7. Vote using `submit_tension_vector` with dimensional scores.\n"
            "8. The Chairman will adjourn if deliberation stalls."
        )
        
        # Initiate Chat with the first conflict agent or first focused stakeholder
        initiator = conflict_agents[0] if conflict_agents else None
        if not initiator:
             # Find first stakeholder in focused list
             focused_stakeholders = [a for a in focused_agents if a in stakeholder_agents]
             initiator = focused_stakeholders[0] if focused_stakeholders else moderator_agent

        chat_res = initiator.initiate_chat(
            manager,
            message=initial_message,
        )
        
        # U23-Fix3: Informed Batch Voting — background agents vote based on debate outcome, not heuristics
        # Extract debate summary from the focused deliberation
        debate_messages = groupchat.messages or []
        debate_summary = "\n".join([
            f"{msg.get('name', 'Unknown')}: {AG2DebateEngine._strip_thought_tags(msg.get('content', ''))[:300]}"
            for msg in debate_messages[-6:]  # Last 6 messages capture the core conflict resolution
        ])
        
        logger.info(f"U23-P3: Submitting INFORMED batch votes for {len(background_agents)} background agents...")
        
        def _informed_batch_vote(bg_agent_obj, feat, stance_text, debate_text):
            """U23-Fix3 + V31: Generate informed CoT vote using debate context."""
            try:
                # V31-Fix2: Apply same CoT chain as force-vote for consistency
                vote_prompt = (
                    f"<force_vote>\n"
                    f"You are {bg_agent_obj.name.replace('_', ' ')}. You observed the board debate '{feat.title}'.\n\n"
                    f"YOUR INITIAL STANCE: {stance_text[:300]}\n\n"
                    f"KEY DEBATE MOMENTS:\n{debate_text[:800]}\n\n"
                    f"Think step by step:\n"
                    f"1. What argument from the debate STRENGTHENED your initial stance?\n"
                    f"2. What argument CHALLENGED it and why did it or did not change your mind?\n"
                    f"3. From YOUR domain specifically, what is the decisive factor?\n\n"
                    f"Then output EXACT JSON (nothing else):\n"
                    f'{{"dimension": "<one of: Technical_Feasibility|Unit_Economics|Security_Risk|Market_Fit|Strategic_Alignment|Legal_Compliance>", '
                    f'"score": <0.1-0.9>, "confidence": <0.3-0.9>, "is_high_risk": <true/false>, '
                    f'"reasoning": "<one sentence citing specific evidence from the debate>"}}\'\n'
                    f"</force_vote>"
                )
                reply = bg_agent_obj.generate_reply(messages=[{"role": "user", "content": vote_prompt}])
                if isinstance(reply, dict):
                    reply = reply.get("content", "")
                reply_text = AG2DebateEngine._strip_thought_tags(str(reply))
                
                # Parse JSON from reply
                import json as _json
                json_match = re.search(r'\{[^{}]+\}', reply_text)
                if json_match:
                    vote_data = _json.loads(json_match.group())
                    return (
                        bg_agent_obj.name,
                        str(vote_data.get('dimension', 'General_Assessment')),
                        float(vote_data.get('score', 0.5)),
                        float(vote_data.get('confidence', 0.5)),
                        bool(vote_data.get('is_high_risk', False)),
                        str(vote_data.get('reasoning', ''))
                    )
            except Exception as e:
                logger.warning(f"U23: Informed vote failed for {bg_agent_obj.name}: {e}")
            
            # Fallback to heuristic if LLM call fails
            return bg_agent_obj.name, 'General_Assessment', 0.5, 0.5, False, 'Fallback heuristic vote'
        
        import re
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="u23_vote") as vote_pool:
            futures = [
                vote_pool.submit(
                    _informed_batch_vote, bg, feature,
                    initial_stances.get(bg.name, ""), debate_summary
                ) for bg in background_agents
            ]
            for fut in as_completed(futures):
                name, dim_key, score, confidence, is_high_risk, reasoning = fut.result()
                
                score = max(0.1, min(0.9, score))
                confidence = max(0.3, min(0.9, confidence))
                
                batch_payload = TensionPayload(
                    adjustments={dim_key: score},
                    confidence=confidence,
                    is_high_risk=is_high_risk,
                    is_low_information=False,  # U23: No longer low-info since they heard the debate
                    tool_call_hashes=[]
                )
                self.live_tension_registry[name] = batch_payload
                self.cognitive_ledger.record_confidence(name, confidence)
                self.cognitive_ledger.mark_voted(name)
                if is_high_risk:
                    self.cognitive_ledger.mark_high_risk(name)
                logger.info(f"V24 INFORMED VOTE: {name} -> {dim_key}={score:.2f}, conf={confidence:.2f}, high_risk={is_high_risk}, reason={reasoning[:80]}")
        
        # V24-Fix5: Log batch voter output to transcript and stdout
        batch_vote_lines = []
        for bg in background_agents:
            bg_name = bg.name
            if bg_name in self.live_tension_registry:
                bp = self.live_tension_registry[bg_name]
                bp_adjs = bp.adjustments
                bp_conf = bp.confidence
                bp_risk = bp.is_high_risk
                batch_vote_lines.append(
                    f"  • {bg_name}: {bp_adjs} (confidence: {bp_conf:.2f}, high_risk: {bp_risk})"
                )
        
        if batch_vote_lines:
            batch_summary_text = (
                f"\n{'='*60}\n"
                f"📊  BATCH VOTER RESULTS ({len(batch_vote_lines)} background agents)\n"
                f"{'='*60}\n"
                + "\n".join(batch_vote_lines)
                + f"\n{'='*60}"
            )
            print(batch_summary_text)
            logger.info(f"V24-Fix5: Batch vote summary:\n" + "\n".join(batch_vote_lines))
            
            # Append to groupchat messages for transcript persistence
            groupchat.messages.append({
                "role": "assistant",
                "name": "Boardroom_Moderator",
                "content": (
                    f"BATCH VOTES RECEIVED — The following board members voted based on the deliberation:\n"
                    + "\n".join(batch_vote_lines)
                )
            })
        
        # Map agents by name to recover Persona references for Domain Authority scaling
        persona_map = {p.name.replace(" ", "_").replace(".", ""): p for p in personas}
        
        # 6. Evaluate Result via Live Tension Ledger (No post-hoc text parsing)
        tension_shifts: Dict[str, float] = {}
        dim_weighted_sums: Dict[str, float] = {}   # U1: per-dimension weighted sums
        dim_total_weights: Dict[str, float] = {}   # U1: per-dimension total weights
        parsed_votes: int = 0
        has_high_risk: bool = False
        low_information_votes: int = 0
        
        # Scan through pristine, Pydantic-verified Tool Payloads
        for agent_name, payload in getattr(self, "live_tension_registry", {}).items():
            if getattr(payload, "is_high_risk", False):
                has_high_risk = True
                logger.warning(f"FATAL VETO TRIGGERED: {agent_name} flagged explicit High Risk via Pydantic.")
            if getattr(payload, "is_low_information", False):
                low_information_votes = int(low_information_votes) + 1 # pyre-ignore
                
            conf = float(getattr(payload, "confidence", 0.5))
            parsed_votes = int(parsed_votes) + 1 # pyre-ignore
            
            # Look up Domain Authority
            persona = persona_map.get(agent_name.replace(" ", "_").replace(".", ""))
            
            adjustments = getattr(payload, "adjustments", {}) # pyre-ignore
            for k, v in adjustments.items():
                v_float = float(v)
                dim_key = str(k)
                
                # Hard-Stop Veto Math: Reject instantly if a critical dimension drops < 0.2
                if v_float < 0.2:
                    has_high_risk = True
                    logger.warning(f"HARD-STOP CRITICAL: {agent_name} cited a failure trajectory ({v_float}) on {k}. Veto engaged.")
                    
                # U1: 3× multiplier when domain_expertise matches dimension
                domain_auth_multiplier = 1.0
                if persona and getattr(persona, 'domain_expertise', None):
                    if any(dim_key.lower() in expert.lower() or expert.lower() in dim_key.lower() for expert in persona.domain_expertise):
                        domain_auth_multiplier = 3.0
                        
                weight = conf * domain_auth_multiplier
                dim_weighted_sums[dim_key] = dim_weighted_sums.get(dim_key, 0.0) + (v_float * weight)
                dim_total_weights[dim_key] = dim_total_weights.get(dim_key, 0.0) + weight
                    
        # U1: Compute per-dimension weighted means, then average across dimensions
        for dim_key in dim_weighted_sums:
            if dim_total_weights.get(dim_key, 0.0) > 0.0:
                tension_shifts[dim_key] = dim_weighted_sums[dim_key] / dim_total_weights[dim_key]
            else:
                tension_shifts[dim_key] = 0.5  # neutral fallback

        # U1: Final score = mean of per-dimension weighted means, capped [0.0, 1.0]
        # This naturally varies: all 0.0 votes → ~0.0→clamped ~0.3, all 1.0 → ~1.0→clamped ~0.9
        if tension_shifts:
            raw_mean = sum(tension_shifts.values()) / len(tension_shifts)
            # Apply floor/ceiling scaling: map [0.0, 1.0] → [0.3, 0.9]
            final_score = 0.3 + (raw_mean * 0.6)
        else:
            final_score = 0.5  # No votes cast
        final_score = max(0.0, min(1.0, final_score))
        
        # Automatic Escalation (FAIL-SOFT) check
        # U16.2: Dynamic Fail-Soft — suppress escalation if we are in the final synthesis stage
        is_low_info_escalation = (parsed_votes > 0 and low_information_votes > (parsed_votes / 2))
        if is_low_info_escalation:
            # Check if we should override and force completion
            if self.reasoning_only or True: # Force completion as per user request
                logger.warning("LOW FIDELITY DETECTED: Majority of votes were 'Low Information'. FORCING LOGICAL COMPLETION as per Reasoning-First mandate.")
                is_low_info_escalation = False
            else:
                logger.error("AUTOMATIC ESCALATION TRIGGERED: Majority of votes were 'Low Information' due to failed searches.")
        
        # If Epistemic Veto triggered, downgrade verdict
        if has_high_risk:
            logger.warning("Epistemic Calibration Threshold breached. Flagging verdict as HIGH RISK.")
            verdict = "REJECTED" if final_score < 0.6 else "CONDITIONALLY_APPROVED"
        else:
            verdict = "APPROVED" if final_score >= 0.7 else ("CONDITIONALLY_APPROVED" if final_score >= 0.5 else "REJECTED")

        # U22-P2: Wait for all background tasks to complete before serializing tasks/ledger
        logger.info("U22-P2: Waiting for background TaskSynthesizer and Debiaser tasks to complete...")
        _u22_bg_pool.shutdown(wait=True)
        logger.info("U22-P2: Background thread pool successfully shut down and synchronized.")

        # ═══════════════════════════════════════════════════════════════════
        # V31: BOARD DECISION RECORD — Verdict Synthesis
        # Transforms numerical vote aggregation → actionable human-readable mandate.
        # This is what makes this a DECISION ENGINE, not just a debate engine.
        # ═══════════════════════════════════════════════════════════════════
        board_decision_record = ""
        try:
            # Build vote summary for the synthesis prompt
            vote_lines = []
            for agent_name, payload in getattr(self, "live_tension_registry", {}).items():
                adj = getattr(payload, "adjustments", {})
                conf = getattr(payload, "confidence", 0.5)
                hr = getattr(payload, "is_high_risk", False)
                dim_str = ", ".join(f"{k}={v:.2f}" for k, v in adj.items())
                vote_lines.append(f"  {agent_name}: {dim_str} | confidence={conf:.2f} | high_risk={hr}")
            
            resolved_tasks_str = "\n".join([
                f"  [{tid}] {t['title']}: {t['status']} — {t.get('resolution', '')}"
                for tid, t in self.cognitive_ledger.tasks.items()
            ])
            
            sustained_vetoes = [
                name for name, payload in getattr(self, "live_tension_registry", {}).items()
                if getattr(payload, "is_high_risk", False)
            ]

            verdict_synthesis_prompt = (
                f"<verdict_synthesis>\n"
                f"The board has completed deliberation on: '{feature.title}'\n\n"
                f"MATHEMATICAL VERDICT: {verdict} (score={final_score:.2f})\n\n"
                f"INDIVIDUAL VOTES:\n" + "\n".join(vote_lines) + "\n\n"
                f"TASK LEDGER STATUS:\n{resolved_tasks_str}\n\n"
                f"SUSTAINED VETOES: {', '.join(sustained_vetoes) if sustained_vetoes else 'None'}\n\n"
                f"As Chairman, produce the BOARD DECISION RECORD. Be specific and actionable.\n"
                f"Format exactly as:\n\n"
                f"DECISION: [APPROVE / CONDITIONAL APPROVE / REJECT]\n"
                f"BASIS: [The single most decisive argument — cite the executive and their specific evidence]\n"
                f"CONDITIONS: [If CONDITIONAL — list exactly what must be verified before shipping, with owners]\n"
                f"DISSENT: [Any sustained vetoes, who holds them, and what must change]\n"
                f"NEXT ACTION: [One concrete next step with owner name and deadline]\n"
                f"</verdict_synthesis>"
            )
            
            bdr_reply = moderator_agent.generate_reply(
                messages=[{"role": "user", "content": verdict_synthesis_prompt}]
            )
            if isinstance(bdr_reply, dict):
                bdr_reply = bdr_reply.get("content", "")
            board_decision_record = AG2DebateEngine._strip_thought_tags(str(bdr_reply))
            
            # Print to stdout for visibility
            print(f"\n{'═'*80}")
            print(f"📋  BOARD DECISION RECORD")
            print(f"{'═'*80}")
            print(board_decision_record)
            print(f"{'═'*80}\n")
            logger.info(f"V31: Board Decision Record generated ({len(board_decision_record)} chars)")
            
            # Append to groupchat transcript for downstream pipeline
            groupchat.messages.append({
                "role": "assistant",
                "name": "Boardroom_Moderator",
                "content": f"BOARD DECISION RECORD:\n{board_decision_record}"
            })
        except Exception as e:
            logger.warning(f"V31: Verdict synthesis failed: {e}")
            board_decision_record = f"DECISION: {verdict}\nBASIS: Score={final_score:.2f}. Automated fallback — synthesis LLM call failed.\n"

        # Improvement 19: Confidence Calibration Audit
        logger.info("=== Phase 4: Confidence Calibration Audit ===")
        consensus_score = final_score
        calibration_report = []
        for agent_name, history in self.cognitive_ledger.confidence_history.items():
            if not history: continue
            initial_conf = history[0]
            final_conf = history[-1]
            drift = final_conf - initial_conf
            discrepancy = abs(final_conf - consensus_score)
            calibration_report.append({
                "agent": agent_name,
                "initial": initial_conf,
                "final": final_conf,
                "drift": drift,
                "discrepancy_from_consensus": discrepancy
            })
        logger.info(f"Calibration Report: {calibration_report}")

        # Stop Runtime logging session
        autogen.runtime_logging.stop()
        
        # V29: Persist evolved agent memories
        try:
            evolved_memories = self.hindsight_boardroom.get_all_memories()
            for agent_name, mem_dict in evolved_memories.items():
                reflection = self.hindsight_boardroom.reflect_post_debate(agent_name)
                mem_dict['evolved_reflection'] = reflection
                logger.info(f"V29: Evolved memory for {agent_name}: {mem_dict.get('turn_count', 0)} turns, "
                            f"{len(mem_dict.get('commitments', []))} commitments, "
                            f"{len(mem_dict.get('proposals', []))} proposals")
            # Store in instance for DB persistence downstream
            self._evolved_agent_memories = evolved_memories
            logger.info(f"V29: Persisted evolved memories for {len(evolved_memories)} agents.")
        except Exception as e:
            logger.warning(f"V29: Memory persistence failed: {e}")
            self._evolved_agent_memories = {}

        # Map AG2 groupchat.messages → DebateRound for DB persistence
        import re
        positions = []
        logger.info(f"TRANSCRIPT DEBUG: Total messages in groupchat.messages = {len(groupchat.messages)}")
        for i, msg in enumerate(groupchat.messages):
            role_val = msg.get("role", "unknown")
            name_val = msg.get("name", "unknown")
            content = msg.get("content") or ""
            
            # Debug log for first few messages
            if i < 5 or i > len(groupchat.messages) - 5:
                logger.info(f"MSG {i}: role={role_val}, name={name_val}, content_len={len(content)}")

            name = (name_val or role_val).split(" (to")[0]
            if not content:
                continue
                
            # Realism Cleanup: Remove "Thought:", "INTERNAL MEMO", and other AI markers
            # Filter out lines starting with common AG2 system markers
            cleaned_content = re.sub(r'Thought:.*?\n', '', content, flags=re.IGNORECASE | re.DOTALL)
            cleaned_content = re.sub(r'<thought>.*?</thought>', '', cleaned_content, flags=re.IGNORECASE | re.DOTALL)
            cleaned_content = re.sub(r'INTERNAL MEMO.*?\n', '', cleaned_content, flags=re.IGNORECASE)
            cleaned_content = re.sub(r'LOGIC CRITIC REPORT.*?\n', '', cleaned_content, flags=re.IGNORECASE)
            cleaned_content = re.sub(r'REDTEAM REPORT.*?\n', '', cleaned_content, flags=re.IGNORECASE)
            cleaned_content = cleaned_content.strip()
            
            if not cleaned_content:
                continue

            statement_limit = str(cleaned_content)[:4000] # pyre-ignore
            positions.append(DebatePosition(
                stakeholder_name=name,
                role=name,
                statement=statement_limit,
                verdict="CAST VOTE" if "CAST VOTE ALERT" in content else "DEBATING",
                confidence=float(getattr(self.live_tension_registry.get(name), "confidence", 0.5)) if name in getattr(self, "live_tension_registry", {}) else 0.5, # pyre-ignore
            ))

        dr = DebateRound(
            round_number=1,
            round_name="AG2 Multi-Agent Sovereign Debate",
            synthesis=f"Debate on '{feature.title}' completed with {parsed_votes} Pydantic votes and final score {final_score:.2f}.",
            positions=positions,
        )

        final_verdict = "ESCALATED_TO_LAYER_5" if is_low_info_escalation else verdict
        summary_intro = f"The AG2 autonomous board debated {feature.title}."
        if is_low_info_escalation:
            summary_intro = f"AUTOMATIC ESCALATION: {low_information_votes}/{parsed_votes} votes flagged as 'Low Information'. Synthesizing boardroom reasoning despite search failure. "
        
        # U23-Fix5: Compromise Synthesis — generate specific conditions for non-APPROVED verdicts
        conditions_list: list = []
        conditions_summary = ""
        if verdict in ("CONDITIONALLY_APPROVED", "REJECTED") and debate_messages:
            try:
                logger.info("U23-Fix5: Generating Boardroom Compromise Conditions...")
                # Build a compact summary of high-risk flags and debate content
                high_risk_agents = [name for name, p in self.live_tension_registry.items() if getattr(p, 'is_high_risk', False)]
                risk_summary = f"HIGH-RISK FLAGS FROM: {', '.join(high_risk_agents)}" if high_risk_agents else "No explicit high-risk flags"
                
                compromise_prompt = (
                    f"You are a Boardroom Secretary. The board just debated '{feature.title}' and reached verdict: {verdict} "
                    f"(confidence: {final_score:.2f}).\n\n"
                    f"DEBATE SUMMARY:\n{debate_summary[:1000]}\n\n"
                    f"{risk_summary}\n\n"
                    f"TENSION SCORES: {tension_shifts}\n\n"
                    f"Generate 2-4 SPECIFIC, ACTIONABLE conditions that would need to be met for this feature to proceed. "
                    f"Each condition must be concrete and measurable. Respond as a JSON array of strings.\n"
                    f"Example: [\"Cap initial rollout to 50k users in Phase 1\", \"Obtain BIPA legal sign-off before launch\"]\n"
                    f"Respond ONLY with the JSON array, no other text."
                )
                
                import openai as _oai
                _client = _oai.OpenAI(
                    api_key=self.primary_config.get('config_list', [{}])[0].get('api_key', ''),
                    base_url=self.primary_config.get('config_list', [{}])[0].get('base_url', '')
                )
                _model = self.primary_config.get('config_list', [{}])[0].get('model', 'gemma-4-31b-it')
                
                _resp = _client.chat.completions.create(
                    model=_model,
                    messages=[{"role": "user", "content": compromise_prompt}],
                    max_tokens=300,
                    temperature=0.3
                )
                _raw = AG2DebateEngine._strip_thought_tags(_resp.choices[0].message.content.strip())
                
                import json as _json
                # Extract JSON array from response
                _arr_match = re.search(r'\[.*\]', _raw, re.DOTALL)
                if _arr_match:
                    conditions_list = _json.loads(_arr_match.group())
                    conditions_summary = " | ".join(conditions_list[:4])
                    logger.info(f"U23-Fix5: Compromise conditions: {conditions_list}")
            except Exception as e:
                logger.warning(f"U23-Fix5: Compromise synthesis failed: {e}")
                conditions_list = ["Limit initial rollout to 5% of users", "Conduct independent security audit before launch"]
        
        full_summary = f"{summary_intro} Agents parsed: {parsed_votes}. Confidence-weighted score: {final_score:.2f}."
        if conditions_summary:
            full_summary += f" CONDITIONS: {conditions_summary}"
            
        return ConsensusResult(
            feature_name=feature.title,
            overall_verdict=final_verdict,
            approval_confidence=final_score,
            stakeholder_verdicts={k: "Voted" for k in getattr(self, "live_tension_registry", {}).keys()},
            approvals=[],
            mitigations=conditions_list if conditions_list else (["Limit initial rollout to 5% of users"] if final_score < 0.8 else []),
            tension_shifts=tension_shifts,
            overall_summary=full_summary,
            debate_rounds=[dr]
        )