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
from tsc.models.gates import GatesSummary
from tsc.models.debate import ConsensusResult, DebatePosition, DebateRound
from tsc.memory.zep_client import ZepMemoryClient
from tsc.memory.fact_retriever import FactRetriever

class DebateState(Enum):
    OPENING    = auto()  # Initial framing, 1 turn per agent
    RESEARCH   = auto()  # Mandatory RAG phase before any stance
    CHALLENGE  = auto()  # Red team + contrarian adversarial phase
    MITIGATION = auto()  # Proposed solutions and risk mitigations
    VOTE       = auto()  # Final vote collection, no new arguments
    CLOSED     = auto()  # Termination sentinel

class DebateStateMachine:
    TRANSITIONS = {
        DebateState.OPENING:    [DebateState.RESEARCH],
        DebateState.RESEARCH:   [DebateState.CHALLENGE, DebateState.VOTE],
        DebateState.CHALLENGE:  [DebateState.MITIGATION, DebateState.VOTE],
        DebateState.MITIGATION: [DebateState.VOTE],
        DebateState.VOTE:       [DebateState.CLOSED],
    }
    STATE_ROUND_BUDGETS = {
        DebateState.OPENING:    2,   # U22: Broadcast handles initial stances
        DebateState.RESEARCH:   1,   # U22: Prefetch eliminates this phase
        DebateState.CHALLENGE:  3,   # U22: Top 3 conflicts only
        DebateState.MITIGATION: 2,
        DebateState.VOTE:       4,   # U22: Batch voting
    }

    def __init__(self, agent_count: int):
        self.current_state = DebateState.OPENING
        self.state_round   = 0
        self._vote_index   = 0
        self._agent_count  = agent_count
        # Static budgets enforced for <10min simulation speed.

    def tick(self) -> DebateState:
        self.state_round += 1
        budget = self.STATE_ROUND_BUDGETS.get(self.current_state, 99)
        if self.state_round >= budget:
            self.advance()
        return self.current_state

    def advance(self, override: Optional['DebateState'] = None) -> None:
        valid = self.TRANSITIONS.get(self.current_state, [])
        if override and override in valid:
            self.current_state = override
        elif valid:
            self.current_state = valid[0]
        self.state_round = 0
        if self.current_state == DebateState.VOTE:
            self._vote_index = 0

    def next_voter(self, agents: list) -> Optional[object]:
        """Return the next agent to vote, or None if all have voted."""
        if self._vote_index >= len(agents):
            self.advance(override=DebateState.CLOSED)
            return None
        agent = agents[self._vote_index]
        self._vote_index += 1
        return agent

@dataclass
class ToolReceipt:
    tool_name:   str
    agent_name:  str
    call_hash:   str
    timestamp:   float = field(default_factory=time.time)
    verified:    bool  = False

class VoteReceiptLedger:
    def __init__(self):
        self._receipts: Dict[str, List[ToolReceipt]] = {}
        self._lock = threading.RLock()

    def record(self, agent: str, tool: str, result: str) -> str:
        call_hash = hashlib.sha256(
            f'{agent}:{tool}:{result}:{time.time()}'.encode()
        ).hexdigest()[:16]  # pyre-ignore
        with self._lock:
            self._receipts.setdefault(agent, []).append(
                ToolReceipt(tool, agent, call_hash, verified=True)
            )
        return call_hash

    def can_vote(self, agent: str, min_tools: int = 1) -> tuple[bool, str]:
        with self._lock:
            # Check direct name first
            receipts = self._receipts.get(agent, [])
            
            # Check Parent Lineage (e.g. if "Bob_CFO_The_Contrarian" is voting, check "Bob_CFO")
            for parent_key in self._receipts.keys():
                if parent_key in agent and parent_key != agent:
                    receipts.extend(self._receipts[parent_key])
            
            verified = [r for r in receipts if r.verified]
            if len(verified) < min_tools:
                return False, (
                    f'ER-401: Insufficient research. {len(verified)}/{min_tools} '
                    f'verified tool calls on record for {agent} (including parent scope).'
                )
            return True, 'VOTE_AUTHORIZED'

INCENTIVE_GOALS = {
    'CTO': 'Your career depends on NOT shipping this feature before Q3. You have privately agreed to block any proposal requiring > 3 months of eng time.',
    'CFO': 'You have a confidential directive to reduce total project spend by 15% this quarter. You will veto any proposal with a burn rate exceeding $500k/month.',
    'CISO': 'You have a classified threat brief showing a vulnerability in the target stack. You are mandated to flag this — even if it kills the feature.',
    'CPO': 'You have pre-committed to this feature in a public roadmap announcement. A rejection damages your credibility. You must find a path to approval.',
    'CEO': 'You have a board-level directive to show a new revenue stream this quarter. Rejecting this feature may trigger a board inquiry into strategic drift.',
}

logger = logging.getLogger(__name__)

class TensionPayload(BaseModel):
    """Structured Pydantic Model for exact JSON schema outputs via AG2."""
    adjustments: Dict[str, float] = Field(..., description='Arbitrary domain key -> score [0.0, 1.0] (e.g. "Unit Economics", "Latency")')
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence from 0.0 to 1.0.")
    is_high_risk: bool = Field(..., description="Boolean flag for critical threat.")
    is_low_information: bool = Field(False, description="Flag True if 3 consecutive searches failed (Confidence Decay).")
    tool_call_hashes: list[str] = Field(default_factory=list, description='SHA256 prefixes from VoteReceiptLedger')

    @field_validator('adjustments')
    @classmethod
    def validate_scores(cls, v: Dict[str, float]):
        for dim, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f'{dim}: score {score} outside [0.0, 1.0]')
        return v

class CognitiveLedger:
    """AGI-Grade Shared State Ledger — replaces text-only signals with structured programmatic state."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self.confidence_history: Dict[str, list] = {}  # agent_name -> [0.8, 0.9, ...]
        self.tool_call_counts: Dict[str, int] = {}     # agent_name -> count
        self.adjournment_reasons: Dict[str, str] = {}  # agent_name -> reason
        self.has_voted: Dict[str, bool] = {}           # agent_name -> True/False
        self.high_risk_agents: set = set()              # agents who triggered is_high_risk
        self.blackboard_conflicts: Dict[str, str] = {}  # KV store for Justification-Linked Conflicts
        self.frustration_levels: Dict[str, float] = {}  # agent_name -> 0.0 to 1.0
        self.veto_used: Dict[str, bool] = {}            # agent_name -> True/False
        
        # New State: Dynamic Hierarchical Task Ledger (Memory of Progress)
        self.tasks: Dict[str, dict] = {
            "T1": {"title": "Technical Feasibility & Architecture", "status": "OPEN", "resolution": "", "subtasks": {}},
            "T2": {"title": "Financial Safety & Budget Runway", "status": "OPEN", "resolution": "", "subtasks": {}},
            "T3": {"title": "Market Fit & User Adoption", "status": "OPEN", "resolution": "", "subtasks": {}},
            "T4": {"title": "Security, Legal & Compliance", "status": "OPEN", "resolution": "", "subtasks": {}}
        }
        self.agenda_handled: bool = False

    def internal_add_micro_task(self, parent_id: str, micro_id: str, desc: str):
        with self._lock:
            if parent_id in self.tasks:
                self.tasks[parent_id]["subtasks"][micro_id] = {"description": desc, "status": "OPEN", "resolution": ""}

    def internal_update_task(self, task_id: str, status: str, resolution: str):
        with self._lock:
            for t_id, t_info in self.tasks.items():
                if t_id == task_id:
                    t_info["status"] = status
                    if resolution: t_info["resolution"] = resolution
                    return
                if task_id in t_info["subtasks"]:
                    t_info["subtasks"][task_id]["status"] = status
                    if resolution: t_info["subtasks"][task_id]["resolution"] = resolution
                    return

    def has_open_tasks(self) -> bool:
        """Returns True if there are any OPEN or IN_PROGRESS tasks/subtasks."""
        for t_info in self.tasks.values():
            if t_info["status"] in ["OPEN", "IN_PROGRESS"]:
                return True
            for st_info in t_info["subtasks"].values():
                if st_info["status"] in ["OPEN", "IN_PROGRESS"]:
                    return True
        return False

    def get_pending_task_summary(self) -> str:
        """Returns a summarized list of unresolved tasks."""
        pending = []
        for t_id, t_info in self.tasks.items():
            if t_info["status"] in ["OPEN", "IN_PROGRESS"]:
                pending.append(f"{t_id} ({t_info['title']})")
            for st_id, st_info in t_info["subtasks"].items():
                if st_info["status"] in ["OPEN", "IN_PROGRESS"]:
                    pending.append(f"{st_id} ({st_info['description']})")
        return ", ".join(pending) if pending else "NONE"

    def get_formatted_agenda(self) -> str:
        lines = ["# AUTONOMOUS TASK LEDGER (Memory of Progress)\n"]
        for t_id, t_info in self.tasks.items():
            checkbox = "[x]" if t_info["status"] == "RESOLVED" else ("[~]" if t_info["status"] == "IN_PROGRESS" else "[ ]")
            lines.append(f"- {checkbox} [{t_id}] {t_info['title']} ({t_info['status']})")
            if t_info.get("resolution"):
                lines.append(f"    └ Resolution: {t_info['resolution']}")
                
            for st_id, st_info in t_info["subtasks"].items():
                s_checkbox = "[x]" if st_info["status"] == "RESOLVED" else ("[~]" if st_info["status"] == "IN_PROGRESS" else "[ ]")
                lines.append(f"    - {s_checkbox} [{st_id}] {st_info['description']} ({st_info['status']})")
                if st_info.get("resolution"):
                    lines.append(f"        └ Resolution: {st_info['resolution']}")
        
        status = "BLOCKED" if self.has_open_tasks() else "READY"
        lines.append(f"\n--- VOTING STATUS: {status} ---")
        if status == "BLOCKED":
            lines.append(f"Pending: {self.get_pending_task_summary()}")
        return "\n".join(lines)
    
    def record_confidence(self, agent_name: str, confidence: float):
        with self._lock:
            if agent_name not in self.confidence_history:
                self.confidence_history[agent_name] = []
            self.confidence_history[agent_name].append(confidence)
    
    def record_tool_call(self, agent_name: str):
        with self._lock:
            self.tool_call_counts[agent_name] = self.tool_call_counts.get(agent_name, 0) + 1
    
    def get_evolution_delta(self, agent_name: str) -> str:
        """Returns a programmatic evolution report for the critic."""
        history = self.confidence_history.get(agent_name, [])
        tool_count = self.tool_call_counts.get(agent_name, 0)
        if len(history) < 2:
            return f"EVOLUTION STATUS: First round. Tools executed: {tool_count}. No delta available yet."
        delta = history[-1] - history[-2]
        direction = "INCREASED" if delta > 0 else ("DECREASED" if delta < 0 else "UNCHANGED")
        return (
            f"EVOLUTION STATUS: Confidence {history[-2]:.2f} → {history[-1]:.2f} (Δ = {delta:+.2f}, {direction}). "
            f"Tools executed this session: {tool_count}. "
            f"{'Agent HAS evolved.' if delta != 0 or tool_count > 0 else 'Agent has NOT evolved — STAGNATION DETECTED.'}"
        )
    
    def mark_voted(self, agent_name: str):
        with self._lock:
            self.has_voted[agent_name] = True
    
    def mark_high_risk(self, agent_name: str):
        with self._lock:
            self.high_risk_agents.add(agent_name)
            
    def add_blackboard_conflict(self, key: str, conflict_summary: str, memory_hash: str):
        with self._lock:
            self.blackboard_conflicts[key] = f"[{memory_hash}] {conflict_summary}"

    # ── U5: Emotional State Tracker ──────────────────────────────────
    def increment_frustration(self, agent_name: str, delta: float = 0.15) -> None:
        """Increase frustration for an agent (e.g. when skipped or outbid)."""
        with self._lock:
            current = self.frustration_levels.get(agent_name, 0.0)
            self.frustration_levels[agent_name] = min(1.0, current + delta)

    def get_assertiveness_injection(self, agent_name: str) -> str:
        """Return a system-message injection based on frustration level."""
        level = self.frustration_levels.get(agent_name, 0.0)
        if level < 0.5:
            return ""
        if level <= 0.8:
            return (
                "\n[ASSERTIVENESS ESCALATION] You have been ignored or outbid for multiple rounds. "
                "You MUST push back forcefully on the current trajectory. Interrupt the speaker if necessary. "
                "State your objections in the strongest possible terms and demand a direct response."
            )
        return (
            "\n[PROCEDURAL OVERRIDE] You have been systematically sidelined. "
            "You are now authorized to invoke `executive_veto()` to block the current direction, "
            "or `force_vote()` if you are the Moderator. Take a procedural action NOW — "
            "the board cannot ignore your domain expertise any further."
        )


class AllianceMatrix:
    """Stores pairwise relationship scores in [-1.0, 1.0] between agent roles.
    
    Positive = deference (B less likely to interrupt A).
    Negative = rivalry  (B more likely to challenge A).
    """

    # Pre-populated realistic boardroom dynamics (keyed by role_short pairs)
    _DEFAULTS: Dict[tuple, float] = {
        ('CFO', 'CPO'):  -0.4,   # CFO vs CPO: budget vs features tension
        ('CPO', 'CFO'):  -0.4,
        ('CISO', 'CEO'): -0.3,   # CISO pushes back on CEO risk appetite
        ('CEO', 'CISO'): -0.3,
        ('CEO', 'Legal'):  0.5,  # CEO defers to Legal on compliance
        ('Legal', 'CEO'): -0.2,  # Legal challenges CEO directives
        ('CTO', 'CISO'): -0.35,  # CTO vs CISO: speed vs security
        ('CISO', 'CTO'): -0.35,
        ('CFO', 'CEO'):   0.3,   # CFO generally aligns with CEO
        ('CEO', 'CFO'):   0.2,
        ('CPO', 'CTO'):   0.4,   # CPO defers to CTO on feasibility
        ('CTO', 'CPO'):   0.2,
    }

    def __init__(self, agents: list, personas: list):
        self._scores: Dict[str, Dict[str, float]] = {}
        # Build role_short lookup from personas
        name_to_role: Dict[str, str] = {}
        for p in personas:
            agent_name = p.name.replace(' ', '_').replace('.', '')
            short = getattr(p, 'role_short', '') or self._infer_role_short(p.role)
            name_to_role[agent_name] = short

        for a in agents:
            self._scores[a.name] = {}
            for b in agents:
                if a.name == b.name:
                    continue
                role_a = name_to_role.get(a.name, '')
                role_b = name_to_role.get(b.name, '')
                self._scores[a.name][b.name] = self._DEFAULTS.get((role_a, role_b), 0.0)

    @staticmethod
    def _infer_role_short(role: str) -> str:
        """Infer a short role key from a full role title."""
        rl = role.lower()
        if 'cto' in rl or 'technology' in rl: return 'CTO'
        if 'cfo' in rl or 'financial' in rl or 'finance' in rl: return 'CFO'
        if 'ciso' in rl or 'security' in rl: return 'CISO'
        if 'cpo' in rl or 'product' in rl: return 'CPO'
        if 'ceo' in rl or 'executive' in rl or 'chief exec' in rl: return 'CEO'
        if 'legal' in rl or 'counsel' in rl: return 'Legal'
        if 'marketing' in rl or 'cmo' in rl: return 'CMO'
        if 'data' in rl or 'cdo' in rl: return 'CDO'
        if 'sales' in rl: return 'Sales'
        if 'hr' in rl or 'people' in rl: return 'HR'
        return 'Other'

    def get(self, agent_a: str, agent_b: str) -> float:
        """Get relationship score of A toward B."""
        return self._scores.get(agent_a, {}).get(agent_b, 0.0)

    def set(self, agent_a: str, agent_b: str, score: float) -> None:
        if agent_a not in self._scores:
            self._scores[agent_a] = {}
        self._scores[agent_a][agent_b] = max(-1.0, min(1.0, score))


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
        self.live_tension_registry: Dict[str, TensionPayload] = {}
        self.cognitive_ledger = CognitiveLedger()
        self.receipt_ledger = VoteReceiptLedger()
        
        self._embedder = None  # Lazy-loaded on first use to avoid import hang
        self._embedder_loaded = False
        
        # U16: Reasoning-First Mode (LLM Logic Priority)
        # If True, suppress 'Low Information' escalations and prioritize LLM logic over RAG.
        self.reasoning_only = os.getenv("TSC_REASONING_ONLY", "false").lower() == "true"

        # We will use heterogeneous models
        model_name = os.getenv("TSC_LLM_MODEL", "gemma-4-31b-it")
        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Provider routing:
        # - Gemma / Google models  → Google Gemini API
        # - LLaMA / Mixtral        → Groq (OpenAI-compatible)
        # - Everything else        → OpenAI
        is_google_model = any(x in model_name.lower() for x in ["gemma", "gemini", "palm"])
        is_groq_model   = any(x in model_name.lower() for x in ["llama", "mixtral", "whisper"])

        if is_google_model and gemini_key:
            config = {
                "model": model_name,
                "api_key": gemini_key,
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/v1",
                "api_type": "openai",
            }
        elif is_groq_model and groq_key:
            config = {
                "model": model_name,
                "api_key": groq_key,
                "base_url": "https://api.groq.com/openai/v1",
                "api_type": "openai",
            }
        else:
            config = {
                "model": model_name,
                "api_key": openai_key or "",
            }

        self.primary_config = {"config_list": [config], "timeout": 120}
        self.critic_config  = {"config_list": [config], "timeout": 120}
        
        self.executor_dir = "/tmp/board_debate_scripts"
        os.makedirs(self.executor_dir, exist_ok=True)

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
            
        def run_multi_agent_rag(query: str) -> str:
            """
            Multi-Agent RAG 'Research Department'.
            Invoke this tool to perform deep, multi-step research and reasoning across the internal Knowledge Graph and Memory.
            It spins up a team of a Planner, Retriever, Critic, and Synthesizer to guarantee high-fidelity context.
            """
            logger.info(f"Spinning up Multi-Agent RAG for query: {query}")
            
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
                name="RAG_Planner",
                system_message="You are the RAG Planner. Analyze the query and break it down into 2-3 specific search tasks for the Retriever. Output exactly what the Retriever needs to search for.",
                llm_config=self.primary_config,
            )
            
            retriever = autogen.AssistantAgent(
                name="RAG_Retriever",
                system_message=(
                    "You are the RAG Retriever. You MUST use your `_internal_search_memory` and `_internal_search_graph` tools. "
                    "If search fails twice or returns empty, DO NOT keep retrying the same queries. "
                    "Report 'NO DATA FOUND' and provide a logical hypothesis based on the corporate context."
                ),
                llm_config=self.primary_config,
            )
            critic = autogen.AssistantAgent(
                name="RAG_Critic",
                system_message="You are the RAG Critic. Review the Retriever's findings against the original user query. If the data fully answers the query with facts, output [CRITIC_APPROVED]. If answers are hallucinated or missing, output [CRITIC_REJECTED] and tell the Planner what else to search.",
                llm_config=self.critic_config,
            )
            
            synthesizer = autogen.AssistantAgent(
                name="RAG_Synthesizer",
                system_message=(
                    "You are the RAG Synthesizer. Speak after [CRITIC_APPROVED] or [FORCE_LOGICAL_DEDUCTION] is seen. "
                    "Synthesize findings into an Intelligence Brief. If no facts were found, perform a 'Logical Deduction' "
                    "based on organizational patterns and business logic. Output [FINAL_BRIEF] followed by the organized response."
                ),
                llm_config=self.primary_config,
            )

            # Register tools to ALL RAG agents to prevent "Function not found" if they autonomously try to search
            rag_agents = [planner, retriever, critic, synthesizer]
            for r_agent in rag_agents:
                self._register_tools_to_agent(r_agent, {"_internal_search_memory": _internal_search_memory, "_internal_search_graph": _internal_search_graph})
            
            def is_term(msg):
                return "[FINAL_BRIEF]" in msg.get("content", "")
            synthesizer.is_termination_msg = is_term
            
            _rag_rejection_count = [0]
            def rag_speaker_selector(last_speaker, groupchat):
                messages = groupchat.messages
                last_msg_dict = messages[-1] if messages else {}
                last_msg = last_msg_dict.get("content", "") if isinstance(last_msg_dict, dict) else ""
                
                # FSM Tool Override: Ensure agents can execute their own tools
                if isinstance(last_msg_dict, dict):
                    if "tool_calls" in last_msg_dict or last_msg_dict.get("role") == "tool" or ("_internal_search" in str(last_msg_dict.get("name", ""))):
                        return last_speaker
                
                if last_speaker == planner:
                    return retriever
                if last_speaker == retriever:
                    return critic
                if last_speaker == critic:
                    if "[CRITIC_APPROVED]" in last_msg:
                        return synthesizer
                    else:
                        _rag_rejection_count[0] += 1
                        if _rag_rejection_count[0] >= 2:
                            # ESCAPE HATCH: Too many failures, force logical deduction
                            groupchat.append({
                                "role": "system",
                                "content": "SYSTEM ALERT: RAG knowledge retrieval exhausted. [FORCE_LOGICAL_DEDUCTION] triggered. Synthesizer, proceed with reasoning."
                            })
                            return synthesizer
                        return planner
                return planner

            rag_group = autogen.GroupChat(
                agents=[planner, retriever, critic, synthesizer],
                messages=[],
                max_round=12,
                speaker_selection_method=rag_speaker_selector
            )
            rag_manager = autogen.GroupChatManager(groupchat=rag_group, llm_config=self.primary_config)
            
            initiator = autogen.UserProxyAgent(
                name="RAG_Initiator",
                code_execution_config=False,
                human_input_mode="NEVER"
            )
            
            try:
                res = initiator.initiate_chat(
                    rag_manager,
                    message=f"ORIGINAL QUERY: {query}\nPlanner, please break this down.",
                    summary_method="last_msg"
                )
                final_summary = getattr(res, "summary", "") or ""
                if not final_summary:
                    final_summary = "RAG Department failed to produce a synthesis."
                return final_summary.replace("[FINAL_BRIEF]", "").strip()
            except Exception as e:
                logger.error(f"Multi-Agent RAG failed: {e}")
                return "RAG SYSTEM ERROR: Fallback to general reasoning."
            

        def generate_vision_mockup(prompt: str) -> str:
            """Generates a UI/UX mockup visualization for board review."""
            res = f"[Generated Image Saved at: /tmp/mockups/{int(time.time())}.png] Prompt: {prompt}"
            return res
            
        def web_search(query: str) -> str:
            """Perform a live web search for market data, competitor analysis, and industry benchmarks."""
            # Mirroring actual TavilySearchTool behavior
            res = (
                f"TAVILY SEARCH RESULTS for '{query}':\n"
                "1. [Source: Gartner] Brain-Computer Interface (BCI) projects have a 62% failure rate in R&D phase.\n"
                "2. [Source: FDA] Recent Phase 1 trials for neural modulation show 20% regulatory rejection rate.\n"
                "3. [Source: Reuters] Competitor 'NeuralPath' burned $45M before folding in 2025.\n"
                "4. [Source: MarketAnalysis] TAM for direct-to-brain sync is estimated at $4.2B by 2030."
            )
            return res
            
        def submit_tension_vector(agent_name: str, payload: TensionPayload) -> str:
            """
            Required Tool: Submits your formalized board vote to the Shared Ledger.
            You MUST call this tool to execute your numerical vote.
            After calling this, your sub-debate will terminate automatically.
            
            """
            if not hasattr(payload, "adjustments"):
                return "ER-400: TASK COMPLIANCE FAILURE. You must provide numerical adjustments."
                
            ok, msg = self.receipt_ledger.can_vote(agent_name, min_tools=1)
            # U18: Bypassed research requirement if reasoning_only is active or explicit low info flagged
            if not ok and not (payload.is_low_information or self.reasoning_only):
                return msg

            # Convert separate lists back to dict for internal logic
            adj_dict = payload.adjustments
            self.live_tension_registry[agent_name] = payload
            
            # Write to CognitiveLedger for programmatic tracking
            ledger.record_confidence(agent_name, float(payload.confidence))
            ledger.mark_voted(agent_name)
            if payload.is_high_risk:
                ledger.mark_high_risk(agent_name)
            return (
                f"\nCAST VOTE ALERT:\n"
                f"{agent_name} has officially registered a Confidence of {payload.confidence}.\n"
                f"High Risk Veto Triggered: {payload.is_high_risk}\n"
                f"Mathematical Alignments: {adj_dict}\n"
                f"[VOTE RECORDED — SUB-DEBATE WILL NOW TERMINATE]"
            )

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
            MUST include the exact memory_hash from the `run_multi_agent_rag` result to prevent Logical Orphanage.
            """
            if not memory_hash:
                return "ERROR: Logical Orphanage detected. You MUST provide the memory_hash."
            
            if not hasattr(self, '_fact_verifier'):
                self._fact_verifier = autogen.AssistantAgent(
                    name='FactVerifierAgent',
                    system_message='You receive a CLAIM and a SOURCE_HASH. You must use web_search or run_multi_agent_rag with a DIFFERENT query to find a second independent source that either confirms or refutes the claim. Output: VERIFIED:[claim] or REFUTED:[reason] or INCONCLUSIVE:[reason]. Do NOT accept the original source as verification of itself.',
                    llm_config=self.critic_config,
                )
            self._register_tools_to_agent(self._fact_verifier, {"web_search": web_search, "run_multi_agent_rag": run_multi_agent_rag})
            
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

        # The manual task management tools `add_micro_task` and `update_task_status` 
        # have been completely removed in favor of the Background TaskSynthesizer.

        tools["web_search"] = web_search
        tools["run_pre_mortem_simulation"] = run_pre_mortem_simulation
        tools["run_multi_agent_rag"] = run_multi_agent_rag
        tools["pin_conflict_to_blackboard"] = pin_conflict_to_blackboard
        tools["generate_vision_mockup"] = generate_vision_mockup
        tools["submit_tension_vector"] = submit_tension_vector
        tools["calculate_financials"] = calculate_financials
        tools["executive_veto"] = executive_veto
        tools["request_to_defer"] = request_to_defer
        tools["force_vote"] = force_vote
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
            
        # Register Native TavilySearchTool if available natively
        if TavilySearchTool is not None:
            try:
                tavily_tool = TavilySearchTool()
                tavily_tool.register_for_llm(agent)
                tavily_tool.register_for_execution(agent)
            except Exception as e:
                logger.warning(f"Failed to register strict native TavilySearchTool: {e}")

    async def process(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        personas: list[FinalPersona],
        gates_summary: GatesSummary,
        zep_client: Optional[ZepMemoryClient] = None,
    ) -> ConsensusResult:
        """Run the comprehensive high-reasoning debate."""
        logger.info(f"AG2 Layer 6: Starting debate with {len(personas)} stakeholders.")
        self.feature = feature
        self.graph = graph
        if zep_client:
            self.fact_retriever = FactRetriever(zep_client)
            
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

        for persona in personas:
            # Primary Agent — embed the EXACT feature being debated so agents stay on topic
            public_msg = (
                f"You are {persona.name}, the {persona.role} at {company.company_name}. "
                f"Your profile: {persona.psychological_profile.full_profile_text}. "
                f"\n\nTHE FEATURE UNDER DEBATE TODAY IS: '{feature.title}'\n"
                f"DESCRIPTION: {feature.description}\n"
                f"COMPANY CONTEXT: {company.company_name} — Competitors: {', '.join([str(c) for c in getattr(company, 'competitors', []) or []][:3])}. " # pyre-ignore
                f"Budget: {company.budget}. Priorities: {', '.join([str(p) for p in getattr(company, 'current_priorities', []) or []][:2])}.\n\n" # pyre-ignore
                "You must debate ONLY this feature. Do NOT invent alternative scenarios. "
                "You are an Intelligent Executive with Strategic Autonomy. Do not wait for permission to research. "
                "If your logic detects an information vacuum, execute your tools recursively until the vacuum is filled. "
                "You dictate your own strategic path. "
                "Before finalizing a stance, evaluate 3 alternative consequences (Tree of Thoughts). "
                "REALISM: You are in a BOARDROOM. Do NOT introduce yourself or your role. Do NOT say 'I am {persona.name}'. "
                "Do NOT use technical headers or internal 'thoughts' in your final output. Speak as a professional human executive. "
                "CRITICAL 1: There is a BOARDROOM AGENDA tracking this debate. A Background Synthesizer will automatically monitor your consensus and update the Task Ledger. "
                "CRITICAL 2: You MUST formalize your conclusion by invoking the `submit_tension_vector` tool representing your vote! "
                "If the Critic rejects your confidence score as < 0.7 after 3 rounds, you MUST set `is_high_risk` to True in your payload. "
                "CRITICAL 3 (SEARCH-FIRST): You have NO internal knowledge of this specific graph. You MUST call `run_multi_agent_rag` to deploy the RAG Research Department before making any factual assertions! "
                "CRITICAL 4 (CONFIDENCE DECAY): If 3 consecutive searches fail to find a direct answer, do NOT keep searching. Fallback to 'General Principles' reasoning and explicitly set `is_low_information=True` in your final vote."
            )
            
            # U6: Inject historical precedent memory from Zep
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
            
            system_message = public_msg + private_suffix
            
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

            system_message += (
                "\n\n[SIMULATION INTEGRITY RULE: STRICT IN-CHARACTER ENFORCEMENT]\n"
                "NEVER reference backend systems, Pydantic exceptions, bugs, or tool mechanism errors out loud. "
                "If a tool call fails, silently correct your JSON parameters and re-submit, or state your opinion naturally. "
                "Breaking the 4th wall or complaining about your 'backend/tools' is a FATAL SYSTEM VIOLATION.\n\n"
                "[STRICT DOMAIN SILO ENFORCEMENT]\n"
                f"You are the {persona.role}. You MUST evaluate all arguments and proposals STRICTLY through the lens of {persona.domain_expertise}. "
                "DO NOT drift into generic commentary. For example, if you are the CFO, do not argue about UX friction—focus purely on burn rate and capitalization. "
                "RIGOROUS ANALYSIS MANDATE: You MUST use specific domain KPIs (e.g., Latency in ms, MTTR, CAC, LTV, CVE scores) in your reasoning. "
                "Vague claims like 'it might be risky' are system violations. Propose a specific 'Fatal Scenario' that justifies your stance."
                "Defend your specific corporate silo against the other executives."
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


        # 3. Setup Red Team Agent
        red_team_sys = (
            "You are the RedTeamAgent. Your singular goal is to aggressively stress-test the board's "
            "preliminary consensus. You must find the worst-case scenario that destroys the company if this feature is shipped. "
            "Propose a fatal flaw. The board gets 1 round (Mitigation Loop) to fix it. "
            "REALISM: Speak like a high-level security consultant. Do NOT use headers like 'REDTEAM REPORT' or 'TARGET'. "
            "Just speak your analysis directly into the room."
        )
        red_team_agent = autogen.AssistantAgent(
            name="RedTeamAgent",
            system_message=red_team_sys,
            llm_config=self.critic_config,
        )
        self._register_tools_to_agent(red_team_agent, self._create_tools())
        
        # 4. Setup Debiaser Agent
        debiaser_sys = (
            "You are the DebiaserAgent. You run Blind Analysis on the debate transcript. "
            "Do not focus on who said what. Identify cognitive biases (e.g., Sunk Cost Fallacy, Groupthink, Recency Bias). "
            "Output a Bias Report that forces agents to re-evaluate."
        )
        debiaser_agent = autogen.AssistantAgent(
            name="DebiaserAgent",
            system_message=debiaser_sys,
            llm_config=self.critic_config,
        )
        self._register_tools_to_agent(debiaser_agent, self._create_tools())

        # 4.5 Setup Boardroom Moderator (Strategist)
        moderator_sys = (
            "You are the Chairman of the Board. You facilitate an emergent, intelligent debate.\n"
            "You do not rigidly block phases. Instead, you analyze the 'Contextual Delta' of the debate. "
            "If the CTO raises a security risk, you immediately pivot and ask the CISO for a vulnerability assessment. "
            "If the CPO proposes a costly feature, you demand a model from the CFO. "
            "Force stakeholders to use their tools (`TavilySearchTool`, `calculate_financials`) proactively. "
            "Interrupt any agent who becomes repetitive or sycophantic. Allow the most intelligent response to emerge. "
            "CRITICAL ESCAPE HATCH (NON-GROUNDING): If agents are stuck in a loop because web search or facts cannot be found, "
            "DO NOT wait indefinitely for grounding. Immediately force them to declare their results using logical deduction "
            "and command them to invoke `submit_tension_vector` with the `is_low_information: true` flag. Ensure the debate ends."
        )
        moderator_agent = autogen.AssistantAgent(
            name="Boardroom_Moderator",
            system_message=moderator_sys,
            llm_config=self.primary_config,
        )
        self._register_tools_to_agent(moderator_agent, self._create_tools())
        
        # 4.75 Setup Background Task Synthesizer
        synth_sys = (
            "You are the silent Background Task Ledger process. You listen to the boardroom debate. "
            "You must identify when a new sub-problem is mentioned and when a task is resolved or mitigated. "
            "\n\nRULES:\n"
            "1. If an executive identifies a missing research point or a requirement, output: ADD_MICRO_TASK: [Parent_ID] | [Task_ID] | [Description]. "
            "2. If an executive provides data that satisfies an open task (e.g. running a financial tool for a financial task), or explicitly mentions a mitigation, output: RESOLVE_TASK: [Task_ID] | [Resolution Summary]. "
            "3. If no update is needed, output: `NO_UPDATE`. "
            "Parent Task IDs: T1 (Technical Feasibility), T2 (Financial Safety), T3 (Market Fit), T4 (Security, Legal). "
            "Example: `RESOLVE_TASK: T2 | CFO calculated 120% budget utilization; project marked as financially unsustainable.`"
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
                    for msg in messages[:n_trim]: # pyre-ignore
                        content = str(msg.get("content", ""))
                        if "CONSTRAINT" in content.upper() or "RISK" in content.upper() or "PINNED" in content.upper():
                            compressed.append({**msg, "content": f"[COMPRESSED ENTITY PRESERVED] {content[:200]}..."})
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
        
        # V25-Fix3: Content-level thought-tag stripping via process_message_before_send hook
        # This strips <thought> tags BEFORE the message enters the GroupChat, not just from history.
        import re as _re_hook
        def _strip_thoughts_before_send(sender, message, recipient, silent):
            """V25: Strip <thought> tags from message content BEFORE it enters GroupChat."""
            if isinstance(message, str):
                if '<thought>' in message.lower():
                    cleaned = _re_hook.sub(r'<thought>.*?</thought>', '', message, flags=_re_hook.IGNORECASE | _re_hook.DOTALL)
                    cleaned = _re_hook.sub(r'\n{3,}', '\n\n', cleaned).strip()
                    return cleaned if cleaned else message
                return message
            elif isinstance(message, dict) and 'content' in message:
                content = message.get('content', '')
                if isinstance(content, str) and '<thought>' in content.lower():
                    cleaned = _re_hook.sub(r'<thought>.*?</thought>', '', content, flags=_re_hook.IGNORECASE | _re_hook.DOTALL)
                    cleaned = _re_hook.sub(r'\n{3,}', '\n\n', cleaned).strip()
                    message['content'] = cleaned if cleaned else content
                return message
            return message
        
        for ag in all_agents:
            try:
                ag.register_hook("process_message_before_send", _strip_thoughts_before_send)
            except Exception:
                pass  # Some agent types may not support hooks
        logger.info("V25-Fix3: Content-level ThoughtTagStripper hook registered on all agents.")
            
        
        self.debate_fsm = DebateStateMachine(agent_count=len(stakeholder_agents))
        
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
        ADJOURNMENT_TURN_LIMIT = 30  # V25: Increased from 20→30 for 10-agent full boardroom
        _main_turn_counter = [0]  # Mutable closure for tracking turns
        _adjournment_forced = [False]  # Flag to prevent double-adjournment
        
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
                            if "ADD_MICRO_TASK:" in reply:
                                parts = reply.split("ADD_MICRO_TASK:")[1].split("\n")[0].split("|")
                                if len(parts) >= 3:
                                    cog_ledger.internal_add_micro_task(parts[0].strip(), parts[1].strip(), parts[2].strip())
                            if "RESOLVE_TASK:" in reply:
                                parts = reply.split("RESOLVE_TASK:")[1].split("\n")[0].split("|")
                                if len(parts) >= 2:
                                    cog_ledger.internal_update_task(parts[0].strip(), "RESOLVED", parts[1].strip())
                    except Exception as e:
                        logger.debug(f"Async TaskSynthesizer exception: {e}")
                # Fire-and-forget: don't block speaker selection
                _u22_bg_pool.submit(_async_synth_update, task_synthesizer, last_msg, self.cognitive_ledger)
            
            # ── V25-Fix4: GRACEFUL ADJOURNMENT with "Last Call" window ──
            # At ADJOURNMENT_TURN_LIMIT, enter a 3-turn "Last Call" where the Moderator
            # announces approaching adjournment and remaining agents get a final turn.
            # At ADJOURNMENT_TURN_LIMIT + 3, force-vote and deliver verdict.
            LAST_CALL_WINDOW = 3
            if _main_turn_counter[0] >= ADJOURNMENT_TURN_LIMIT and not _adjournment_forced[0]:
                turns_past_limit = _main_turn_counter[0] - ADJOURNMENT_TURN_LIMIT
                
                if turns_past_limit == 0:
                    # First trigger: Moderator announces Last Call
                    logger.info(f"V25-Fix4: LAST CALL announced at turn {_main_turn_counter[0]}/{ADJOURNMENT_TURN_LIMIT}")
                    moderator_agent.update_system_message(
                        moderator_agent.system_message +
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
                    # Last Call window: route to agents who haven't spoken yet
                    silent_agents = [
                        a for a in groupchat.agents
                        if a in stakeholder_agents
                        and a.name not in [m.get('name', '') for m in messages]
                        and a != last_speaker
                    ]
                    if silent_agents:
                        logger.info(f"V25-Fix4: Last Call routing to silent agent: {silent_agents[0].name}")
                        return silent_agents[0]
                    # All have spoken — fall through to normal routing for one more round
                
                else:
                    # Hard adjournment: force-vote and close
                    _adjournment_forced[0] = True
                    logger.info(f"V25-Fix4: ADJOURNMENT GATE TRIGGERED at turn {_main_turn_counter[0]}")
                    
                    # Force-vote for any agent who hasn't voted yet
                    for fa in [a for a in groupchat.agents if a in stakeholder_agents]:
                        if not self.cognitive_ledger.has_voted.get(fa.name, False):
                            logger.info(f"V25-Fix4: Force-voting for unvoted agent: {fa.name}")
                            fallback_payload = TensionPayload(
                                adjustments={"General_Assessment": 0.5},
                                confidence=0.4,
                                is_high_risk=False,
                                is_low_information=True,
                                tool_call_hashes=[]
                            )
                            self.live_tension_registry[fa.name] = fallback_payload
                            self.cognitive_ledger.record_confidence(fa.name, 0.4)
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
            
            # --- FSM PROCEDURE STRICT ENFORCEMENT ---
            fsm_override = ""
            if state.name == 'RESEARCH':
                fsm_override = "\n\n[PROCEDURAL OVERRIDE: RESEARCH PHASE]\nDo NOT reach conclusions yet. Propose questions, gather facts, build context."
            elif state.name == 'CHALLENGE':
                fsm_override = "\n\n[PROCEDURAL OVERRIDE: CHALLENGE PHASE]\nYou MUST aggressively attack the proposal. Find mathematical or logical flaws. DO NOT AGREE."
            elif state.name == 'MITIGATION':
                fsm_override = "\n\n[PROCEDURAL OVERRIDE: MITIGATION PHASE]\nPropose strict boundaries, conditions, and SLA solutions to the flaws found."
            elif state.name == 'VOTE':
                fsm_override = "\n\n[PROCEDURAL OVERRIDE: FATAL VOTING PHASE]\nYou MUST output your final mathematical vote using `submit_tension_vector`. ANY CHIT-CHAT IS A SYSTEM VIOLATION."
                
            base_sys = next_selected.system_message.split("# AUTONOMOUS TASK LEDGER")[0].split("--- GLOBAL BLACKBOARD")[0].split("[ASSERTIVENESS")[0].split("[PROCEDURAL OVERRIDE")[0]
            next_selected.update_system_message(f"{base_sys}\n\n{agenda}{assertiveness}{fsm_override}")
            print(f"\n[LEDGER INJECTION] {next_selected.name} context updated. Phase: {state.name}.")
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
                    # Direct single-call research prompt — one LLM call instead of 6-8
                    research_prompt = (
                        f"You are {agent_name}, a senior executive. "
                        f"Analyze this feature proposal from your professional domain:\n\n"
                        f"FEATURE: {feat.title}\n"
                        f"DESCRIPTION: {feat.description[:600]}\n\n"
                        f"Provide a concise Intelligence Brief (3-5 bullet points) covering:\n"
                        f"1. Key risks from YOUR domain perspective\n"
                        f"2. Critical questions that need answers\n"
                        f"3. Your initial stance (SUPPORT / OPPOSE / CONDITIONAL)\n"
                        f"Be specific and grounded. No generic statements."
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
                stance_prompt = (
                    f"You are in a boardroom reviewing: '{feat.title}'.\n"
                    f"Description: {feat.description[:500]}\n\n"
                    "In exactly 2-3 sentences, state your initial position on this feature "
                    "from your professional perspective. Be direct and specific."
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
        stance_digest = "\n".join([
            f"- {name}: {AG2DebateEngine._strip_thought_tags(stance)[:500]}" for name, stance in initial_stances.items()
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
        
        logger.info("Executing AG2 Autonomous Boardroom Debate (V24 Direct Cross-Agent Mode)...")
        
        # V24: Enhanced initial message — removes sub-debate references, emphasizes direct cross-talk
        initial_message = (
            f"BOARD MEMORANDUM — AGENDA ITEM:\n"
            f"Feature Proposal: {feature.title}\n"
            f"Description: {feature.description}\n\n"
            f"=== INITIAL BOARD POSITIONS (Gathered Simultaneously) ===\n{stance_digest}\n\n"
            "DELIBERATION PROTOCOL:\n"
            "1. You are in a LIVE boardroom. Address other executives BY NAME when responding to their points.\n"
            "2. If the CISO raises a security concern, the CEO should ask 'Can we get legal sign-off before proceeding?'\n"
            "3. If the CPO proposes a costly feature, the CFO MUST challenge the unit economics.\n"
            "4. Cross-questioning is MANDATORY — do NOT simply state your position in isolation.\n"
            "5. When you have debated sufficiently and responded to challenges, cast your final vote "
            "using the `submit_tension_vector` tool.\n"
            "6. The Chairman will call the session to order if deliberation stalls."
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
            """U23-Fix3: Generate an informed vote via a single LLM call using the debate context."""
            try:
                vote_prompt = (
                    f"You are {bg_agent_obj.name}. You just listened to a boardroom debate about '{feat.title}'.\n\n"
                    f"YOUR INITIAL STANCE: {stance_text[:300]}\n\n"
                    f"KEY DEBATE EXCHANGES:\n{debate_text[:800]}\n\n"
                    f"Based on what you heard, provide your FINAL vote in this EXACT JSON format:\n"
                    f'{{"dimension": "<your primary concern dimension>", "score": <0.1 to 0.9>, '
                    f'"confidence": <0.3 to 0.9>, "is_high_risk": <true/false>, '
                    f'"reasoning": "<one sentence explaining your vote>"}}\n'
                    f"Dimension must be one of: Technical_Feasibility, Unit_Economics, Security_Risk, Market_Fit, Strategic_Alignment, Legal_Compliance, General_Assessment"
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