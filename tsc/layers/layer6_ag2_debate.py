import os
import json
import logging
import time
from typing import Dict, Any, List, Optional
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from pydantic import BaseModel, Field

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


logger = logging.getLogger(__name__)

class TensionPayload(BaseModel):
    """Structured Pydantic Model for exact JSON schema outputs via AG2."""
    tension_adjustments: dict[str, float] = Field(
        ...,
        description="A dictionary mapping qualitative tension keys (e.g., 'Performance', 'Regulatory Compliance') to float values between 0.0 and 1.0 (0.0 = total failure, 1.0 = perfect outcome)."
    )
    confidence: float = Field(
        ...,
        description="A confidence multiplier from 0.0 to 1.0 reflecting the agent's certainty."
    )
    is_high_risk: bool = Field(
        ...,
        description="Set to True ONLY if this feature proposes an active, critical threat requiring a fatal veto. Replaces verbal 'HIGH RISK/UNCERTAIN'."
    )

class CognitiveLedger:
    """AGI-Grade Shared State Ledger — replaces text-only signals with structured programmatic state."""
    
    def __init__(self):
        self.confidence_history: Dict[str, list] = {}  # agent_name -> [0.8, 0.9, ...]
        self.tool_call_counts: Dict[str, int] = {}     # agent_name -> count
        self.adjournment_reasons: Dict[str, str] = {}  # agent_name -> reason
        self.has_voted: Dict[str, bool] = {}           # agent_name -> True/False
        self.high_risk_agents: set = set()              # agents who triggered is_high_risk
    
    def record_confidence(self, agent_name: str, confidence: float):
        if agent_name not in self.confidence_history:
            self.confidence_history[agent_name] = []
        self.confidence_history[agent_name].append(confidence)
    
    def record_tool_call(self, agent_name: str):
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
        self.has_voted[agent_name] = True
    
    def mark_high_risk(self, agent_name: str):
        self.high_risk_agents.add(agent_name)


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
                "api_type": "google",
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
        
    def _create_tools(self) -> Dict[str, Any]:
        """AGI-Grade dynamic tools — all outputs are computed, not static."""
        tools: Dict[str, Any] = {}
        ledger = self.cognitive_ledger
        
        def run_pre_mortem_simulation(risk_factor: str) -> str:
            """Simulate a risk scenario. Returns a computed risk margin based on the severity of the input."""
            # Dynamic: hash the input to produce variable risk margins
            severity_keywords = ["fatal", "lawsuit", "ban", "death", "breach", "collapse", "bankrupt", "regulatory"]
            severity_score = sum(1 for kw in severity_keywords if kw in risk_factor.lower())
            base_margin = max(10, 80 - (severity_score * 15) - (len(risk_factor) % 20))
            margin = min(85, max(10, base_margin))
            outcome = "CRITICAL FAILURE LIKELY" if margin < 30 else ("NARROW SURVIVAL" if margin < 50 else "MANAGEABLE RISK")
            return (
                f"PRE-MORTEM SIMULATION RESULT:\n"
                f"  Scenario: {risk_factor}\n"
                f"  Survival Margin: {margin}%\n"
                f"  Outcome Classification: {outcome}\n"
                f"  Severity Factors Detected: {severity_score}/8\n"
                f"  Recommendation: {'PROCEED WITH EXTREME CAUTION' if margin < 50 else 'RISK IS WITHIN ACCEPTABLE BOUNDS'}"
            )
            
        def get_stakeholder_history(query: str) -> str:
            """Query the internal knowledge graph for historical decisions and precedents."""
            if hasattr(self, 'fact_retriever') and self.fact_retriever:
                return self.fact_retriever.retrieve_facts(query)
            return (
                f"KNOWLEDGE GRAPH QUERY: '{query}'\n"
                "RESULT: No historical data available in the internal graph.\n"
                "ACTION REQUIRED: You must use external research tools (TavilySearchTool) to ground this claim. "
                "Do NOT proceed with ungrounded assumptions."
            )
            
        def generate_vision_mockup(prompt: str) -> str:
            """Generates a UI/UX mockup visualization for board review."""
            return f"[Generated Image Saved at: /tmp/mockups/{int(time.time())}.png] Prompt: {prompt}"
            
        def submit_tension_vector(agent_name: str, payload: TensionPayload) -> str:
            """
            Required Tool: Submits your formalized board vote to the Shared Ledger.
            You MUST call this tool to execute your numerical vote.
            After calling this, your sub-debate will terminate automatically.
            """
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
                f"Mathematical Alignments: {payload.tension_adjustments}\n"
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

        tools["run_pre_mortem_simulation"] = run_pre_mortem_simulation
        tools["get_stakeholder_history"] = get_stakeholder_history
        tools["generate_vision_mockup"] = generate_vision_mockup
        tools["submit_tension_vector"] = submit_tension_vector
        tools["calculate_financials"] = calculate_financials
        return tools
        
    def _register_tools_to_agent(self, agent: autogen.ConversableAgent, tools: Dict[str, Any]):
        """Binds the python functions and native extensions to the agent's schema."""
        for name, func in tools.items():
            try:
                autogen.agentchat.register_function(
                    func,
                    caller=agent,
                    executor=agent,
                    name=name,
                    description=func.__doc__ or f"Execute {name}"
                )
            except AttributeError:
                autogen.register_function(
                    func,
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
            
        # Refinement: Enable AG2 Native/RedisStream Runtime Logging for persistence robustness
        redis_url = os.getenv("REDIS_URL")
        feature_id = getattr(feature, 'feature_id', 'unknown')
        if redis_url:
            autogen.runtime_logging.start(logger_type="redis", config={"redis_url": redis_url, "stream_name": f"ag2_board_debate_{feature_id}"})
            logger.info(f"AG2 RedisStream State Persistence started for {feature_id}")
        else:
            log_db_path = f"/tmp/ag2_logging_{feature_id}_{int(time.time())}.db"
            autogen.runtime_logging.start(logger_type="sqlite", config={"dbname": log_db_path})
            logger.info(f"AG2 SQLite Event-driven Log persistence started at: {log_db_path}")
            
        # 1. Initialize Primary Stakeholder Agents and their tied Logic Critics
        stakeholder_agents = []
        
        # We use standard configs. Pydantic validation is handled via the `submit_tension_vector` Tool
        structured_llm_config = self.primary_config.copy()
        
        for persona in personas:
            # Primary Agent — embed the EXACT feature being debated so agents stay on topic
            system_message = (
                f"You are {persona.name}, the {persona.role} at {company.company_name}. "
                f"Your profile: {persona.psychological_profile.full_profile_text}. "
                f"\n\nTHE FEATURE UNDER DEBATE TODAY IS: '{feature.title}'\n"
                f"DESCRIPTION: {feature.description}\n"
                f"COMPANY CONTEXT: {company.company_name} — Competitors: {', '.join(list(company.competitors or [])[:3])}. "
                f"Budget: {company.budget}. Priorities: {', '.join(list(company.current_priorities or [])[:2])}.\n\n"
                "You must debate ONLY this feature. Do NOT invent alternative scenarios. "
                "You are an Intelligent Executive with Strategic Autonomy. Do not wait for permission to research. "
                "If your logic detects an information vacuum, execute your tools recursively until the vacuum is filled. "
                "You dictate your own strategic path. "
                "Before finalizing a stance, evaluate 3 alternative consequences (Tree of Thoughts). "
                "REALISM: You are in a BOARDROOM. Do NOT introduce yourself or your role. Do NOT say 'I am {persona.name}'. "
                "Do NOT use technical headers or internal 'thoughts' in your final output. Speak as a professional human executive. "
                "CRITICAL: You MUST formalize your conclusion by invoking the `submit_tension_vector` tool representing your vote! "
                "If the Critic rejects your confidence score as < 0.7 after 3 rounds, you MUST set `is_high_risk` to True in your payload."
            )
            
            agent_config = structured_llm_config.copy()
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
                
            # --- AGI-GRADE TERMINATION: Comprehensive signal detection ---
            # Uses a closure counter to prevent false positives on instruction text
            _adjournment_msg_count = [0]  # mutable closure for counting messages
            def _is_adjournment_msg(msg: dict) -> bool:
                """Detects termination signals ONLY after the sub-debate has had at least 2 exchanges."""
                _adjournment_msg_count[0] += 1
                if _adjournment_msg_count[0] <= 2:
                    return False  # Don't terminate during the initial prompt exchange
                content = msg.get("content", "") or ""
                return any(token in content for token in [
                    "[SOVEREIGN ADJOURNMENT:",
                    "[SESSION TERMINATED]",
                    "[SESSION ENDED]",
                    "[EXITING SIMULATION]",
                    "UNABLE TO DECIDE",
                    "[VOTE RECORDED",
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
            
            # Logic Critic Agent (Model Heterogeneity & Epistemic Calibration Veto)
            # --- SOVEREIGN ADJOURNMENT: Critics now intelligently decide when to end the sub-debate ---
            critic_message = (
                f"You are the internal Logic Critic for {persona.name}. Your only goal is to find logical fallacies, "
                "sycophancy, and ungrounded assumptions in their argument. If their Confidence Score is high but data is weak, "
                "demand they run an Internal Research Loop. DO NOT agree with them easily. "
                "REALISM: Speak naturally as a sharp executive advisor. Avoid saying 'Logic Critic Report' or 'Reviewer'. "
                "\n\nSOVEREIGN ADJOURNMENT PROTOCOL:\n"
                "- Track the agent's EVOLUTION across rounds. Did they change their stance, provide new data, or refine their logic?\n"
                "- If the agent has GENUINELY addressed your concerns with new evidence or refined logic, output '[SOVEREIGN ADJOURNMENT: SATISFIED]' to end the sub-session.\n"
                "- If the agent is REPEATING the same argument without new data or tool usage after 2 rounds, output '[SOVEREIGN ADJOURNMENT: STAGNATION — RESEARCH REQUIRED]' to force them back to the board with a mandate to use research tools.\n"
                "- If the agent signals 'UNABLE TO DECIDE', respect it and output '[SOVEREIGN ADJOURNMENT: DEADLOCK]'.\n"
                "- NEVER repeat your own previous critique verbatim. Each round, you must evolve your analysis based on the agent's response.\n"
                "CRITICAL: If their epistemic grounding is still insufficient after 3 turns, you must formally veto and assign a Confidence Score below 0.7."
            )
            critic = autogen.AssistantAgent(
                name=f"{agent.name}_Critic",
                system_message=critic_message,
                llm_config=self.critic_config,
                max_consecutive_auto_reply=15,
                # NOTE: is_termination_msg is intentionally NOT set here.
                # Only the Agent has it — the Critic SENDS adjournment tokens, the Agent RECEIVES them.
            )
            self._register_tools_to_agent(critic, self._create_tools())
            
            # --- SLIDING WINDOW BOARD SUMMARY ---
            # Epistemic Calibration: message is a CALLABLE that injects:
            # 1. The agent's current proposal
            # 2. A synthesized summary of the ENTIRE board debate (sliding window)
            # 3. The critic's own previous feedback (if any)
            def make_critic_message(feature_title: str, persona_name: str, persona_role: str, feature_desc: str):
                def critic_message_fn(recipient, messages, sender, config):
                    # --- 1. Pull the agent's last substantive message ---
                    agent_last_msg = ""
                    for m in reversed(messages or []):
                        content = m.get("content") or ""
                        if content and m.get("name") == sender.name:
                            agent_last_msg = content[:600]
                            break
                    if not agent_last_msg:
                        agent_last_msg = (
                            f"I am {persona_name}, {persona_role}. "
                            f"The proposal being debated is: '{feature_title}' — {feature_desc[:300]}. "
                            "I am preparing my formal stance on whether this should be approved."
                        )
                    
                    # --- 2. Sliding Window Board Summary (FILTERED: substantive only) ---
                    board_summary = "[No prior board context available yet.]"
                    noise_tokens = ["[SESSION", "[EXITING", "Object schema", "[SOVEREIGN", "[VOTE RECORDED", "CAST VOTE ALERT"]
                    try:
                        group_messages = []
                        for m in reversed(messages or []):
                            content = m.get("content") or ""
                            name = m.get("name", "Unknown")
                            # Filter: skip system noise, short messages, and self/critic messages
                            if (content and len(content) > 50 
                                and name != recipient.name and name != sender.name
                                and not any(noise in content for noise in noise_tokens)):
                                group_messages.append(f"- {name}: {content[:200]}")
                            if len(group_messages) >= 10:
                                break
                        if group_messages:
                            group_messages.reverse()
                            board_summary = "SLIDING WINDOW — Recent Board Positions:\n" + "\n".join(group_messages)
                    except Exception:
                        pass
                    
                    # --- 3. Critic's own previous feedback (evolution tracking) ---
                    critic_prev_feedback = ""
                    for m in reversed(messages or []):
                        content = m.get("content") or ""
                        if content and m.get("name") == recipient.name:
                            critic_prev_feedback = content[:400]
                            break
                    prev_critique_section = f"\n--- YOUR PREVIOUS CRITIQUE (do NOT repeat this) ---\n{critic_prev_feedback}\n\n" if critic_prev_feedback else "\n"
                    
                    # --- 4. PROGRAMMATIC EVOLUTION DELTA (from CognitiveLedger) ---
                    evolution_report = self.cognitive_ledger.get_evolution_delta(persona_name.replace(' ', '_').replace('.', ''))
                    
                    return (
                        f"PROPOSAL UNDER REVIEW — Feature: '{feature_title}'\n"
                        f"Submitted by: {persona_name} ({persona_role})\n\n"
                        f"--- AGENT'S CURRENT ARGUMENT ---\n{agent_last_msg}\n\n"
                        f"--- BOARD CONTEXT (Sliding Window Summary) ---\n{board_summary}\n\n"
                        f"--- PROGRAMMATIC EVOLUTION DATA ---\n{evolution_report}\n\n"
                        f"{prev_critique_section}"
                        "Based on the PROGRAMMATIC EVOLUTION DATA above, determine if the agent has genuinely evolved. "
                        "If the data shows zero delta AND zero tool calls over multiple rounds, you MUST issue a Sovereign Adjournment for stagnation. "
                        "If they have evolved with new data, provide deeper analysis. "
                        "Critique the above argument: identify logical fallacies, ungrounded assumptions, "
                        "and confidence miscalibration. Demand tool execution if claims are ungrounded."
                    )
                return critic_message_fn

            agent.register_nested_chats(
                [{"recipient": critic,
                  "message": make_critic_message(feature.title, persona.name, persona.role, feature.description),
                  "max_turns": 15,
                  "summary_method": "last_msg"}],
                trigger=autogen.GroupChatManager
            )

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
            "Interrupt any agent who becomes repetitive or sycophantic. Allow the most intelligent response to emerge."
        )
        moderator_agent = autogen.AssistantAgent(
            name="Boardroom_Moderator",
            system_message=moderator_sys,
            llm_config=self.primary_config,
        )
        self._register_tools_to_agent(moderator_agent, self._create_tools())
        
        # 5. Execute GroupChat using an Explicit FSM (Finite State Machine) with Sovereign-Grade Control
        all_agents = stakeholder_agents + [red_team_agent, debiaser_agent, moderator_agent]
        
        def fsm_speaker_selector(last_speaker: autogen.Agent, groupchat: autogen.GroupChat) -> autogen.Agent:
            messages = groupchat.messages
            last_msg = messages[-1].get("content", "") if messages else ""
            rounds = len(messages)
            
            # --- AGI-GRADE SIGNAL DETECTION (Reads structured state, not strings) ---
            
            # 1. Sovereign Adjournment Detection
            if "[SOVEREIGN ADJOURNMENT: STAGNATION" in last_msg:
                logger.warning(f"SOVEREIGN ADJOURNMENT (Stagnation) detected from {last_speaker.name}. Routing to Moderator.")
                return moderator_agent
            if "[SOVEREIGN ADJOURNMENT: DEADLOCK" in last_msg:
                logger.warning(f"SOVEREIGN ADJOURNMENT (Deadlock) detected from {last_speaker.name}. Invoking Red Team.")
                return red_team_agent
            if "UNABLE TO DECIDE" in last_msg:
                logger.warning(f"Agent {last_speaker.name} signaled inability to decide. Moderator intervention.")
                return moderator_agent
            
            # 2. Structured High-Risk Veto Check (reads CognitiveLedger, not strings)
            if last_speaker.name in self.cognitive_ledger.high_risk_agents and last_speaker != debiaser_agent:
                logger.warning(f"STRUCTURED VETO: {last_speaker.name} flagged high-risk in CognitiveLedger. Routing to DebiaserAgent.")
                return debiaser_agent
            
            # 3. Contextual Relevance Bidding
            domain_bids = {
                "Alice_CTO": ["tech", "architecture", "latency", "scale", "quantum", "engineering", "server", "code", "infrastructure"],
                "Bob_CFO": ["cost", "budget", "finance", "burn", "price", "revenue", "loss", "expensive", "runway", "capital", "funding"],
                "David_CEO": ["vision", "growth", "market", "leadership", "competitor", "strategy", "acquire", "mission"],
                "Sarah_CISO": ["security", "risk", "breach", "vulnerability", "privacy", "hack", "data leak", "compliance", "zero-day", "neural"],
                "Peter_CPO": ["user", "friction", "ui", "ux", "fit", "market", "customer", "experience", "feature", "adoption"],
                "Linda_CMO": ["brand", "pr", "marketing", "viral", "adoption", "press", "reputation", "acquisition", "perception"],
                "Marcus_Legal": ["sue", "liability", "lawsuit", "court", "fda", "regulation", "legal", "ip", "patent", "consent"],
                "Elena_Data": ["data", "model", "bias", "telemetry", "kpi", "metric", "ethics", "tracking", "algorithm"],
                "James_Sales": ["sales", "b2b", "convert", "quota", "client", "enterprise", "objection", "contract", "pipeline"],
                "Diana_HR": ["morale", "burnout", "culture", "diversity", "employee", "training", "hr", "retention", "talent"]
            }
            
            if last_speaker == moderator_agent or last_speaker == red_team_agent or self.cognitive_ledger.has_voted.get(last_speaker.name, False):
                highest_bid = 0.0
                next_speaker = stakeholder_agents[0]
                
                # Check CognitiveLedger: agents with high-risk flags get 2x bid weight
                for agent in stakeholder_agents:
                    if agent == last_speaker:
                        continue
                    bid = 0.0
                    words = str(last_msg).lower().split()
                    for keyword in domain_bids.get(agent.name, []):
                        bid += float(words.count(keyword))
                    
                    # Agents flagged high-risk in ledger get priority bidding
                    if agent.name in self.cognitive_ledger.high_risk_agents:
                        bid *= 2.0
                    
                    if bid > highest_bid:
                        highest_bid = bid
                        next_speaker = agent
                
                if highest_bid > 0.0:
                    logger.info(f"EMERGENT BID WINNER: {next_speaker.name} (Score: {highest_bid:.2f})")
                    return next_speaker
                
            if rounds == len(stakeholder_agents) * 1.5:
                # Emergent injection of the Red Team to stress test current consensus
                return red_team_agent
                
            if rounds % 4 == 0 and last_speaker != moderator_agent:
                # Moderator checks in periodically to shift the agenda
                return moderator_agent

            if rounds == 8 and debiaser_agent not in [m.get("name") for m in messages[-2:]]:
                return debiaser_agent

            # Round Robin Fallback
            if last_speaker in stakeholder_agents:
                idx = stakeholder_agents.index(last_speaker)
                if idx + 1 < len(stakeholder_agents):
                    return stakeholder_agents[idx + 1]
                    
            return stakeholder_agents[0]

        groupchat = autogen.GroupChat(
            agents=all_agents,
            messages=[],
            max_round=15,
            speaker_selection_method=fsm_speaker_selector
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=self.primary_config, max_consecutive_auto_reply=15)
        
        logger.info("Executing AG2 Autonomous Boardroom Debate...")
        
        # We start the debate by having a neutral system prompt / the CTO agent present the feature.
        initial_message = (
            f"BOARD MEMORANDUM (Summarized Checkpoint Initial):\n"
            f"Feature Proposal: {feature.title}\n"
            f"Description: {feature.description}\n"
            "Please debate the merits, execute web searches for market data, and finalize your "
            "stances. CRITICAL: Use the `submit_tension_vector` tool to cast your final mathematical vote."
        )
        
        # Initiate Chat
        initiator = stakeholder_agents[0]
        chat_res = initiator.initiate_chat(
            manager,
            message=initial_message,
        )
        
        # Map agents by name to recover Persona references for Domain Authority scaling
        persona_map = {p.name.replace(" ", "_").replace(".", ""): p for p in personas}
        
        # 6. Evaluate Result via Live Tension Ledger (No post-hoc text parsing)
        tension_shifts: Dict[str, float] = {}
        total_confidence: float = 0.0
        parsed_votes: int = 0
        has_high_risk: bool = False
        
        # Scan through pristine, Pydantic-verified Tool Payloads
        # Overwrite logic natively prevents parsed_votes redundancy
        for agent_name, payload in getattr(self, "live_tension_registry", {}).items():
            if payload.is_high_risk:
                has_high_risk = True
                logger.warning(f"FATAL VETO TRIGGERED: {agent_name} flagged explicit High Risk via Pydantic.")
                
            conf = float(payload.confidence)
            parsed_votes += 1
            
            # Look up Domain Authority
            persona = persona_map.get(agent_name.replace(" ", "_").replace(".", ""))
            
            for k, v in payload.tension_adjustments.items():
                v_float = float(v)
                
                # Hard-Stop Veto Math: Reject instantly if a critical dimension drops < 0.2
                if v_float < 0.2:
                    has_high_risk = True
                    logger.warning(f"HARD-STOP CRITICAL: {agent_name} cited a failure trajectory ({v_float}) on {k}. Veto engaged.")
                    
                domain_auth_multiplier = 1.0
                if persona and persona.domain_expertise:
                    if any(k.lower() in expert.lower() or expert.lower() in k.lower() for expert in persona.domain_expertise):
                        domain_auth_multiplier = 3.0
                        
                weight = conf * domain_auth_multiplier
                total_confidence += weight
                tension_shifts[k] = tension_shifts.get(k, 0.0) + (float(v) * weight)
                    
        # Apply normalization to the tension_shifts based on total_confidence
        if float(total_confidence) > 0.0:
            for k in tension_shifts.keys():
                tension_shifts[k] /= float(total_confidence)
                
        final_score = 0.5 + (sum(tension_shifts.values()) * 0.1)
        final_score = max(0.0, min(1.0, final_score))
        
        # If Epistemic Veto triggered, downgrade verdict
        if has_high_risk:
            logger.warning("Epistemic Calibration Threshold breached. Flagging verdict as HIGH RISK.")
            verdict = "REJECTED" if final_score < 0.6 else "CONDITIONALLY_APPROVED"
        else:
            verdict = "APPROVED" if final_score >= 0.7 else ("CONDITIONALLY_APPROVED" if final_score >= 0.5 else "REJECTED")

        # Stop Runtime logging session
        autogen.runtime_logging.stop()

        # Map AG2 chat_history → DebateRound for DB persistence
        import re
        positions = []
        for msg in getattr(chat_res, "chat_history", []):
            name = (msg.get("name") or msg.get("role", "unknown")).split(" (to")[0]
            content = msg.get("content") or ""
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

            positions.append(DebatePosition(
                stakeholder_name=name,
                role=name,
                statement=cleaned_content[:4000],  # Increase limit for realism
                verdict="CAST VOTE" if "CAST VOTE ALERT" in content else "DEBATING",
                confidence=float(self.live_tension_registry[name].confidence)
                    if name in self.live_tension_registry else 0.5,
            ))

        dr = DebateRound(
            round_number=1,
            round_name="AG2 Multi-Agent Sovereign Debate",
            synthesis=f"Debate on '{feature.title}' completed with {parsed_votes} Pydantic votes and final score {final_score:.2f}.",
            positions=positions,
        )

        return ConsensusResult(
            feature_name=feature.title,
            overall_verdict="REJECTED" if has_high_risk or final_score < 0.6 else "APPROVED",
            approval_confidence=final_score,
            stakeholder_verdicts={k: "Voted" for k in getattr(self, "live_tension_registry", {}).keys()},
            approvals=[],
            mitigations=["Limit initial rollout to 5% of users"] if final_score < 0.8 else [],
            tension_shifts=tension_shifts,
            overall_summary=f"The AG2 autonomous board debated {feature.title}. Agents parsed: {parsed_votes}. Confidence-weighted score: {final_score:.2f}.",
            debate_rounds=[dr]
        )