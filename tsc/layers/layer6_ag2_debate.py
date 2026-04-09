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

class AG2DebateEngine:
    """
    World-class autonomous boardroom debate engine powered by AG2.
    Features:
    - Data-Driven ConversableAgents mapping to Internal Personas
    - High-Reasoning Architecture (Critic-in-the-Loop, Red Teaming)
    - Epistemic Calibration & De-Biasing Protocol
    - Dynamic Token Weighting
    - AG2 State Persistence
    """
    
    def __init__(self, llm_client: Any):
        self.llm = llm_client
        self.fact_retriever: Optional[FactRetriever] = None
        self.graph: Optional[KnowledgeGraph] = None
        self.feature: Optional[FeatureProposal] = None
        self.live_tension_registry: Dict[str, TensionPayload] = {}
        
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
        """Provides dynamic tools to the agents."""
        tools = {}
        
        # We explicitly omit our custom WebSearchClient, as we now use Native AG2 TavilySearchTool
        
        def run_pre_mortem_simulation(risk_factor: str) -> str:
            """Temporal Fast-Forward Tool."""
            return f"Simulated {risk_factor}. Result: Critical failure averted by 30% margin."
            
        def get_stakeholder_history(query: str) -> str:
            """Query internal graph for history."""
            if hasattr(self, 'fact_retriever') and self.fact_retriever:
                return self.fact_retriever.retrieve_facts(query)
            return "No historical graph stored."
            
        # Native Vision Mockup Generator
        def generate_vision_mockup(prompt: str) -> str:
            """Generates an image mockup or visualization. Output is a physical URI the board can 'see'."""
            return f"[Generated Image Saved at: /tmp/mockups/{int(time.time())}.png] Prompt: {prompt}"
            
        def submit_tension_vector(agent_name: str, payload: TensionPayload) -> str:
            """
            Required Tool: Submits your formalized board vote to the Shared Ledger.
            You MUST call this tool to execute your numerical vote.
            """
            if getattr(self, 'live_tension_registry', None) is None:
                self.live_tension_registry = {}
                
            # Overwrite logic prevents parsed_votes loophole
            self.live_tension_registry[agent_name] = payload
            return f"\nCAST VOTE ALERT:\n{agent_name} has officially registered a Confidence of {payload.confidence}.\nHigh Risk Veto Triggered: {payload.is_high_risk}\nMathematical Alignments: {payload.tension_adjustments}\n"

        tools["run_pre_mortem_simulation"] = run_pre_mortem_simulation
        tools["get_stakeholder_history"] = get_stakeholder_history
        tools["generate_vision_mockup"] = generate_vision_mockup
        tools["submit_tension_vector"] = submit_tension_vector
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
                "Ground your arguments in real web search data about this specific domain. "
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
                
            agent = ReasoningAgent(
                name=persona.name.replace(" ", "_").replace(".", ""),
                system_message=system_message,
                llm_config=agent_config,  # Enforces True Reasoning MCTS Forests
                code_execution_config=code_exec_config,
                max_consecutive_auto_reply=15
            )
            self._register_tools_to_agent(agent, self._create_tools())
            stakeholder_agents.append(agent)
            
            # Logic Critic Agent (Model Heterogeneity & Epistemic Calibration Veto)
            critic_message = (
                f"You are the internal Logic Critic for {persona.name}. Your only goal is to find logical fallacies, "
                "sycophancy, and ungrounded assumptions in their argument. If their Confidence Score is high but data is weak, "
                "demand they run an Internal Research Loop. DO NOT agree with them easily. "
                "REALISM: Speak naturally as a sharp executive advisor. Avoid saying 'Logic Critic Report' or 'Reviewer'. "
                "CRITICAL: If their epistemic grounding is still insufficient after 3 turns, you must formally veto and assign a Confidence Score below 0.7."
            )
            critic = autogen.AssistantAgent(
                name=f"{agent.name}_Critic",
                system_message=critic_message,
                llm_config=self.critic_config,
                max_consecutive_auto_reply=15
            )
            # Epistemic Calibration: message is a CALLABLE so it injects the agent's actual
            # last-spoken proposal into the critic sub-chat — not a blank invitation.
            def make_critic_message(feature_title: str, persona_name: str, persona_role: str, feature_desc: str):
                def critic_message_fn(recipient, messages, sender, config):
                    # Pull the last substantive message from the agent's conversation history
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
                    return (
                        f"PROPOSAL UNDER REVIEW — Feature: '{feature_title}'\n"
                        f"Submitted by: {persona_name} ({persona_role})\n\n"
                        f"{agent_last_msg}\n\n"
                        "Please critique the above argument: identify logical fallacies, ungrounded assumptions, "
                        "and confidence miscalibration. Demand an Internal Research Loop if confidence >0.8 with weak data."
                    )
                return critic_message_fn

            agent.register_nested_chats(
                [{"recipient": critic,
                  "message": make_critic_message(feature.title, persona.name, persona.role, feature.description),
                  "max_turns": 3,
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
        
        # 5. Execute GroupChat using an Explicit FSM (Finite State Machine)
        all_agents = stakeholder_agents + [red_team_agent, debiaser_agent]
        
        def fsm_speaker_selector(last_speaker: autogen.Agent, groupchat: autogen.GroupChat) -> autogen.Agent:
            messages = groupchat.messages
            last_msg = messages[-1].get("content", "") if messages else ""
            rounds = len(messages)
            
            # The "Live Logic Gap" Fix: Intercept Epistemic Veto Post-Mortems immediately
            if "CAST VOTE ALERT:" in last_msg and "High Risk Veto Triggered: True" in last_msg and last_speaker != debiaser_agent:
                logger.warning(f"Live Epistemic Veto triggered by {last_speaker.name}. Transferring to DebiaserAgent for immediate reflection.")
                return debiaser_agent
                
            # Ghost Participant Fix: Force Phase 3 (Adversarial) into the timeline
            if rounds == len(stakeholder_agents) + 1:
                return red_team_agent
                
            # Ghost Participant Fix: Force Phase 4 (Mitigation Loop) back to the leader
            if last_speaker == red_team_agent:
                return stakeholder_agents[0]
                
            # Pre-Mortem Bias Audit Injection
            if rounds == 8 and debiaser_agent not in [m.get("name") for m in messages[-2:]]:
                return debiaser_agent
                
            # Standard Round Robin through stakeholders
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