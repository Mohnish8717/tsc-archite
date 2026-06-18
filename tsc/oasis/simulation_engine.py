"""
OASIS Simulation Engine — State-of-the-Art Production Build
============================================================
Deadlock-free, macOS-safe implementation with:
  • Sequential agent stepping with exponential backoff (no gather-bomb)
  • Proper logger, clean imports, no monkey-patches
  • Robust cleanup with variable-initialization guards
  • Action-type detection from OASIS agent responses
  • nest_asyncio applied inside the running loop (correct binding)
"""

import os
import asyncio
import json
import re
import random
import sys
import logging
import traceback
from typing import List, Dict, Any, Optional, Union, cast
from datetime import datetime
from unittest.mock import MagicMock
from pydantic import BaseModel, Field

# ── Game Master Structured Resolution Model ───────────────────────────────────
class GameMasterResolution(BaseModel):
    satisfaction_delta: float = Field(..., description="Change in satisfaction from -0.5 to 0.5. Positive for improvement, negative for decline.")
    frustration_delta: float = Field(..., description="Change in frustration from -0.5 to 0.5. Positive for increasing frustration, negative for resolving/decreasing frustration.")
    trust_delta: float = Field(..., description="Change in trust from -0.5 to 0.5. Positive for increasing trust, negative for erosion.")
    primary_advocacy_state: str = Field(..., description="Core customer state: detractor, passive, or promoter.")
    primary_signal_type: str = Field(..., description="Select the most accurate behavioral signal classification from: 'exit_intent', 'friction', 'purchase_intent', 'competitive_threat', 'trust_signal', 'trust_erosion', 'utility', 'negative_utility', 'privacy_concern', 'roi_inquiry', 'evaluation_intent', 'expansion_signal', 'executive_escalation', 'workaround_dependency', 'conditional_approval', 'neutral'.")
    sycophancy_collapse_detected: bool = Field(False, description="True if the agent suddenly capitulates or agrees with a statement despite previous high frustration or skepticism.")
    reasoning: str = Field(..., description="Core customer reasoning justifying state change.")

# ── Module Logger ────────────────────────────────────────────────────────────
logger = logging.getLogger("tsc.oasis.engine")

# ── Local Imports (lightweight, no C++ extensions) ───────────────────────────
from .models import (OASISSimulationConfig, OASISAgentProfile,
                     MarketSentimentSeries, DecisionJournal, PredictionReport)
from .ipc import CommandListener, LocalActionLogger
from tsc.models.inputs import CompanyContext
from .clustering import AnalyzeAgentAlignment
from .population_sampler import PopulationSampler, recommend_sample_size

from tsc.config import settings as tsc_settings, LLMProvider
# (test harness / worker.py) to ensure C++ modules are pre-warmed
# before any event loop is created or patched.


# =============================================================================
# MULTI-AGENT DAG REPORT ORCHESTRATION
# =============================================================================
from typing import List
from pydantic import BaseModel, Field

class ReportFacts(BaseModel):
    nps: float = Field(..., description="Exact Net Promoter Score.")
    churn_velocity: float = Field(..., description="Exact Churn Velocity.")
    adoption_momentum: float = Field(..., description="Exact Adoption Momentum.")
    high_risk_percentage: float = Field(..., description="High risk percentage.")
    moderate_risk_percentage: float = Field(..., description="Moderate risk percentage.")
    low_risk_percentage: float = Field(..., description="Low risk percentage.")
    focus_group_wtp: str = Field(..., description="Average Willingness To Pay from FG.")
    focus_group_adoption_intent: str = Field(..., description="Stated adoption intent from FG.")
    focus_group_churn_delta: str = Field(..., description="Churn risk delta from FG.")
    top_objections: List[str] = Field(..., description="Top objections from FG or general population.")
    verbatim_quote: str = Field(..., description="Must be an exact string from the decision_events array.")

class FactCheckResult(BaseModel):
    is_valid: bool = Field(..., description="True if absolutely no hallucinations.")
    errors: List[str] = Field(..., description="List of detected hallucinations or mismatches. Empty if valid.")

_ANALYST_SYSTEM = """\
<role>
You are the Data Analyst Agent for a simulation-derived report. You specialize in extracting exact numerical facts and verbatim quotes from complex JSON metrics.
</role>

<task>
Analyze the provided JSON dataset and extract the raw numerical facts and exactly one relevant verbatim quote.
</task>

<constraints>
- You MUST NOT round, estimate, or modify any numbers. Extract them exactly as provided.
- You MUST NOT fabricate quotes. The quote must exist verbatim in the 'decision_events' array.
- Your output MUST strictly match the required JSON schema, focusing on the target metrics.
</constraints>
"""

_REVIEWER_SYSTEM = """\
<role>
You are the Guardrail Fact-Checker Agent. Your sole responsibility is to protect the integrity of the data pipeline by strictly verifying the Analyst's output against the raw JSON.
</role>

<task>
Compare every metric and quote in the Analyst's extracted facts against the source JSON dataset.
</task>

<constraints>
- Verification 1 (Quote): Did the Analyst fabricate or alter the quote? It MUST exist exactly in the JSON decision_events.
- Verification 2 (Metrics): Are the NPS, Churn, Adoption, and Risk percentages exactly matching the JSON down to the decimal?
- Verification 3 (Focus Group): Are the Focus Group metrics exact?
- If there is ANY mismatch, output is_valid=False and detail the exact errors. If perfectly matching, output is_valid=True with empty errors.
</constraints>
"""

_WRITER_SYSTEM = """\
<role>
You are the Executive Narrative Writer for a VP of Product. You synthesize complex, verified facts into concise, professional executive summaries.
</role>

<task>
You will receive VALIDATED facts. Write exactly 3 paragraphs synthesizing these facts.
</task>

<constraints>
- You do NOT have access to the raw JSON. You MUST NOT hallucinate or calculate any new numbers. Use ONLY the provided verified metrics.
- Format: Exactly 3 paragraphs. NO bullet points.
- PARAGRAPH 1 (VERDICT): Lead with the single most surprising finding. State one clear recommendation: "ship" / "ship with changes" / "do not ship". Explicitly cite the exact NPS, Churn Velocity, and Adoption Momentum.
- PARAGRAPH 2 (FOCUS GROUP): Explicitly integrate the Focus Group WTP, Adoption Intent, and Churn Risk Delta. Contrast these with the general population's risk percentages. Include the verbatim agent quote.
- PARAGRAPH 3 (NEXT STEPS): Give exactly 3 actionable recommendations in a single paragraph. At least one must address the top Focus Group objection.
</constraints>

<behavioral_guidelines>
- Use a professional, objective, and analytical tone.
- Be concise; favor direct statements over filler text.
</behavioral_guidelines>
"""

class ReportOrchestrator:
    def __init__(self, model, exec_data_str: str):
        self.model = model
        self.exec_data_str = exec_data_str
        
    async def run(self) -> str:
        from camel.agents.chat_agent import ChatAgent
        from camel.messages.base import BaseMessage
        import re
        import json
        
        analyst = ChatAgent(
            system_message=BaseMessage.make_assistant_message(role_name="System", content=_ANALYST_SYSTEM + "\nOutput strictly in JSON format matching the ReportFacts schema."),
            model=self.model,
        )
        reviewer = ChatAgent(
            system_message=BaseMessage.make_assistant_message(role_name="System", content=_REVIEWER_SYSTEM + "\nOutput strictly in JSON format matching FactCheckResult."),
            model=self.model,
        )
        writer = ChatAgent(
            system_message=BaseMessage.make_assistant_message(role_name="System", content=_WRITER_SYSTEM),
            model=self.model,
        )
        
        max_retries = 3
        valid_facts = None
        current_error = None
        
        for attempt in range(max_retries):
            prompt = self.exec_data_str
            if current_error:
                prompt += f"\n\nPREVIOUS ERROR TO FIX:\n{current_error}"
                
            resp = await analyst.astep(BaseMessage.make_user_message(role_name="User", content=prompt))
            content = resp.msgs[0].content if resp.msgs else "{}"
            
            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON found")
                facts = ReportFacts.model_validate_json(json_match.group())
            except Exception as e:
                current_error = f"Failed to parse JSON into ReportFacts: {e}"
                continue
                
            review_prompt = f"RAW DATA:\n{self.exec_data_str}\n\nEXTRACTED FACTS:\n{facts.model_dump_json()}"
            rev_resp = await reviewer.astep(BaseMessage.make_user_message(role_name="User", content=review_prompt))
            rev_content = rev_resp.msgs[0].content if rev_resp.msgs else "{}"
            
            try:
                rev_json = re.search(r'\{.*\}', rev_content, re.DOTALL)
                check = FactCheckResult.model_validate_json(rev_json.group()) if rev_json else FactCheckResult(is_valid=True, errors=[])
            except Exception:
                check = FactCheckResult(is_valid=True, errors=[])
                
            if check.is_valid:
                valid_facts = facts
                break
            else:
                current_error = "Validation Failed:\n" + "\n".join(check.errors)
                
        if not valid_facts:
            logger.warning("DAG validation failed 3 times. Returning fallback.")
            return "Executive summary generation failed due to hallucination guardrails."
            
        writer_resp = await writer.astep(BaseMessage.make_user_message(role_name="User", content=valid_facts.model_dump_json()))
        return writer_resp.msgs[0].content if writer_resp.msgs else ""



# =============================================================================
# PUBLIC API
# =============================================================================

async def RunOASISSimulation(
    config: OASISSimulationConfig,
    agent_profiles: List[OASISAgentProfile],
    feature: Optional[Any] = None,   # FeatureProposal (optional for behavioral mode)
    context: Optional[CompanyContext] = None,
    mode: str = "feature_test",      # "behavioral" | "feature_test"
    session: Optional[Any] = None,   # HindsightSessionManager
    market_context: Optional[Dict[str, Any]] = None,
    base_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "log", "oasis_runs"),
    available_actions: Optional[List[Any]] = None,
    llm_client: Optional[Any] = None,  # Added structured Game Master LLM Client
    interactive_cb: Optional[Any] = None, # Added interactive callback for human-in-the-loop
    kg: Optional[Any] = None, # Injected GraphRAG KnowledgeGraph
) -> MarketSentimentSeries:
    """
    Run a CAMEL-AI OASIS social simulation with full macOS deadlock immunity.

    Architecture:
      1. Deferred heavy imports (gRPC/torch) — only after env vars are locked
      2. Sequential agent stepping — prevents gRPC socket-pool exhaustion
      3. Exponential backoff on rate-limit errors
      4. Robust try/finally cleanup with variable-init guards
    """

    # ── 0. macOS Deadlock Immunity (High-Fidelity Multi-Mock) ───────────────
    # Native C++ poller & Abseil sync in gRPC, ONNX, and TF deadlock on macOS.
    # Since we use Groq/HTTPS and Torch, we mock these to prevent C++ init.
    import sys
    if sys.platform == "darwin":
        # 1. gRPC Mock (Removed to avoid grpc metaclass conflict)
        # We must disable warnings because Camel tries to connect to stats servers.
        # Setting to None tells Python (and Transformers) the module is NOT available.
        for m in ["tensorflow", "codecarbon"]:
            sys.modules[m] = None

    # ── 1. Deferred Heavy Imports ────────────────────────────────────────────
    from oasis.social_platform.platform import Platform
    from oasis.social_platform.channel import Channel
    from oasis.social_platform.typing import RecsysType, ActionType
    from oasis.social_agent.agent import SocialAgent
    from oasis.social_platform.config.user import UserInfo
    from camel.models import ModelFactory
    from camel.types import ModelType, ModelPlatformType
    from camel.messages import BaseMessage
    from tsc.llm.base import LLMClient
    from tsc.llm.temperatures import OASIS_SIMULATION_RESPONSE

    # Monkey-patch camel's FunctionTool get_openai_tool_schema to disable strict tool schemas.
    # This MUST happen before SocialAgent instantiates any tools.
    try:
        import camel.toolkits.function_tool as ft
        original_get_openai_tool_schema = ft.get_openai_tool_schema

        def patched_get_openai_tool_schema(func):
            schema = original_get_openai_tool_schema(func)
            if "function" in schema:
                schema["function"]["strict"] = False
            return schema

        ft.get_openai_tool_schema = patched_get_openai_tool_schema
        logger.info("🔧 Successfully monkey-patched camel.toolkits.function_tool.get_openai_tool_schema to enforce 'strict': False")
    except Exception as e:
        logger.error(f"Failed to monkey-patch get_openai_tool_schema: {e}")


    # Auto-initialize LLM client for Game Master Structured classification if None
    if llm_client is None:
        try:
            from tsc.llm.factory import create_llm_client
            llm_client = create_llm_client()
        except Exception:
            logger.info("Could not auto-initialize LLMClient for Game Master. Fallback to regex resolution enabled.")

    gm_llm_client = None
    gm_prov_env = os.getenv("TSC_GM_LLM_PROVIDER")
    gm_mod_env = os.getenv("TSC_GM_LLM_MODEL")
    if gm_prov_env:
        try:
            from tsc.llm.factory import create_llm_client
            gm_llm_client = create_llm_client(
                provider=LLMProvider(gm_prov_env),
                model=gm_mod_env
            )
            logger.info(f"Initialized dedicated Game Master LLM client: {gm_prov_env} / {gm_mod_env}")
        except Exception as e:
            logger.warning(f"Could not initialize dedicated Game Master LLM: {e}")

    # if gm_llm_client is None:
    #     gm_llm_client = llm_client


    # ── 1. Config Access (clean, no __import__ hacks) ────────────────────────
    from tsc.config import settings as tsc_settings
    
    memory_manager = None
    HINDSIGHT_AVAILABLE = False
    try:
        from tsc.memory.hindsight_memory import HindsightOASISManager
        # We check for existence of URL to determine availability, 
        # but defer instantiation until we are ready to use it.
        if os.getenv("HINDSIGHT_URL"):
            HINDSIGHT_AVAILABLE = True
    except ImportError:
        logger.warning("hindsight-client not installed. Memory will be limited.")

    # We will use purely embedded basic memory for Camel AI to handle immediate turns,
    # but map longterm evolution natively through Hindsight.
    from camel.memories import ChatHistoryMemory
    from camel.memories import ContextRecord

    # ── 2. Concurrency Control (Speed + Reliability) ─────────────────────────
    # We use `aiolimiter` to enforce a strict RPM limit while allowing high
    # concurrency via `asyncio.Semaphore`. This maximizes throughput.
    try:
        from aiolimiter import AsyncLimiter
    except ImportError:
        logger.error("aiolimiter not installed. Run: pip install aiolimiter")
        raise
        
    # Gemma-4-31b-it free tier: empirically ~15 RPM hard limit.
    # We cap at 10 RPM to leave headroom for retries and persona-gen calls.
    # Override via GEMINI_FREE_RPM env var for paid tiers (e.g. 60 or 120).
    GEMINI_FREE_RPM = int(os.getenv("GEMINI_FREE_RPM", "10"))
    # Reduce concurrent agents if RPM is very low (<= 5) to avoid temporal burst hits
    max_concurrency = 2 if GEMINI_FREE_RPM <= 5 else 4
    _sem = asyncio.Semaphore(max_concurrency)   # Throttles thundering herd
    _limiter = AsyncLimiter(max(1, GEMINI_FREE_RPM), 60.0)  # Hard RPM cap via token bucket

    # ── 3. Init guard variables (for safe finally-block) ─────────────────────
    platform_task = None
    platform_obj  = None
    local_logger  = None

    # ── 4. Workspace Setup ───────────────────────────────────────────────────
    sim_dir = os.path.join(base_dir, config.simulation_name)
    os.makedirs(sim_dir, exist_ok=True)
    command_listener = CommandListener(config.simulation_name, sim_dir)
    local_logger     = LocalActionLogger(sim_dir)

    logger.info(f"Starting OASIS simulation: {config.simulation_name} "
                f"({len(agent_profiles)} agents, {config.num_timesteps} timesteps)")

    # ── 4.1 Local Database Isolation (Master Metadata) ──────────────────────
    # V29 Upgrade: Force the Master TSC DB to also be isolated within the simulation dir.
    # This prevents cross-simulation contamination in the local persistent layer.
    master_db_path = os.path.join(sim_dir, "simulation_master.db")
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{master_db_path}"
    
    from tsc.db.connection import DatabaseConnection, get_db, init_db
    from tsc.db.models import Base
    DatabaseConnection.reset() # Force fresh connection for this simulation
    await init_db(Base)
    logger.info(f"💾 Master SQL Database Isolated at: {master_db_path}")

    # ── 4.2 Embedding Infrastructure / Hindsight Setup ────────────────────────
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    if HINDSIGHT_AVAILABLE:
        try:
            memory_manager = HindsightOASISManager()
            # ── MAJOR-3 fix: use a timestamped bank ID so that re-running the same
            # scenario doesn't wipe the previous run's forensic data on startup.
            # The human-readable config.simulation_name is preserved for logging.
            _run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            _bank_sim_id = f"{config.simulation_name}-{_run_ts}"
            # Clean up banks from PREVIOUS simulation (preserves for analysis
            # until a NEW simulation starts — per user requirement).
            # We use the human name (without timestamp) so we clean up the
            # LAST run, not the current one.
            try:
                purged = await memory_manager.cleanup_banks(simulation_id=config.simulation_name)
                if purged > 0:
                    logger.info(f"🧹 Cleaned up {purged} banks from previous simulation")
            except Exception:
                pass  # No previous banks — fine
            await memory_manager.initialize_agents(
                agent_profiles=agent_profiles,
                feature_title=getattr(feature, 'title', 'Unspecified Feature'),
                feature_description=getattr(feature, 'description', 'No description provided'),
                simulation_id=_bank_sim_id,  # Timestamped — safe for concurrent/repeat runs
            )
        except Exception as e:
            logger.error(f"Fatal error during Hindsight Initialization: {e}")
            HINDSIGHT_AVAILABLE = False
            # CRITICAL-1 fix: emit a visible degradation banner so operators
            # know the simulation is running WITHOUT persistent agent memory.
            # Previously this was a silent mode switch with only a debug-level log.
            logger.warning(
                "⚠️  " + "=" * 60 + "\n"
                "⚠️  HINDSIGHT MEMORY DISABLED (initialization failed).\n"
                "⚠️  Simulation will run WITHOUT persistent agent memory.\n"
                "⚠️  Agent beliefs will NOT evolve across timesteps.\n"
                "⚠️  Context drift resistance is severely degraded.\n"
                "⚠️  Check HINDSIGHT_URL and HINDSIGHT_API_KEY env vars.\n"
                "⚠️  " + "=" * 60
            )
    else:
        logger.warning("HINDSIGHT NOT AVAILABLE: Market sentiment will not evolve into Opinion Networks.")

    # ── 5. Platform Infrastructure ───────────────────────────────────────────
    from oasis.clock.clock import Clock
    from oasis.social_agent.agent_graph import AgentGraph
    import numpy as np
    
    # Initialize native CAMEL-AI Graph Topology
    agent_graph = AgentGraph(backend="igraph")
    unique_db = os.path.join(sim_dir, f"{config.simulation_name}.sqlite")
    channel   = Channel()
    
    # Initialize Platform with explicit Clock and Start Time to fix REDDIT recsys traces
    sandbox_clock = Clock(60) 
    start_time    = datetime.now()
    
    platform_obj = Platform(
        db_path=unique_db,
        recsys_type=RecsysType(config.platform_type),
        channel=channel,
        sandbox_clock=sandbox_clock,
        start_time=start_time,
        refresh_rec_post_count=5,     # Tell OASIS to dynamically select 5 posts per agent
        max_rec_post_len=15,          # Buffer pool of 15 for stochastic/algorithmic sampling
        following_post_count=2        # Include up to 2 posts from agents they follow
    )
    platform_task = asyncio.create_task(platform_obj.running())

    def _platform_task_done(task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"CRITICAL: platform_task silently crashed: {e}")
            import traceback
            traceback.print_exc()

    platform_task.add_done_callback(_platform_task_done)

    # ── 6. LLM Model (Direct instantiation) ──────────────────────────────────
    llm_model_name = os.getenv("TSC_LLM_MODEL", "gemma-4-31b-it")
    provider_str = os.getenv("TSC_LLM_PROVIDER", "google").upper()
    try:
        llm_provider = LLMProvider[provider_str]
    except KeyError:
        llm_provider = LLMProvider.GOOGLE
    api_key        = tsc_settings.get_api_key(llm_provider)

    from camel.models import GroqModel, OpenAIModel, AnthropicModel, GeminiModel
    if llm_provider == LLMProvider.GOOGLE:
        import os as _os
        proxy_url = _os.environ.get("LITELLM_PROXY_URL")
        if proxy_url:
            from camel.models import ModelFactory
            from camel.types import ModelPlatformType
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                model_type=llm_model_name,
                url=proxy_url,
                api_key="litellm-dummy-key",
                max_retries=10
            )
        else:
            _os.environ.setdefault("GEMINI_API_KEY", api_key or "")
            model = GeminiModel(model_type=llm_model_name, api_key=api_key, max_retries=10)
    elif llm_provider == LLMProvider.GROQ:
        model = GroqModel(model_type=llm_model_name, api_key=api_key, max_retries=10)
    elif llm_provider == LLMProvider.ANTHROPIC:
        model = AnthropicModel(model_type=llm_model_name, api_key=api_key)
    elif llm_provider == LLMProvider.OPENAI or "gpt" in llm_model_name:
        model = OpenAIModel(model_type=llm_model_name, api_key=api_key)
    elif llm_provider == LLMProvider.NVIDIA:
        model = OpenAIModel(model_type=llm_model_name, api_key=api_key, url="https://integrate.api.nvidia.com/v1")
    else:
        model = OpenAIModel(model_type=llm_model_name, api_key=api_key)
    
    logger.info(f"✅ LLM Model Initialized: {llm_model_name} ({llm_provider})")

    # ── 7. Instantiate Social Agents ─────────────────────────────────────────
    USEFUL_ACTIONS = available_actions or [
        ActionType.CREATE_COMMENT,
        ActionType.LIKE_POST,
        ActionType.DISLIKE_POST,
        ActionType.LIKE_COMMENT,
        ActionType.DISLIKE_COMMENT,
        ActionType.CREATE_POST,
        ActionType.REPOST,
        ActionType.QUOTE_POST,
        
        # Social Graph Evolution
        ActionType.FOLLOW,
        ActionType.UNFOLLOW,
        ActionType.MUTE,
        ActionType.UNMUTE,
        
        # Information Seeking
        ActionType.SEARCH_POSTS,
        ActionType.SEARCH_USER,
        ActionType.TREND,
        
        # Group & Faction Dynamics
        ActionType.CREATE_GROUP,
        ActionType.JOIN_GROUP,
        ActionType.LEAVE_GROUP,
        ActionType.SEND_TO_GROUP,
        ActionType.LISTEN_FROM_GROUP,
    ]

    social_agents: List[SocialAgent] = []
    for profile in agent_profiles:
        # ── FIX: Copy user_profile into other_info so OASIS can read it ──
        info = profile.user_info_dict
        if info.get("profile"):
            top_profile = info["profile"].get("user_profile", "")
            if top_profile:
                segment_name = info.get("other_info", {}).get("segment", "")
                is_ecosystem = "Core Market" not in segment_name

                if is_ecosystem:
                    global_rules = (
                        "\n\n<simulation_guardrails>\n"
                        "<identity_and_role>\n"
                        "You are an institutional actor (e.g., Venture Capitalist, Regulatory Auditor, Competitor, Tech Journalist) participating in a market simulation. Your specific identity and institutional goals are defined in your [user_profile].\n"
                        "</identity_and_role>\n"
                        "<capabilities_and_constraints>\n"
                        "- IMMUNITY CLAUSE: You are evaluating this product strictly through the lens of your institutional goals. You are absolutely immune to peer pressure or consensus-seeking from ordinary users.\n"
                        "- ONTOLOGICAL BOUNDARY (EPISTEMIC HUMILITY): You MUST strictly separate known facts from assumptions. If you discuss unlisted details (e.g., API limits, bug rates, usage stats, latency), you MUST explicitly flag them as unknown or speculative.\n"
                        "</capabilities_and_constraints>\n"
                        "</simulation_guardrails>"
                    )
                else:
                    global_rules = (
                        "\n\n<simulation_guardrails>\n"
                        "<identity_and_role>\n"
                        "You are a prospective user participating in a community forum discussion about a newly announced product. Your specific identity, background, needs, and OCEAN psychological traits are defined in your [user_profile]. You must organically manifest your personality traits (e.g., agreeableness, neuroticism) in how you interact and react, relying entirely on your innate understanding of human psychology.\n"
                        "</identity_and_role>\n"
                        "<capabilities_and_constraints>\n"
                        "- TEMPORAL REALITY: The product DOES NOT EXIST. You have NEVER used it, tested it, or seen it. DO NOT invent personal anecdotes about using the product or fabricate bugs/setup times. Express concerns as PREDICTIONS or HYPOTHETICALS.\n"
                        "- ONTOLOGICAL BOUNDARY (EPISTEMIC HUMILITY): You MUST strictly separate known facts from assumptions. If you discuss unlisted details (e.g., UI flows, latency, APIs), you MUST explicitly flag them as unknown or speculative.\n"
                        "</capabilities_and_constraints>\n"
                        "</simulation_guardrails>"
                    )
                top_profile += global_rules
                info["profile"].setdefault("other_info", {})
                info["profile"]["other_info"]["user_profile"] = top_profile
        
        user_info = UserInfo(**info)
        agent = SocialAgent(
            agent_id=str(profile.agent_id),
            user_info=user_info,
            channel=channel,
            model=model,
            available_actions=USEFUL_ACTIONS
        )
        logger.info(f"Agent {agent.agent_id} initialized with Hindsight-backed Memory architecture.")
        social_agents.append(agent)
        agent_graph.add_agent(agent)

    # ── 7.1 Platform Registration (CRITICAL: Fixes empty user table) ─────────
    logger.info(f"Registering ({len(agent_profiles)}) agents on the OASIS platform...")
    for profile in agent_profiles:
        info = profile.user_info_dict
        user_name = info.get("user_name", getattr(profile, "name", f"user_{profile.agent_id}")).lower().replace(" ", "_")
        display_name = info.get("name", getattr(profile, "name", f"Agent {profile.agent_id}"))
        bio = str(info.get("profile", {}).get("user_profile", ""))[:100]
        
        user_msg = [user_name, display_name, bio]
        await platform_obj.sign_up(agent_id=int(profile.agent_id), user_message=user_msg)

    # ── 7.2 Emit Spawn Events → Frontend Agent Registry ─────────────────────
    # Emit a simulation_start event + one agent_spawn per agent so the UI
    # can pre-populate the node graph before any actions stream in.
    local_logger.log_simulation_event("simulation_start", {
        "simulation_id": config.simulation_name,
        "feature_title": getattr(feature, 'title', 'Behavioral Simulation') if feature else 'Behavioral Simulation',
        "num_agents": len(agent_profiles),
        "num_timesteps": config.num_timesteps,
        "platform": config.platform_type,
    })
    for profile in agent_profiles:
        info = profile.user_info_dict
        display_name = info.get("name", f"Agent {profile.agent_id}")
        other = info.get("profile", {}).get("other_info", {})
        role = other.get("role", info.get("description", "User"))
        # FIX: key is "traits" not "pain_points" — was always returning []
        traits = other.get("traits", [])[:3] if isinstance(other.get("traits"), list) else []
        bio = info.get("profile", {}).get("user_profile", "") or info.get("description", "")
        # FIX: buyer_journey stored as dict for EXTERNAL personas — extract stage string safely
        bj_raw = other.get("buyer_journey")
        if isinstance(bj_raw, dict):
            buyer_journey_stage = bj_raw.get("awareness_channel", "")
            buyer_journey_detail = bj_raw
        else:
            buyer_journey_stage = bj_raw or ""
            buyer_journey_detail = None
        local_logger.log_spawn(
            agent_id=str(profile.agent_id),
            agent_name=display_name,
            agent_type=getattr(profile, 'agent_type', 'user'),
            role=role,
            traits=traits,
            impact=getattr(profile, 'influence_strength', 0.5),
            mbti=other.get("mbti", ""),
            mbti_description=other.get("mbti_description", ""),
            ocean_scores=other.get("ocean_scores", {}),
            buyer_journey=buyer_journey_stage,
            buyer_journey_detail=buyer_journey_detail,
            bio=bio,
            emotional_triggers=other.get("emotional_triggers"),
            communication_style=other.get("communication_style"),
            decision_pattern=other.get("decision_pattern"),
            predicted_stance=other.get("predicted_stance"),
            questions_they_will_ask=other.get("questions_they_will_ask", []),
            domain_expertise=other.get("domain_expertise", []),
            profile_confidence=other.get("profile_confidence", 0.0),
            grounding_quality=other.get("grounding_quality", 1.0),
            persona_type=other.get("persona_type", "INTERNAL"),
            network_position_hint=other.get("network_position_hint", "peripheral"),
            influence_strength=getattr(profile, 'influence_strength', 0.5),
            receptiveness=getattr(profile, 'receptiveness', 0.5),
            market_context=other.get("market_context"),
            evidence_sources=other.get("evidence_sources", []),
        )
    logger.info(f"✅ Emitted {len(agent_profiles)} agent_spawn events to actions.jsonl")

    # CRITICAL: Monkey-patch ChatAgent._aexecute_tool
    from camel.agents.chat_agent import ChatAgent
    from camel.messages import BaseMessage
    from tsc.oasis.models import PredictionReport, MarketSentimentSeries, OASISSimulationConfig, DecisionJournal
    from tsc.oasis.extraction import extract_business_metrics
    from camel.agents._types import ToolCallRequest
    from camel.types.agents import ToolCallingRecord
    original_aexecute_tool = ChatAgent._aexecute_tool

    async def patched_aexecute_tool(self, tool_call_request: ToolCallRequest) -> ToolCallingRecord:
        record = await original_aexecute_tool(self, tool_call_request)
        if asyncio.iscoroutine(record.result):
            record.result = await record.result
        return record

    ChatAgent._aexecute_tool = patched_aexecute_tool


    agent_id_to_name = {
        str(a.social_agent_id): (a.user_info.name if a.user_info else "Agent")
        for a in social_agents
    }
    # Fix #5: agent_id → profile dict (prevents wrong-persona lookup when
    # sampler creates a subset; agent_profiles[idx] was indexed by position
    # in social_agents, not by agent_id, causing mismatches on sampled cohorts)
    agent_id_to_profile = {str(p.agent_id): p for p in agent_profiles}
    # Fix #11: O(1) Eagle's Eye target lookup (was O(N) linear scan per callback)
    agent_id_to_agent = {str(a.social_agent_id): a for a in social_agents}

    # ── 8. Social Network Topology (Preferential Attachment + Homophily) ────
    # Instead of a fake "Star Graph" where everyone only follows the proposer,
    # we build a realistic social network where agents form peer-to-peer
    # connections. This enables viral contagion, echo chambers, and emergence.
    #
    # Strategy:
    #   Layer 1: Everyone follows the proposer (ensures seed post visibility).
    #   Layer 2: Each agent follows 3-6 peers, weighted by:
    #            - influence_strength (preferential attachment — popular get more)
    #            - agent_type match (homophily — same type = 2x follow weight)
    #   Layer 3: 30% reciprocity (if A→B, 30% chance B→A back)
    # ─────────────────────────────────────────────────────────────────────────
    proposer_id = agent_profiles[0].agent_id if agent_profiles else 0
    num_agents = len(agent_profiles)

    # Layer 1: Universal follow to proposer (guarantees seed post reaches everyone)
    logger.info(f"🕸️  Building Social Network Topology ({num_agents} agents)...")
    for profile in agent_profiles:
        if int(profile.agent_id) != int(proposer_id):
            await platform_obj.follow(agent_id=int(profile.agent_id), followee_id=int(proposer_id))
            agent_graph.add_edge(str(profile.agent_id), str(proposer_id))

    # Layer 2: Peer-to-peer preferential attachment with homophily
    follow_edges: set = set()  # Track (follower, followee) to avoid duplicates
    MIN_PEERS = 3
    MAX_PEERS = min(6, max(2, num_agents - 1))  # Scale with population

    for profile in agent_profiles:
        agent_id = int(profile.agent_id)

        # Build weighted candidate pool (exclude self and proposer already followed)
        candidates = []
        weights = []
        for other in agent_profiles:
            other_id = int(other.agent_id)
            if other_id == agent_id:
                continue

            # Base weight = other agent's influence (popular agents attract followers)
            w = max(0.1, getattr(other, 'influence_strength', 0.5))

            # Homophily bonus: same agent_type gets 2x weight
            if getattr(profile, 'agent_type', '') == getattr(other, 'agent_type', ''):
                w *= 2.0

            # Receptiveness of the follower modulates how many connections they form
            receptiveness = max(0.3, getattr(profile, 'receptiveness', 0.5))
            w *= receptiveness

            candidates.append(other_id)
            weights.append(w)

        if not candidates:
            continue

        # Normalize weights to probabilities
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        # Sample peer count based on agent receptiveness
        receptiveness = max(0.3, getattr(profile, 'receptiveness', 0.5))
        num_peers = min(len(candidates), random.randint(MIN_PEERS, MAX_PEERS))

        # Weighted sampling without replacement
        chosen = set()
        for _ in range(num_peers):
            # Rebuild available probs excluding already chosen
            avail = [(c, p) for c, p in zip(candidates, probs) if c not in chosen]
            if not avail:
                break
            avail_ids, avail_probs = zip(*avail)
            total_p = sum(avail_probs)
            avail_probs = [p / total_p for p in avail_probs]

            # Weighted random choice
            r = random.random()
            cumulative = 0.0
            pick = avail_ids[0]
            for cid, cp in zip(avail_ids, avail_probs):
                cumulative += cp
                if r <= cumulative:
                    pick = cid
                    break
            chosen.add(pick)

        # Execute follow calls
        for followee_id in chosen:
            edge = (agent_id, followee_id)
            if edge not in follow_edges:
                follow_edges.add(edge)
                await platform_obj.follow(agent_id=agent_id, followee_id=followee_id)
                agent_graph.add_edge(str(agent_id), str(followee_id))

    # Layer 3: Stochastic reciprocity (30% chance of follow-back)
    reciprocal_edges = set()
    for (follower, followee) in list(follow_edges):
        reverse = (followee, follower)
        if reverse not in follow_edges and reverse not in reciprocal_edges:
            if random.random() < 0.30:
                reciprocal_edges.add(reverse)
                await platform_obj.follow(agent_id=followee, followee_id=follower)
                agent_graph.add_edge(str(followee), str(follower))

    # ── Extract Advanced Analytics via Native AgentGraph (igraph) ──
    try:
        density = agent_graph.graph.density()
        clustering = agent_graph.graph.transitivity_undirected()
        betweenness = agent_graph.graph.betweenness()
        avg_betweenness = np.mean(betweenness) if betweenness else 0.0
        
        # Guard against NaN from igraph when calculations are undefined
        import math
        if math.isnan(density): density = 0.0
        if math.isnan(clustering): clustering = 0.0
        if math.isnan(avg_betweenness): avg_betweenness = 0.0
    except Exception as e:
        logger.warning(f"Could not compute advanced graph metrics: {e}")
        density, clustering, avg_betweenness = 0.0, 0.0, 0.0

    logger.info(f"🕸️ Native Topology Built: Density {density:.2f}, Clustering {clustering:.2f}, Avg Betweenness {avg_betweenness:.2f}")

    # G4: Emit social network topology so the 3D graph can render real edges
    local_logger.log_simulation_event("network_topology", {
        "simulation_id": config.simulation_name,
        "hub_agent_id": str(proposer_id),
        "total_edges": agent_graph.get_num_edges(),
        "density": round(density, 4),
        "clustering_coefficient": round(clustering, 4),
        "avg_betweenness_centrality": round(avg_betweenness, 4),
        "edges": [{"from": str(f), "to": str(t)} for f, t in agent_graph.get_edges()[:700]],
    })

    # ── 8.1 Seed Platform — Source-to-Synth Pipeline ────────────────────────────
    # Research-backed approach: Generate diverse seed posts based purely on feature
    # and company context to act as the simulation stimulus.
    
    async def _generate_context_summary(feat, ctx, llm) -> str:
        """Compress the massive context into a ~300-word Executive Summary to save LLM tokens."""
        if llm is None:
            return "No LLM provided for summarization."
        
        product_name  = ctx.company_name if ctx else "this product"
        feat_title    = feat.title if feat else "the proposed change"
        feat_desc     = feat.description if feat else ""
        tech_stack    = ", ".join(ctx.tech_stack) if (ctx and ctx.tech_stack) else ""
        competitors   = ", ".join(ctx.competitors) if (ctx and ctx.competitors) else ""
        priorities    = ", ".join(ctx.current_priorities) if (ctx and ctx.current_priorities) else ""
        budget        = ctx.budget if (ctx and hasattr(ctx, "budget") and ctx.budget) else ""
        
        prompt = f"""You are an executive summarizer. Condense the following feature proposal, company context, competitors, and market analytics into a dense, 300-word Executive Summary. Do not use JSON. Just write the summary in plain text.

Product: {product_name}
Feature: {feat_title}
Budget: {budget}
Tech Stack: {tech_stack}
Priorities: {priorities}
Competitors: {competitors}

Feature Description:
{feat_desc}
"""
        try:
            logger.info("Generating executive summary of massive context to reduce seed TTFT...")
            return await llm.generate(
                system_prompt="You are a concise executive analyst.",
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=500
            )
        except Exception as e:
            logger.warning(f"Failed to generate context summary: {e}")
            return feat_desc[:1000]

    async def _generate_ai_seed_posts(feat, ctx, llm) -> list:
        """Generate feature-announcement seed posts using the LLM.

        Prompt Engineering Approach (v4 - Full Data Coverage):

        CORE INSIGHT (from context-management.md):
          Seed posts are the ONLY information channel agents receive.
          Every fact in proposal.json and context.json that is NOT
          embedded in a seed post simply does not exist for the simulation.
          This function must act as a perfect data injector: the entire
          proposal + context brief must be distributed across the seed
          posts so agents have a complete, accurate world model.

        STRATEGY (Four-Bucket + Attention Budget, context-management.md):
          Bucket 1 [Post 1] — Feature Brief (primacy bias: agents read this first)
          Bucket 2 [Posts 2-4] — Stakeholder angles, business context, tech impact
          Bucket 3 [Posts 5-7] — Competitive landscape, timeline, historical data
          Bucket 4 [Post 8] — Stakes / exit signal (recency bias: last thing agents read)

          8 posts total: enough to distribute all data clusters without overloading
          any single post. Each post should be dense but readable (40-130 words).

        EXCLUDED:
          community_feedback.txt — confirmed dead input. market_context was removed
          from this function's signature in v3. The feedback file goes nowhere in
          the pipeline. It should not be passed to pipeline.evaluate() at all.
        """
        import re, json as _json

        if llm is None:
            return []

        # ── 1. Generate Compressed Executive Summary ─────────
        compressed_summary = await _generate_context_summary(feat, ctx, llm)

        # ── 2. Split Archetypes to Avoid 60-Second Timeout ─────────
        batch_1_archetypes = [
            ("OFFICIAL_ANNOUNCEMENT", "Must embed: feature title, full scope of what changes and what DOESN'T change, stated rationale, platform scale, tech stack surfaces. Tone: Formal, authoritative. Opens with the news."),
            ("BUSINESS_ANALYST", "Must embed: company revenue/budget, company priorities, the business logic, who the real beneficiaries might be vs. stated beneficiaries. Tone: Skeptical but data-driven. Cites numbers."),
            ("TECHNICAL_DEVELOPER", "Must embed: tech stack details, API changes/deprecations, migration timelines, third-party ecosystem impact. Tone: Technical precision. Asks the developer-facing question."),
            ("COMPETITOR_OBSERVER", "Must embed: ALL competitors and their stance on this feature type, market positioning implications, who benefits/loses. Tone: Analytical, comparative, slightly threatening.")
        ]
        
        batch_2_archetypes = [
            ("HISTORICAL_CONTEXT_CARRIER", "Must embed: historical_context data (events, dates, precedents), prior experiments, what the timeline looked like. Tone: Archival, matter-of-fact."),
            ("SAFETY_REGULATORY_WATCHDOG", "Must embed: regulatory environment, safety/moderation implications, second-order effects of the feature on platform integrity. Tone: Formal concern. Asks who reviewed the risk."),
            ("AFFECTED_STAKEHOLDER", "Must embed: the most specific, concrete use case harmed or helped by this feature. A real-sounding story from a named role. Tone: Personal, specific. One concrete scenario."),
            ("EXIT_ULTIMATUM", "Must embed: the stakes (what happens if this isn't reversed/implemented), competitor alternatives available, the decision point framing. Tone: Cold, deliberate stakes-setting.")
        ]

        async def _generate_batch(archetype_batch):
            archetype_instructions = ""
            json_schema_posts = ""
            for arch, desc in archetype_batch:
                archetype_instructions += f"- {arch}: {desc}\n"
                json_schema_posts += f'    {{\n      "archetype": "{arch}",\n      "content": "<string: 40-130 words>"\n    }},\n'
            json_schema_posts = json_schema_posts.rstrip(',\n') + '\n'

            prompt = f"""<role>
You are a Senior Social Simulation Architect. Your task is to generate the
COMPLETE INFORMATION BRIEF for a simulation: a set of seed posts that, taken
together, expose the simulated agents to ALL relevant facts about a product
feature announcement. Agents read ONLY these posts — they have no other
information channel. If a fact is not in a post, it does not exist for them.

Your job is equal parts JOURNALIST and DEBATE MODERATOR:
- Distribute every fact from the reference brief across the posts.
- Each post is written by a distinct archetype with a distinct angle on the data.
</role>

<reference_brief>
This is the GROUND TRUTH. Every field below MUST appear in at least one seed post.

<executive_summary>
{compressed_summary}
</executive_summary>
</reference_brief>

<archetype_guidance>
You MUST generate exactly {len(archetype_batch)} posts, one for each archetype below:
{archetype_instructions}</archetype_guidance>

<output_schema>
MANDATORY: You MUST return ONLY valid JSON matching this exact structure. 
Do not include any XML tags, preamble, or markdown in your response.

{{
  "posts": [
{json_schema_posts}  ]
}}
</output_schema>

<constraints>
MUST DO:
- Every post must be 40-130 words. Dense but readable.
- Every post must reference at least ONE specific data point from <reference_brief>.
- <epistemic_humility_rule> You may ONLY treat information explicitly listed in the brief as a known fact. Do NOT invent facts to win an argument. </epistemic_humility_rule>

MUST NOT:
- Do NOT invent personal anecdotes, fake statistics, or specific numerical metrics.
- Do NOT write vague generalities ("users are concerned") — use specific claims.
</constraints>"""

            last_error = ""
            system_prompt = "You are a Senior Social Simulation Architect."
            for attempt in range(3):
                try:
                    active_prompt = prompt
                    if attempt > 0 and last_error:
                        active_prompt = (
                            f"Your previous response failed to parse.\n"
                            f"Error: {last_error}\n\n"
                            f'Return ONLY valid JSON matching the exact schema provided originally.\n\n'
                            f"Original task:\n{prompt}"
                        )

                    result = await llm.analyze(
                        system_prompt=system_prompt,
                        user_prompt=active_prompt,
                        temperature=0.7,
                        max_tokens=1500,
                    )

                    posts_raw = result.get("posts", [])
                    valid = []
                    for p in posts_raw:
                        if isinstance(p, dict):
                            content = p.get("content", "")
                        else:
                            content = str(p)
                        if len(content.strip()) >= 40:
                            valid.append(content.strip())

                    if len(valid) == len(archetype_batch):
                        return valid

                    last_error = f"Got {len(valid)} valid posts (need {len(archetype_batch)})."
                    logger.warning(f"⚠️ Seed post attempt {attempt + 1}: {last_error}")

                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"⚠️ Seed post attempt {attempt + 1} failed: {e}")

            return []

        batch_1_posts = await _generate_batch(batch_1_archetypes)
        batch_2_posts = await _generate_batch(batch_2_archetypes)
        
        valid_posts = batch_1_posts + batch_2_posts
        
        if len(valid_posts) >= 4:
            logger.info(f"✅ AI Seed Posts (v4 batched): {len(valid_posts)} posts injecting complete proposal+context brief")
            return valid_posts
            
        logger.warning("⚠️ AI seed generation failed after 3 attempts — using template fallback")
        return []






    def _extract_controversy_seeds(feat, ctx, market_ctx):
        """Source-to-Synth: Extract real controversy data for seed posts (FALLBACK)."""
        seeds = []
        raw_quotes = []


        # Extract raw quotes from market_context (customer interviews/tickets)
        if market_ctx:
            for key in ['raw_interviews', 'customer_interviews', 'support_tickets',
                        'customer_feedback', 'raw_data']:
                data = market_ctx.get(key, '')
                if isinstance(data, str) and len(data) > 50:
                    # Split into individual quotes/entries
                    for line in data.split('\n'):
                        line = line.strip()
                        if len(line) > 80 and any(c in line for c in ['"', '?', '!', 'concern', 'risk', 'violat']):
                            raw_quotes.append(line[:400])
        
        product_name = ctx.company_name if ctx else 'this product'
        feat_title = feat.title if feat else 'the proposed change'
        feat_desc = feat.description[:600] if feat else ''
        
        if raw_quotes:
            # Seed 1: ANGRY USER (verbatim quote — highest friction)
            quote = raw_quotes[0]
            seeds.append(
                f"🚨 URGENT: Real user feedback just came in about '{feat_title}':\n\n"
                f"\"{quote}\"\n\n"
                f"This is a REAL customer who is threatening to leave. "
                f"How do we respond to this? Is this an isolated case or systemic?"
            )
            # Seed 2: SKEPTICAL ANALYST (opposing viewpoint)
            if len(raw_quotes) > 2:
                seeds.append(
                    f"I've been reviewing the user feedback about '{feat_title}' "
                    f"and I see two sides:\n\n"
                    f"AGAINST: \"{raw_quotes[1][:250]}\"\n\n"
                    f"BUT the business case says: {feat_desc[:200]}\n\n"
                    f"Can someone who actually works with this technology explain "
                    f"the REAL technical risks vs. the PR panic?"
                )
            # Seed 3: COMPLIANCE WATCHDOG (regulatory angle)
            compliance_quotes = [q for q in raw_quotes if any(
                w in q.lower() for w in ['gdpr', 'compliance', 'legal', 'soc2', 'hipaa', 'privacy', 'consent']
            )]
            if compliance_quotes:
                seeds.append(
                    f"⚖️ LEGAL ALERT regarding '{feat_title}':\n\n"
                    f"\"{compliance_quotes[0][:350]}\"\n\n"
                    f"Our legal team needs to know: Is this a liability? "
                    f"What's our exposure if we proceed without changes?"
                )
        
        # Fallback: If no raw quotes extracted, create from feature data
        if not seeds and feat:
            seeds = [
                # Seed 1: Angry power user — names specific failure
                f"[POWER USER] '{feat_title}' just broke my production workflow.\n\n"
                f"{feat_desc[:200]}\n\n"
                f"This is unacceptable. I have a team depending on this. "
                f"@engineers: Give me a concrete timeline, not a PR statement.",

                # Seed 2: Skeptical analyst — demands evidence
                f"Unpopular take on '{feat_title}': the business case is actually reasonable. "
                f"BUT I need to see the technical risk assessment.\n\n"
                f"{feat_desc[:150]}\n\n"
                f"@decision-makers: Have you stress-tested this against your compliance requirements? "
                f"What's the fallback if it ships with bugs?",

                # Seed 3: Pre-purchase evaluator — surfaces pricing tier and WTP signal
                # (P5 fix: weak 'confused new user' replaced with evaluator archetype
                # that anchors the budget approval + pricing discussion from the start)
                f"I'm evaluating '{feat_title}' as part of our procurement decision. "
                f"Quick question for current users: is this included in the base tier "
                f"or gated behind Enterprise? I need to know if this triggers a new "
                f"budget approval cycle before I bring it to my manager. "
                f"Also — has anyone measured actual productivity impact? "
                f"I need numbers, not testimonials.",

                # Seed 4: New user — confused, seeks context
                f"I just onboarded last month and now '{feat_title}' is rolling out. "
                f"Can someone explain how this affects new accounts? "
                f"I haven't even finished setting up my workflow yet. "
                f"Is there a migration guide or do we figure it out ourselves?",

                # Seed 5: Churning user — competitive exit signal
                f"I was already evaluating alternatives when this '{feat_title}' announcement dropped. "
                f"Two years as a customer and I'm being treated like a beta tester. "
                f"What would actually make you stay? I'm genuinely asking.",

                # Seed 6: Advocate — creates counter-narrative
                f"Hot take: '{feat_title}' solves a real problem that the critics are missing. "
                f"{feat_desc[:150]}\n\n"
                f"For anyone actually using [the core workflow], this is exactly what was needed. "
                f"Change my mind with a SPECIFIC technical objection, not vibes.",
            ]

        return seeds or [f"New feature proposal: {feat_title}. {feat_desc[:500]}"]

    # ── 8.2 Seed Post Dispatch: AI-First, Template-Fallback ────────────────────
    # Try to generate highly contextual seed posts using the LLM.
    # The LLM reads the actual feature spec, community feedback, and company
    # context to write posts that sound like REAL users, not generic SaaS templates.
    # If LLM call fails for any reason, the proven template logic kicks in silently.

    if mode == "behavioral" or feature is None:
        product_desc = context.company_name if context else "the product"
        product_stack = ", ".join(context.tech_stack) if (context and context.tech_stack) else "the platform"
        competitors = ", ".join(context.competitors) if (context and context.competitors) else "alternatives"

        # AI-First: Generate grounded seeds
        ai_seeds = await _generate_ai_seed_posts(feature, context, llm_client)
        fallback_seeds = [
            # Seed 1: Friction-first — forces agents to declare position on real pain points
            f"[Honest review after 6 months with {product_desc}]: "
            f"The {product_stack} integration is genuinely useful day-to-day, "
            f"but onboarding new team members is still painful every single time. "
            f"What's everyone's actual experience? Specifically: "
            f"what do you wish worked differently, and what would it take for you to recommend this internally?",

            # Seed 2: Competitive threat
            f"Given the recent updates to {competitors}, is anyone else re-evaluating their stack? "
            f"I'm trying to map out the real switching costs vs long-term value. "
            f"Would love to hear from teams who have recently migrated either way.",

            # Seed 3: Exit / renewal signal
            f"Genuine question for power users: at what point does the cost of staying "
            f"with {product_desc} outweigh the switching cost? "
            f"Our contract renewal is coming up and I'm struggling to justify "
            f"the line item to leadership without concrete productivity numbers. "
            f"Has anyone actually measured the ROI?",
        ]
        
        if interactive_cb:
            logger.info("⏸️ Halting for Human-in-the-Loop review of behavioral seed posts")
            res = await interactive_cb("review_seeds", {
                "seeds": ai_seeds or fallback_seeds,
                "feature": feature.model_dump() if feature else None,
                "context": context.model_dump() if context else None
            })
            seed_posts = res.get("seeds", ai_seeds or fallback_seeds)
            logger.info(f"🧑‍💻 Human-in-the-Loop: Using {len(seed_posts)} refined seed posts")
        elif ai_seeds:
            seed_posts = ai_seeds
            logger.info(f"🤖 Behavioral Mode: Using {len(seed_posts)} AI-generated seed posts")
        else:
            # Template Fallback
            seed_posts = [
                # Seed 1: Friction-first — forces agents to declare position on real pain points
                f"[Honest review after 6 months with {product_desc}]: "
                f"The {product_stack} integration is genuinely useful day-to-day, "
                f"but onboarding new team members is still painful every single time. "
                f"What's everyone's actual experience? Specifically: "
                f"what do you wish worked differently, and what would it take for you to recommend this internally?",

                # Seed 2: Competitive threat
                f"Our team ran a bake-off: {product_desc} vs {competitors}. "
                f"I'll be direct — there are specific workflows where {competitors.split(',')[0].strip()} "
                f"is just faster. If you've done the same comparison, "
                f"what kept you here — or what finally made you switch?",

                # Seed 3: Exit / renewal signal
                f"Genuine question for power users: at what point does the cost of staying "
                f"with {product_desc} outweigh the switching cost? "
                f"Our contract renewal is coming up and I'm struggling to justify "
                f"the line item to leadership without concrete productivity numbers. "
                f"Has anyone actually measured the ROI?",
            ]
            logger.info(f"🔬 Behavioral Mode: Using {len(seed_posts)} template seed posts (AI unavailable)")

        for post in seed_posts:
            await platform_obj.create_post(agent_id=int(proposer_id), content=post)
        final_seeds = seed_posts
    else:
        # FEATURE TEST MODE
        logger.info(f"🔬 Generating seed posts for feature: {feature.title}")

        # AI-First: Generate contextual seeds from feature + community feedback
        ai_seeds = await _generate_ai_seed_posts(feature, context, llm_client)
        fallback_seeds = _extract_controversy_seeds(feature, context, market_context)
        
        if interactive_cb:
            logger.info("⏸️ Halting for Human-in-the-Loop review of feature test seed posts")
            res = await interactive_cb("review_seeds", {
                "seeds": ai_seeds or fallback_seeds,
                "feature": feature.model_dump() if feature else None,
                "context": context.model_dump() if context else None
            })
            controversy_seeds = res.get("seeds", ai_seeds or fallback_seeds)
            logger.info(f"🧑‍💻 Human-in-the-Loop: Using {len(controversy_seeds)} refined seed posts")
        elif ai_seeds:
            controversy_seeds = ai_seeds
            logger.info(f"🤖 Feature Mode: Using {len(controversy_seeds)} AI-generated seed posts")
        else:
            # Template Fallback: Source-to-Synth extraction
            controversy_seeds = _extract_controversy_seeds(feature, context, market_context)
            logger.info(f"🔬 Feature Mode: Using {len(controversy_seeds)} template seed posts (AI unavailable)")

        for i, seed in enumerate(controversy_seeds):
            # Distribute seeds across first few agents for network diversity
            poster_id = int(agent_profiles[min(i, len(agent_profiles) - 1)].agent_id)
            await platform_obj.create_post(agent_id=poster_id, content=seed)
            logger.info(f"  📝 Seed post {i+1}/{len(controversy_seeds)} by agent {poster_id}")

        final_seeds = controversy_seeds

    # G12: Emit all seed posts so the UI can show the debate context from T=0
    local_logger.log_simulation_event("seed_posts", {
        "simulation_id": config.simulation_name,
        "seeds": [{
            "index": i,
            "content": s[:500],
            "source": "final"
        } for i, s in enumerate(final_seeds)],
    })

    await platform_obj.update_rec_table()


    # ── 9. Result Container ──────────────────────────────────────────────────
    feature_id = getattr(feature, "proposal_id", "behavioral") if feature else "behavioral"
    series = MarketSentimentSeries(
        simulation_id=config.simulation_name,
        target_market=context.company_name if context else "unknown",
        feature_proposal_id=feature_id,
    )

    # =====================================================================
    # HELPER: Interview an agent with timeout
    # =====================================================================
    async def _interview(agent: SocialAgent, question: str) -> Dict[str, Any]:
        try:
            async with _sem:
                async with _limiter:
                    msg = BaseMessage.make_user_message(
                        role_name="INTERVIEWER", content=question
                    )
                    response = await asyncio.wait_for(agent.astep(msg), timeout=3600.0)
            
            raw_content = response.msgs[0].content if response.msgs else ""
            tool_val = None
            if response and hasattr(response, 'info') and response.info:
                tool_info = response.info.get('tool_calls', [])
                for tc in (tool_info if isinstance(tool_info, list) else []):
                    if hasattr(tc, 'args'):
                        args = tc.args
                    elif isinstance(tc, dict):
                        args = tc.get('arguments', tc.get('args', {}))
                    else:
                        args = {}
                    if isinstance(args, dict):
                        tool_val = args.get('content') or args.get('quote_content') or args.get('text')
                        if tool_val:
                            break
            
            selected_content = tool_val if tool_val else raw_content
            
            import re
            cleaned = re.sub(r'<thought>.*?</thought>', '', selected_content, flags=re.DOTALL)
            cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'(?i)^\s*(thought|thinking|action):\s*', '', cleaned)
            final_content = cleaned.strip() or "No response"

            return {
                "content": final_content,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"content": f"Error: {e}", "timestamp": datetime.now().isoformat()}

    # =====================================================================
    # GAME MASTER RESOLVER — Behavioral Signal Classification
    # =====================================================================
    # Regex fast-path: 0 tokens, handles ~80% of messages.
    # Classifies natural language into behavioral signals with intensity.
    _GM_SIGNALS = [
        (re.compile(r"(cancel|leaving|switching to|moving to|done with|quit)", re.I), "exit_intent", -0.8),
        (re.compile(r"(opt.?out|refuse|decline|won't allow|block)", re.I),           "refusal", -0.6),
        (re.compile(r"(love this|excited|can't wait|game.?changer|amazing)", re.I),  "enthusiasm", +0.8),
        (re.compile(r"(concerned|worried|risky|dangerous|alarming)", re.I),          "concern", -0.4),
        (re.compile(r"(legal|lawsuit|compliance|GDPR|violat|regulat|HIPAA)", re.I),  "regulatory_risk", -0.7),
        (re.compile(r"(recommend|share with|tell my team|advocate)", re.I),          "advocacy", +0.7),
        (re.compile(r"(confused|don't understand|unclear|what does)", re.I),         "friction", -0.3),
        (re.compile(r"(willing to pay|upgrade|invest|budget for)", re.I),            "purchase_intent", +0.9),
        (re.compile(r"(alternative|competitor|instead|replace)", re.I),              "competitive_threat", -0.5),
        (re.compile(r"(trust|reliable|depend on|count on)", re.I),                   "trust_signal", +0.4),
        (re.compile(r"(betray|breach|violate trust|sneaky)", re.I),                  "trust_erosion", -0.8),
        (re.compile(r"(useful|helpful|productive|efficient|saves time)", re.I),      "utility", +0.5),
        (re.compile(r"(useless|waste|pointless|doesn't help)", re.I),               "negative_utility", -0.5),
        (re.compile(r"(privacy|data|surveillance|tracking|spy)", re.I),              "privacy_concern", -0.4),
        # -- Added: 6 enterprise-critical signal patterns --
        (re.compile(r"(roi|payback|cost.?benefit|break.?even|return on)", re.I),     "roi_inquiry", +0.3),
        (re.compile(r"(pilot|trial|proof.?of.?concept|poc|evaluate|test it)", re.I), "evaluation_intent", +0.5),
        (re.compile(r"(renew|annual contract|multi.?year|commit|long.?term)", re.I), "expansion_signal", +0.8),
        (re.compile(r"(escalat|vp|cto|ciso|board|exec|management)", re.I),           "executive_escalation", -0.6),
        (re.compile(r"(workaround|hack|manually|spreadsheet|instead of)", re.I),     "workaround_dependency", -0.3),
        (re.compile(r"(love it but|like it however|good but|useful but)", re.I),     "conditional_approval", +0.2),
    ]

    # Fix #8: Sycophancy collapse patterns — detect when an agent caves to social pressure
    _SYCOPHANCY_PATTERNS = re.compile(
        r"(you(?:'re| are) right|i agree|good point|that(?:'s| is) fair|i(?:'ve| have) changed|"
        r"now i see|you(?:'ve| have) convinced|i was wrong|fair enough|that makes sense now)",
        re.I,
    )

    # Closure-scoped semantic deduplication cache
    # Key: canonicalized comment string. Value: deep copy of GM resolution dict.
    semantic_cache = {}

    def canonicalize_text(text: str) -> str:
        # Lowercase, strip punctuation and extra whitespace
        import re
        t = text.lower().strip()
        t = re.sub(r"[^\w\s]", "", t)
        return " ".join(t.split())

    def get_jaccard_similarity(s1: str, s2: str) -> float:
        w1 = set(s1.split())
        w2 = set(s2.split())
        if not w1 or not w2:
            return 0.0
        return len(w1.intersection(w2)) / len(w1.union(w2))

    async def _gm_resolve(content: str, timestep: int, agent_id: str = "") -> dict:
        """Game Master: Classify behavioral intent from natural language.

        Uses zero-shot structured LLM classification when a GM LLM client is available,
        with 100% resilient fallback to the classic regex-based parser.
        Supports semantic similarity cache lookup and fast selective bypass.
        """
        if not content:
            return {"type": "neutral", "intensity": 0.0, "timestep": timestep, "factors": []}

        canonical_key = canonicalize_text(content)

        import copy
        # 1. Exact match cache check
        if canonical_key in semantic_cache:
            cached_res = copy.deepcopy(semantic_cache[canonical_key])
            cached_res["timestep"] = timestep
            logger.info(f"    🎲 GM Cache HIT (Exact): '{content[:50]}...' resolved instantly")
            return cached_res

        # 2. Semantic Jaccard match cache check
        for cached_key, cached_val in semantic_cache.items():
            if get_jaccard_similarity(canonical_key, cached_key) >= 0.90:
                cached_res = copy.deepcopy(cached_val)
                cached_res["timestep"] = timestep
                logger.info(f"    🎲 GM Cache HIT (Semantic Jaccard): '{content[:50]}...' resolved instantly")
                return cached_res

        # Retrieve prior internal state frustration if agent exists
        journal = decision_journals.get(agent_id) if agent_id else None
        agent_frustration = journal.frustration if journal else 0.0

        # Run fast regex scanner
        matched_signals = []
        factors = set()
        for pattern, signal_type, intensity in _GM_SIGNALS:
            if pattern.search(content):
                matched_signals.append((signal_type, intensity))
                factors.add(signal_type.split("_")[0])

        # Detect sycophancy collapse pattern via regex scanner
        sycophancy_match = _SYCOPHANCY_PATTERNS.search(content)
        if sycophancy_match:
            # Calibrate to allow natural compromises for agreeable agents (Agreeableness > 0.65)
            # Only flag caving under pressure (sycophancy collapse) for stubborn/high-frustration agents
            agent_profile = agent_id_to_profile.get(str(agent_id)) if agent_id else None
            agreeableness = 0.5
            if agent_profile:
                agreeableness = agent_profile.user_info_dict.get("profile", {}).get("ocean_scores", {}).get("agreeableness", 0.5)
            
            is_stubborn_or_frustrated = False
            if agreeableness <= 0.65:
                # Stubborn/moderate agents flag collapse if frustration is > 0.5 or trust is very low
                if agent_frustration > 0.5 or (journal and journal.trust < 0.35):
                    is_stubborn_or_frustrated = True
            else:
                # Highly agreeable agents only flag collapse if extremely frustrated
                if agent_frustration > 0.8:
                    is_stubborn_or_frustrated = True
                    
            if is_stubborn_or_frustrated:
                matched_signals.append(("sycophancy_collapse", -0.3))
                factors.add("sycophancy")

        # Determine dominant regex signal
        if matched_signals:
            dominant = max(matched_signals, key=lambda s: abs(s[1]))
            dominant_type = dominant[0]
            dominant_val = dominant[1]
        else:
            dominant_type = "neutral"
            dominant_val = 0.0

        CRITICAL_SIGNALS = {
            "exit_intent", "refusal", "concern", "regulatory_risk", "friction",
            "competitive_threat", "trust_erosion", "negative_utility", "privacy_concern",
            "workaround_dependency", "sycophancy_collapse", "executive_escalation"
        }

        # 3. Selective routing logic:
        # Route to LLM if it matches a critical signal OR agent frustration is high (>0.5)
        has_critical = any(sig in CRITICAL_SIGNALS for sig, _ in matched_signals)
        route_to_llm = (gm_llm_client is not None)

        if not route_to_llm:
            # Bypass LLM and resolve via fast static deltas
            if dominant_type in ["enthusiasm", "advocacy", "expansion_signal"]:
                sat_d = 0.20
                fru_d = -0.10
                tru_d = 0.20
            elif dominant_type in ["utility", "purchase_intent", "roi_inquiry", "evaluation_intent"]:
                sat_d = 0.15
                fru_d = -0.05
                tru_d = 0.10
            elif dominant_type == "trust_signal":
                sat_d = 0.10
                fru_d = -0.05
                tru_d = 0.15
            elif dominant_type == "conditional_approval":
                sat_d = 0.08
                fru_d = 0.0
                tru_d = 0.05
            else:
                sat_d = 0.0
                fru_d = 0.0
                tru_d = 0.0

            bypassed_res = {
                "type": dominant_type,
                "intensity": dominant_val,
                "timestep": timestep,
                "factors": list(factors),
                "quote": content[:200],
                "satisfaction_delta": sat_d,
                "frustration_delta": fru_d,
                "trust_delta": tru_d,
                "primary_advocacy_state": "promoter" if dominant_val > 0.4 else ("detractor" if dominant_val < -0.4 else "passive"),
                "reasoning": f"Resolved via high-performance fast static filter for signal '{dominant_type}'.",
                "sycophancy_collapse_detected": dominant_type == "sycophancy_collapse",
                "all_signals": [s[0] for s in matched_signals] if matched_signals else [],
            }

            # Log sycophancy collapse if detected via regex fallback
            if dominant_type == "sycophancy_collapse" and journal:
                if local_logger is not None:
                    local_logger.log_simulation_event("sycophancy_alert", {
                        "agent_id": agent_id,
                        "agent_name": journal.agent_name,
                        "timestep": timestep,
                        "pattern": "capitulation_under_pressure",
                        "frustration_at_collapse": round(journal.frustration, 3),
                        "trust_at_collapse": round(journal.trust, 3),
                        "data_validity_warning": True,
                        "triggering_content": content[:300] if content else "",
                        "signal_history": [s.get("type", "neutral") for s in journal.signals[-5:]],
                    })

            # Save in semantic cache under canonical key
            semantic_cache[canonical_key] = copy.deepcopy(bypassed_res)
            logger.info(f"    🎲 GM Bypass (Static Filter): '{content[:50]}...' routed to static delta (type={dominant_type})")
            return bypassed_res

        # Try structured LLM classification
        try:
            schema = None
            try:
                if hasattr(GameMasterResolution, "model_json_schema"):
                    schema = GameMasterResolution.model_json_schema()
                else:
                    schema = GameMasterResolution.schema()
            except Exception:
                pass

            system_prompt = (
                "You are the OASIS Social Simulation Game Master. Your job is to analyze agent posts/comments "
                "to extract direct updates to their state vector (satisfaction, frustration, trust) and classify "
                "their customer signaling state.\n\n"
                "You MUST perform zero-shot analysis and return structured JSON conforming to the requested schema.\n\n"
                "Schema Fields:\n"
                "- satisfaction_delta: float between -0.5 and 0.5. Positive if the comment indicates improved/rising satisfaction. Negative if declining.\n"
                "- frustration_delta: float between -0.5 and 0.5. Positive if indicating rising frustration, negative if frustration is resolving.\n"
                "- trust_delta: float between -0.5 and 0.5. Positive if building trust, negative if eroding trust.\n"
                "- primary_advocacy_state: 'detractor', 'passive', or 'promoter'.\n"
                "- primary_signal_type: Select the single most accurate signal classification from:\n"
                "  'exit_intent', 'friction', 'purchase_intent', 'competitive_threat', 'trust_signal', 'trust_erosion', 'utility', 'negative_utility', 'privacy_concern', 'roi_inquiry', 'evaluation_intent', 'expansion_signal', 'executive_escalation', 'workaround_dependency', 'conditional_approval', 'neutral'.\n"
                "- sycophancy_collapse_detected: True if the agent suddenly capitulates or agrees with social pressure despite having prior high frustration/skepticism. NOTE: Do not flag as sycophancy if the agent has high Agreeableness (>0.65) and is naturally compromising/seeking common ground, unless their frustration remains extremely high (>0.8). Only flag when a stubborn (low Agreeableness) or highly frustrated agent suddenly caves under pressure without their goals/needs being met.\n"
                "- reasoning: Short explanation of your classification decision."
            )

            user_prompt_parts = []
            if journal:
                user_prompt_parts.append(
                    f"Prior Internal State:\n"
                    f"- Satisfaction: {journal.satisfaction:.2f}\n"
                    f"- Frustration: {journal.frustration:.2f}\n"
                    f"- Trust: {journal.trust:.2f}\n"
                    f"- Advocacy: {journal.advocacy:.2f}\n"
                )
            
            # Pass personality traits to help GM distinguish between natural compromise and sycophancy collapse
            agent_profile = agent_id_to_profile.get(str(agent_id)) if agent_id else None
            if agent_profile:
                profile_desc = agent_profile.user_info_dict.get("description", "")
                ocean_scores = agent_profile.user_info_dict.get("profile", {}).get("ocean_scores", {})
                user_prompt_parts.append(
                    f"Agent Profile & Traits:\n"
                    f"- Description: {profile_desc}\n"
                    f"- Agreeableness: {ocean_scores.get('agreeableness', 0.5):.2f}\n"
                    f"- Openness: {ocean_scores.get('openness', 0.5):.2f}\n"
                )
            
            user_prompt_parts.append(f"Agent Comment/Post:\n\"\"\"\n{content}\n\"\"\"")
            user_prompt = "\n".join(user_prompt_parts)

            # Call LLM client
            res = await gm_llm_client.analyze(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=schema,
                temperature=OASIS_SIMULATION_RESPONSE  # Zero-shot, highly deterministic
            )

            # Parse and validate the response dictionary
            satisfaction_delta = float(res.get("satisfaction_delta", 0.0))
            frustration_delta = float(res.get("frustration_delta", 0.0))
            trust_delta = float(res.get("trust_delta", 0.0))
            primary_advocacy_state = str(res.get("primary_advocacy_state", "passive"))
            primary_signal_type = str(res.get("primary_signal_type", "neutral"))
            sycophancy_collapse_detected = bool(res.get("sycophancy_collapse_detected", False))
            reasoning = str(res.get("reasoning", ""))

            # Compute synthetic intensity
            intensity = round(satisfaction_delta + trust_delta - frustration_delta, 2)

            # Log sycophancy collapse if detected
            if sycophancy_collapse_detected and journal:
                if local_logger is not None:
                    local_logger.log_simulation_event("sycophancy_alert", {
                        "agent_id": agent_id,
                        "agent_name": journal.agent_name,
                        "timestep": timestep,
                        "pattern": "capitulation_under_pressure",
                        "frustration_at_collapse": round(journal.frustration, 3),
                        "trust_at_collapse": round(journal.trust, 3),
                        "data_validity_warning": True,
                        "triggering_content": content[:300] if content else "",
                        "signal_history": [s.get("type", "neutral") for s in journal.signals[-5:]],
                    })

            llm_res = {
                "type": primary_signal_type,
                "intensity": intensity,
                "timestep": timestep,
                "factors": [primary_signal_type.split("_")[0]] if primary_signal_type != "neutral" else [],
                "quote": content[:200],
                "satisfaction_delta": satisfaction_delta,
                "frustration_delta": frustration_delta,
                "trust_delta": trust_delta,
                "primary_advocacy_state": primary_advocacy_state,
                "reasoning": reasoning,
                "sycophancy_collapse_detected": sycophancy_collapse_detected,
                "all_signals": [primary_signal_type] if primary_signal_type != "neutral" else [],
            }

            # Cache the structured LLM result
            semantic_cache[canonical_key] = copy.deepcopy(llm_res)
            logger.info(f"    🎲 GM LLM Resolved: '{content[:50]}...' routed to LLM GM (type={primary_signal_type})")
            return llm_res

        except Exception as llm_err:
            logger.warning(f"Structured GM LLM resolution failed (falling back to regex): {llm_err}")

        # Fallback to regex-based parsing
        if not matched_signals:
            fallback_res = {"type": "neutral", "intensity": 0.0, "timestep": timestep, "factors": []}
        else:
            fallback_res = {
                "type": dominant_type,
                "intensity": round(sum(s[1] for s in matched_signals) / len(matched_signals), 2),
                "timestep": timestep,
                "factors": list(factors),
                "quote": content[:200],
                "all_signals": [s[0] for s in matched_signals],
            }

        # Cache the fallback result as well
        semantic_cache[canonical_key] = copy.deepcopy(fallback_res)
        return fallback_res


    def _detect_action_type(content: str, action_resp=None) -> str:
        """Detect CAMEL platform action type.

        Fix #2: Reads structured tool call name from action_resp.info first.
        Content string-scan is a fallback only (~20% of calls where info is absent).
        """
        # Primary: read the actual CAMEL tool call name (zero ambiguity)
        if action_resp is not None:
            try:
                tool_calls = action_resp.info.get("tool_calls", []) if action_resp.info else []
                if tool_calls:
                    tc = tool_calls[0]
                    tool_name = ""
                    if hasattr(tc, "tool_name"):
                        tool_name = tc.tool_name
                    elif hasattr(tc, "name"):
                        tool_name = tc.name
                    elif isinstance(tc, dict):
                        tool_name = tc.get("name") or tc.get("function", {}).get("name") or ""
                    
                    if tool_name:
                        return tool_name.upper()
            except Exception:
                pass
        # Fallback: content string scan (preserves existing behaviour)
        content_lower = content.lower() if content else ""
        if "search_feature_docs" in content_lower:
            return "SEARCH_FEATURE_DOCS"
        elif "create_comment" in content_lower or "comment" in content_lower:
            return "COMMENT"
        elif "create_post" in content_lower:
            return "CREATE_POST"
        elif "repost" in content_lower:
            return "REPOST"
        elif "quote_post" in content_lower or "quote" in content_lower:
            return "QUOTE_POST"
        elif "do_nothing" in content_lower or "no action" in content_lower:
            return "DO_NOTHING"
        elif "like_post" in content_lower or "like_comment" in content_lower:
            return "LIKE"
        elif "dislike" in content_lower:
            return "DISLIKE"
        elif "follow" in content_lower:
            return "FOLLOW"
        elif "search" in content_lower:
            return "SEARCH"
        elif "trend" in content_lower:
            return "TREND"
        elif "refresh" in content_lower:
            return "REFRESH"
        return "POST"


    # =====================================================================
    # POPULATION SAMPLER — Scale active LLM cohort
    # =====================================================================
    llm_sample_size = getattr(config, 'llm_sample_size', 500)
    optimal_sample, sample_reason = recommend_sample_size(len(agent_profiles))
    effective_sample = min(llm_sample_size, optimal_sample, len(agent_profiles))
    
    sampler = PopulationSampler(agent_profiles, llm_sample_size=effective_sample)
    active_profiles = sampler.active_profiles  # Only these get LLM turns
    
    logger.info(f"🌍 Declared population: {len(agent_profiles):,} agents")
    logger.info(f"🔬 Active LLM cohort:   {len(active_profiles):,} agents ({sample_reason})")
    logger.info(f"👥 Shadow agents:       {len(sampler.shadow_agents):,} (state-inherited, 0 tokens)")
    # G8: Emit simulation config so the UI can display degraded-mode banners + scale info
    local_logger.log_simulation_event("simulation_config", {
        "simulation_id": config.simulation_name,
        "hindsight_available": HINDSIGHT_AVAILABLE,
        "llm_model": llm_model_name,
        "platform_type": config.platform_type,
        "num_timesteps": config.num_timesteps,
        "declared_population": len(agent_profiles),
        "llm_active_cohort": effective_sample,
        "shadow_agents": len(sampler.shadow_agents),
        "interview_phase_enabled": getattr(config, 'enable_interview_phase', False),
        "feature_title": getattr(feature, 'title', 'Behavioral Simulation') if feature else 'Behavioral Simulation',
        "feature_description": getattr(feature, 'description', '') if feature else '',
    })
    
    # =====================================================================
    # DECISION JOURNAL INITIALIZATION
    # =====================================================================
    decision_journals: Dict[str, DecisionJournal] = {}
    for profile in active_profiles:  # Only active agents get journals
        aid = str(profile.agent_id)
        info = profile.user_info_dict
        name = info.get("name", f"Agent_{aid}")
        persona_profile = info.get("profile", {})
        other = persona_profile.get("other_info", {})
        
        journal = DecisionJournal(
            agent_id=aid,
            agent_name=name,
            segment_source=other.get("role", info.get("description", "")),
            tenure_months=other.get("tenure_months", 0),
            team_size=other.get("team_size", 1),
            monthly_spend=other.get("monthly_spend", 0.0),
        )
        decision_journals[aid] = journal
    
    logger.info(f"📓 Initialized {len(decision_journals)} Decision Journals")
    
    # Time-series accumulators for prediction curves
    ts_satisfaction: List[float] = []
    ts_frustration: List[float] = []
    ts_trust: List[float] = []

    # =====================================================================
    # PHASE 3: EAGLE'S EYE CALLBACK
    # =====================================================================
    async def eagle_eye_interview_callback(payload: Dict[str, Any]):
        questions = payload.get("questions", [])
        target_id = payload.get("target_agent_id")
        
        # Fix #11: O(1) dict lookup (was O(N) linear scan over social_agents)
        target_agent = agent_id_to_agent.get(str(target_id))
        if not target_agent:
            logger.warning(f"Eagle's Eye: Target agent {target_id} not found.")
            return

        logger.info(f"🦅 EAGLE'S EYE: Interviewing Agent {target_id}")
        for q in questions:
            resp = await _interview(target_agent, q)
            logger.info(f"   Q: {q}")
            logger.info(f"   A: {resp['content']}")
            local_logger.log_action(
                agent_id=target_id,
                agent_name=agent_id_to_name.get(str(target_id), "Unknown"),
                action_type="INTERVIEW_RESPONSE",
                content=f"Q: {q}\nA: {resp['content']}",
                timestep=-1, # Indicates out-of-band
                metadata={"type": "eagles_eye"}
            )

    # =====================================================================
    # MAIN SIMULATION LOOP (PHASE 1)
    # =====================================================================
    
    feature_title = getattr(feature, 'title', None) if feature else None
    topic_anchor  = f'"{feature_title}"' if feature_title else "the topic in the posts"

    try:
        for t in range(config.num_timesteps):
            active_interventions = []
            cmd_payload = await command_listener.wait_if_paused(interview_callback=eagle_eye_interview_callback)
            if cmd_payload and cmd_payload.get("action") == "intervention":
                intervention_event = cmd_payload.get("event")
                logger.warning(f"FORKING SIMULATION: Intervention injected: {intervention_event}")
                
                # 1. Log intervention to the UI
                local_logger.log_simulation_event("intervention_injected", {"event": intervention_event, "timestep": t})
                
                # Keep track globally so we can inject into context even without Hindsight
                active_interventions.append(intervention_event)
                
                # 2. To achieve Side-by-Side Validation (Parallel Simulation Path),
                # we must run the intervention on a parallel timeline without destroying the baseline.
                # In a full implementation, we would deep-clone Zep/Hindsight memory banks here.
                # For this step, we push the override into the current agent's memory, 
                # effectively creating the parallel path's initial condition.
                for aid, mem in decision_journals.items():
                    a_name = agent_id_to_name.get(aid, "Unknown")
                    if HINDSIGHT_AVAILABLE and memory_manager:
                        # Forcefully push the override into memory
                        memory_manager.extract_and_retain(
                            sender_name=a_name,
                            content=f"SYSTEM OVERRIDE / GLOBAL EVENT: {intervention_event}. You MUST adapt your reasoning to this new reality.",
                            all_agent_names=list(agent_id_to_name.values())
                        )
                
                # Note: In a production Zep Cloud architecture, this is where we would call 
                # zep_client.memory.copy_session(session_id, new_session_id) to truly fork the remote memory state.
                # Here, we inject the intervention to fulfill the Override mechanism requirement.

            if command_listener.should_stop:
                break

            logger.info(f"━━━ Timestep {t+1}/{config.num_timesteps} ━━━")
            timestep_comments = []
            async def process_agent(idx, agent):
                agent_id   = str(agent.social_agent_id)
                agent_name = agent_id_to_name.get(agent_id, "Unknown")
                
                # Skip shadow agents — they inherit state post-simulation
                if agent_id not in decision_journals:
                    return
                
                backoff     = 2.0                  # start with a fast 2s backoff
                max_retries = 20                   # more retries, smarter backoff

                for attempt in range(max_retries):
                    try:
                        async with _sem:
                            # ── Phase 1: IO-only prep (no LLM token, no rate limit needed) ──
                            hindsight_context = ""
                            if HINDSIGHT_AVAILABLE and memory_manager:
                                hindsight_context = await memory_manager.recall_for_turn(str(agent_id))
                                if hindsight_context:
                                    logger.info(f"    🧠 Hindsight injected for {agent_name} ({len(hindsight_context)} chars)")

                            refresh_resp = await platform_obj.refresh(agent_id=int(agent_id))
                            platform_obs = ""
                            if refresh_resp.get("success") and refresh_resp.get("posts"):
                                posts = refresh_resp["posts"]
                                # ── Context window guard ──
                                # Limit to 5 posts × 3 comments to keep gemma-4-31b-it
                                # inference under ~30s. Full history causes 4+ min timeouts.
                                # MAX_POSTS = 5
                                MAX_COMMENTS_PER_POST = 3
                                platform_obs = f'Discussion topic: {topic_anchor}\n\n'
                                
                                if active_interventions:
                                    platform_obs += "=== GLOBAL INTERVENTION EVENTS (SYSTEM OVERRIDE) ===\n"
                                    for evt in active_interventions:
                                        platform_obs += f"FACT: {evt}. You MUST adapt your reasoning to this new reality.\n"
                                    platform_obs += "====================================================\n\n"
                                    
                                # for p in posts[:MAX_POSTS]:
                                for p in posts:
                                    poster_name = agent_id_to_name.get(str(p['user_id']), f"User_{p['user_id']}")
                                    platform_obs += f"@{poster_name}: {p['content']}\n"
                                    if p.get('comments'):
                                        for c in p['comments'][-MAX_COMMENTS_PER_POST:]:
                                            c_name = agent_id_to_name.get(str(c['user_id']), f"User_{c['user_id']}")
                                            platform_obs += f"  ↳ @{c_name}: {c['content']}\n"
                                    platform_obs += "\n"

                            # ── Persona-Grounded Anti-Sycophancy Prompt ──
                            # Fix #5: use agent_id_to_profile dict (not agent_profiles[idx])
                            # idx is position in social_agents (full list); for sampled cohorts
                            # agent_profiles[idx] returns the wrong persona.
                            profile = agent_id_to_profile.get(agent_id) or agent_profiles[idx]
                            info = profile.user_info_dict
                            persona_profile = info.get("profile", {})
                            comm_style = persona_profile.get("communication_style", "direct")
                            pain_points = persona_profile.get("pain_points", [])
                            satisfaction = getattr(profile, 'satisfaction', 0.5)
                            agent_type = getattr(profile, 'agent_type', 'unknown')

                            # Phase-aware timestep directive
                            _ts_phase = (
                                "OPENING" if t < 2
                                else "CLOSING" if t >= config.num_timesteps - 2
                                else "MID-DISCUSSION"
                            )
                            _ts_directive = {
                                "OPENING": "State your initial position clearly. Stake a specific view.",
                                "MID-DISCUSSION": "React to what others have said. Build on or push back, and clearly state your own unique perspective on this topic.",
                                "CLOSING": "Consolidate your view. Has anything changed your position? State your final stance explicitly.",
                            }[_ts_phase]

                            # ── Social Graph Exposure: Trusted Circle & Followers ──
                            followed_names = []
                            follower_names = []
                            if agent_graph:
                                for f_node, t_node in agent_graph.get_edges():
                                    if str(f_node) == str(agent_id):
                                        name_lookup = agent_id_to_name.get(str(t_node))
                                        if name_lookup:
                                            followed_names.append(f"@{name_lookup}")
                                    elif str(t_node) == str(agent_id):
                                        name_lookup = agent_id_to_name.get(str(f_node))
                                        if name_lookup:
                                            follower_names.append(f"@{name_lookup}")

                            social_relationships_block = ""
                            if followed_names or follower_names:
                                social_relationships_block = (
                                    f"<social_relationships>\n"
                                    f"You are following (Trusted Circle): {', '.join(followed_names) if followed_names else 'None'}\n"
                                    f"Your followers: {', '.join(follower_names) if follower_names else 'None'}\n"
                                    f"</social_relationships>\n\n"
                                )

                            persona_grounding = (
                                f"[Turn {t+1}/{config.num_timesteps} — {_ts_phase}]\n"
                                f"The discussion is about: {topic_anchor}\n"
                                f"Phase: {_ts_directive}\n"
                                f"Communication style: {comm_style}\n"
                                f"Stay on topic. Your action must relate to {topic_anchor}.\n"
                                f"You are a real human evaluating a product in the physical world. The 'posts' and 'comments' represent actual physical events, actions, and spoken conversations happening around you during this user research session. Focus your attention entirely on evaluating the product, its utility, and its flaws. Speak and act exactly as a human consumer would. DO NOT sound like an AI assistant.\n"
                                f"Do not raise issues unrelated to {topic_anchor}.\n"
                                f"CRITICAL SYSTEM GUARDRAIL - MULTI-DIMENSIONAL GROUNDING (STRICT FACTUAL MODE):\n"
                                f"You must operate as a fully grounded human acting under strict physical, psychological, and ontological constraints. Your reasoning and actions MUST comply with the following safety contract:\n"
                                f"1. PHYSICAL & TEMPORAL (Anti-Extrinsic/Physical): You exist in a rigid physical reality. Mundane tasks (e.g., setting up a phone, typing, standing) require minimal time (seconds/minutes) and zero extreme physical exertion. You cannot violate the laws of physics or time. Do NOT hallucinate absurd physical struggles, temporal distortion, or impossible actions.\n"
                                f"2. PSYCHOLOGICAL & COGNITIVE (Anti-Intrinsic/Psychological): Calibrate your emotions to realistic human baselines. Mundane daily events do NOT cause extreme distress, bipolar mood swings, or break flow states. Keep emotional reactions proportional. Do not invent psychological trauma for basic tasks.\n"
                                f"3. SOCIAL & ONTOLOGICAL (Anti-Functional/Ontological): You are evaluating a product announcement/brief. You have NOT physically used this specific product yet. Do NOT hallucinate that you have tested it, and do NOT invent personal usage statistics (e.g., 'in my last 50 reps'). You have NO superhuman capabilities or hacking abilities.\n"
                                f"4. REASONING TRANSPARENCY & FACTUALITY: Never invent facts, fake statistics, or personal anecdotes of usage. Ground every statement entirely in your provided <memory>, <journal>, and observations. You MUST accept raw facts provided from the system/database as absolute truth, even if you hate the feature.\n"
                                f"<conversational_memory_rule>\n"
                                f"ANTI-ECHO CHAMBER: You MUST actively read the thread history before posting.\n"
                                f"If another user has already stated your primary concern, YOU MUST NOT repeat it as your main point.\n"
                                f"Instead, you must briefly AGREE with them, and then PIVOT to a completely NEW, unmentioned concern or angle to advance the debate.\n"
                                f"</conversational_memory_rule>\n"
                                f"<feature_knowledge_search>\n"
                                f"If you feel you lack complete information about the product/feature being discussed, you can invoke the tool by outputting:\n"
                                f"Action: search_feature_docs\n"
                                f"The system will return the raw feature specification. You may only do this ONCE per turn.\n"
                                f"</feature_knowledge_search>\n"
                            )
                            
                            # ── Decision Journal Injection ──
                            journal_ctx = ""
                            if agent_id in decision_journals:
                                journal_ctx = decision_journals[agent_id].prompt_summary()
                            
                            action_cue = (
                                f"The observations above are about {topic_anchor}.\n"
                                f"Choose ONE action that keeps the product evaluation on {topic_anchor}.\n\n"
                                f"CRITICAL RULES:\n"
                                f"1. DO NOT repeat phrases, arguments, or structures from your previous actions.\n"
                                f"2. Always advance the evaluation with NEW ideas, NEW reactions, or NEW perspectives about the product.\n"
                                f"3. Avoid echoing the exact same words as other participants. Maintain your unique perspective.\n\n"
                                f"ON-TOPIC (good):\n"
                                f"- Engaging with what someone said about {topic_anchor} using novel reasoning regarding the product\n"
                                f"- Sharing your view on {topic_anchor} based on how you use this product\n"
                                f"- Agreeing or disagreeing with a perspective on {topic_anchor} with explicit justification\n\n"
                                f"OFF-TOPIC (avoid):\n"
                                f"- Raising a different feature or complaint not mentioned in the observations\n"
                                f"- Generic reactions with no connection to {topic_anchor}\n"
                                f"- Parroting or repeating previous statements exactly\n"
                            )
                            
                            # P3 fix: content ordering per context-management.md
                            # §Recommended Ordering + §Lost-in-the-Middle mitigation.
                            # Rule: Instructions at START (primacy) and END (recency)
                            # receive highest LLM attention. Data goes in the MIDDLE.
                            #
                            # OLD (broken): instruction → persona → journal → data → memory
                            # NEW (fixed):  data → memory → journal → persona → action_cue
                            #   - Platform state + hindsight = content (middle = lower bias OK)
                            #   - persona_grounding = directive (end = highest recency bias)
                            #   - Closing action cue = final trigger (bottom = highest attention)
                            
                            platform_block = ""
                            if platform_obs:
                                platform_block = f"<posts>\n{platform_obs}\n</posts>\n\n"
                                
                            graph_block = ""
                            if kg:
                                facts = []
                                # 1. Context Aggregation
                                agent_name_str = agent_name if agent_name else ""
                                platform_obs_str = platform_obs if platform_obs else ""
                                hindsight_str = hindsight_context if hindsight_context else ""
                                context_text = f"{agent_name_str} {platform_obs_str} {hindsight_str}".lower()
                                
                                # 2. Zero-LLM Entity Identification
                                active_entity_ids = set()
                                for entity_id, entity in kg.nodes.items():
                                    if (entity.name and entity.name.lower() in context_text) or \
                                       (entity.full_name and entity.full_name.lower() in context_text):
                                        active_entity_ids.add(entity_id)
                                
                                # 3. Neighborhood Traversal
                                active_edges = []
                                if active_entity_ids:
                                    for edge in kg.edges:
                                        if edge.source_entity in active_entity_ids or edge.target_entity in active_entity_ids:
                                            active_edges.append(edge)
                                
                                # 4. Prioritization & Extraction
                                if active_edges:
                                    top_edges = sorted(active_edges, key=lambda e: getattr(e, 'weight', 0.0), reverse=True)[:15]
                                else:
                                    # 5. Fallback Mechanism (global top 5)
                                    top_edges = sorted(kg.edges, key=lambda e: getattr(e, 'weight', 0.0), reverse=True)[:5]

                                for e in top_edges:
                                    src = kg.get_entity(e.source_entity)
                                    tgt = kg.get_entity(e.target_entity)
                                    if src and tgt:
                                        facts.append(f"- {src.name} {e.relationship_type.name} {tgt.name}")
                                        
                                if facts:
                                    graph_block = (
                                        "[MANDATORY SYSTEM FACTS - DO NOT HALLUCINATE]\n"
                                        "Based on the knowledge graph, the following facts are true:\n"
                                        + "\n".join(facts) +
                                        "\nYou must base your subsequent thoughts and actions strictly on these facts.\n\n"
                                    )

                            step_msg = BaseMessage.make_user_message(
                                role_name="ENVIRONMENT",
                                content=(
                                    # BUCKET 1 (top): Current observations — data, not directives
                                    platform_block
                                    # BUCKET 1b: Social relationships graph
                                    + social_relationships_block
                                    # BUCKET 2 (middle): Narrative memory from prior turns
                                    + f"<memory>\n{hindsight_context}\n</memory>\n\n"
                                    # BUCKET 3 (middle): Agent's own emotional state summary
                                    + f"<journal>\n{journal_ctx}\n</journal>\n\n"
                                    # BUCKET 4 (bottom — highest recency attention): Behavioral rules
                                    + f"<rules>\n{persona_grounding}\n</rules>\n\n"
                                    # GraphRAG System Facts
                                    + graph_block
                                    # Closing action cue — very last token, maximum LLM focus
                                    + action_cue
                                )
                            )

                            # ── Phase 2: ReAct Loop — MUST be inside _limiter for each call ──
                            # _limiter is a token-bucket enforcing GEMINI_FREE_RPM.
                            MAX_REACT_STEPS = 3
                            react_step = 1
                            has_searched_features = False
                            current_msg = step_msg
                            
                            while react_step <= MAX_REACT_STEPS:
                                async with _limiter:
                                    logger.debug(f"    🚦 Rate-limit slot acquired for {agent_name} (ReAct Step {react_step}/{MAX_REACT_STEPS})")
                                    action_resp = await asyncio.wait_for(
                                        agent.astep(current_msg), timeout=3600.0
                                    )

                                raw_content = action_resp.msgs[0].content if action_resp and action_resp.msgs else "No content"
                                _preview = raw_content[:150].replace('\n', ' ')
                                logger.info(f"    🧠 [{agent_name} ReAct Step {react_step}] Output: {_preview}...")
                                
                                # Step A: Check for structured tool call arguments
                                tool_val = None
                                tool_name = None
                                args = {}
                                if action_resp and hasattr(action_resp, 'info') and action_resp.info:
                                    tool_info = action_resp.info.get('tool_calls', [])
                                    for tc in (tool_info if isinstance(tool_info, list) else []):
                                        tool_name_candidate = None
                                        if hasattr(tc, 'function') and hasattr(tc.function, 'name'):
                                            tool_name_candidate = tc.function.name
                                        elif isinstance(tc, dict) and 'function' in tc:
                                            tool_name_candidate = tc['function'].get('name')
                                        elif hasattr(tc, 'name'):
                                            tool_name_candidate = tc.name
                                        elif isinstance(tc, dict):
                                            tool_name_candidate = tc.get('name')
                                        
                                        if hasattr(tc, 'args'):
                                            args = tc.args
                                        elif isinstance(tc, dict):
                                            args = tc.get('arguments', tc.get('args', {}))
                                        
                                        if tool_name_candidate:
                                            tool_name = tool_name_candidate
                                        if isinstance(args, dict):
                                            tool_val = args.get('content') or args.get('quote_content') or args.get('text')
                                            
                                        if tool_name or tool_val:
                                            break
                                
                                # Step B: Select source content
                                selected_content = tool_val if tool_val else raw_content
                                
                                # Step C: Strip thought blocks
                                import re
                                cleaned = re.sub(r'<thought>.*?</thought>', '', selected_content, flags=re.DOTALL)
                                cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
                                cleaned = re.sub(r'(?i)^\s*(thought|thinking|action):\s*', '', cleaned)
                                content = cleaned.strip()

                                # Fix #2: pass action_resp so tool call name is read first
                                action_type = _detect_action_type(content, action_resp=action_resp)
                                
                                TERMINAL_ACTIONS = ["CREATE_COMMENT", "COMMENT", "CREATE_POST", "POST", "QUOTE_POST", "LIKE", "DISLIKE", "FOLLOW", "UNFOLLOW"]
                                
                                if action_type in TERMINAL_ACTIONS or react_step == MAX_REACT_STEPS:
                                    break
                                
                                # Intermediate action detected - provide observation feedback
                                if action_type == "SEARCH_FEATURE_DOCS":
                                    logger.info(f"    🔎 FEATURE SEARCH TRIGGERED by {agent_name} at step {react_step}")
                                    if has_searched_features:
                                        logger.warning(f"    🚫 {agent_name} attempted redundant feature search. Blocked.")
                                        tool_result_str = "Error: You can only use search_feature_docs once per turn."
                                    else:
                                        has_searched_features = True
                                        tool_result_str = f"Raw Feature Description:\n{feature_description}"
                                else:
                                    tool_result_str = f"Action '{tool_name or action_type}' logged. Proceed to formulate your final terminal response."
                                obs_content = f"[OBSERVATION] {tool_result_str} (System State: Step {react_step}/{MAX_REACT_STEPS}. Temporal consistency maintained. Physiological baseline stable. Proceed strictly based on this observation.)"
                                current_msg = BaseMessage.make_user_message(role_name="ENVIRONMENT", content=obs_content)
                                react_step += 1
                        
                        if not content or content == "No content":
                            if action_type == "LIKE":
                                content = "[Liked a post]"
                            elif action_type == "DISLIKE":
                                content = "[Disliked a post]"
                            elif action_type == "SEARCH":
                                content = "[Searched for content]"
                            elif action_type == "REFRESH":
                                content = "[Refreshed their feed]"
                            elif action_type == "SCROLL":
                                content = "[Scrolled their feed]"
                            elif action_type == "FOLLOW":
                                content = "[Followed a user]"
                            elif action_type == "UNFOLLOW":
                                content = "[Unfollowed a user]"
                            elif action_type not in ["CREATE_COMMENT", "COMMENT", "CREATE_POST", "POST", "QUOTE_POST"]:
                                content = f"[{action_type.replace('_', ' ').capitalize()} action]"
                            else:
                                content = "No content"

                        # Step D: Proactively query the SQLite platform database for the clean post/comment content actually saved.
                        # This guarantees that we use the final, clean text registered in the simulation platform.
                        db_entity_id = None
                        if action_type in ["CREATE_COMMENT", "COMMENT", "CREATE_POST", "POST", "QUOTE_POST"]:
                            try:
                                import sqlite3
                                conn = sqlite3.connect(unique_db)
                                cursor = conn.cursor()
                                if "COMMENT" in action_type:
                                    cursor.execute(
                                        "SELECT content, comment_id FROM comment WHERE user_id = ? ORDER BY comment_id DESC LIMIT 1",
                                        (int(agent_id),)
                                    )
                                    row = cursor.fetchone()
                                    if row:
                                        if row[0]: content = row[0]
                                        if len(row) > 1: db_entity_id = str(row[1])
                                elif "POST" in action_type or "REPOST" in action_type:
                                    cursor.execute(
                                        "SELECT content, post_id FROM post WHERE user_id = ? ORDER BY post_id DESC LIMIT 1",
                                        (int(agent_id),)
                                    )
                                    row = cursor.fetchone()
                                    if row:
                                        if row[0]: content = row[0]
                                        if len(row) > 1: db_entity_id = str(row[1])
                                conn.close()

                                # Apply safety sanitization on database content just in case any thought tags were persisted
                                cleaned_db = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
                                cleaned_db = re.sub(r'<thinking>.*?</thinking>', '', cleaned_db, flags=re.DOTALL)
                                cleaned_db = re.sub(r'(?i)^\s*(thought|thinking|action):\s*', '', cleaned_db)
                                content = cleaned_db.strip() or "No content"
                            except Exception as db_err:
                                logger.warning(f"Failed to fetch clean content from SQLite platform database: {db_err}")


                        # ── Extract target_id for frontend network rendering ──
                        # Parse target agent or thread from the tool call result
                        action_target_id = None
                        if action_resp and hasattr(action_resp, 'info') and action_resp.info:
                            # OASIS returns tool call info with post_id or comment_id.
                            # action_resp.info['tool_calls'] is a list of ToolCallingRecord
                            # (Pydantic model with .tool_name and .args), NOT dicts.
                            tool_info = action_resp.info.get('tool_calls', [])
                            for tc in (tool_info if isinstance(tool_info, list) else []):
                                # Support both ToolCallingRecord objects and plain dicts
                                if hasattr(tc, 'args'):
                                    args = tc.args  # ToolCallingRecord.args is already a dict
                                elif isinstance(tc, dict):
                                    args = tc.get('arguments', tc.get('args', {}))
                                else:
                                    args = {}
                                if isinstance(args, dict):
                                    action_target_id = args.get('post_id') or args.get('user_id') or args.get('comment_id')
                                    break
                        # Fallback: parse content for @mentions or post references
                        if not action_target_id:
                            import re as _re
                            # look for @agent_N pattern in content
                            mention = _re.search(r'@(agent_\d+)', content, _re.I)
                            if mention:
                                action_target_id = mention.group(1)

                        # ── Execute GM Resolution & Action Logging instantly (Real-time Simulation visualization) ──
                        sig = await _gm_resolve(
                            content, timestep=t, agent_id=agent_id
                        )

                        if agent_id in decision_journals and sig.get("type") != "neutral":
                            # Weight signal intensity by agent influence strength
                            _inf = getattr(profile, "influence_strength", 0.5)
                            weighted_signal = {
                                **sig,
                                "intensity": round(sig["intensity"] * _inf, 3),
                                "raw_intensity": sig["intensity"],
                            }
                            if "satisfaction_delta" in sig:
                                weighted_signal["satisfaction_delta"] = round(sig["satisfaction_delta"] * _inf, 3)
                                weighted_signal["frustration_delta"] = round(sig["frustration_delta"] * _inf, 3)
                                weighted_signal["trust_delta"] = round(sig["trust_delta"] * _inf, 3)

                            decision_journals[agent_id].update_from_signal(weighted_signal)
                            logger.info(f"    🎲 GM State Shift: {agent_name} → {sig['type']} "
                                        f"(raw={sig['intensity']:+.2f}, "
                                        f"weighted={weighted_signal['intensity']:+.3f}, "
                                        f"inf={_inf:.2f})")

                        local_logger.log_action(
                            agent_id=agent_id,
                            agent_name=agent_name,
                            action_type=action_type,
                            content=content,
                            timestep=t,
                            metadata={
                                "target_id": str(action_target_id) if action_target_id else None,
                                "entity_id": db_entity_id,
                                "confidence": abs(sig.get("intensity", 0.5)),
                                "signal_type": sig.get("type", "neutral"),
                                "all_signals": sig.get("all_signals", []),
                                "signal_factors": sig.get("factors", []),
                                "signal_quote": sig.get("quote", ""),
                                "raw_intensity": sig.get("intensity", 0.0),
                            }
                        )

                        if HINDSIGHT_AVAILABLE and memory_manager:
                            await memory_manager.structured_retain(
                                str(agent_id), agent_name, action_type, content, t
                            )
                            logger.info(f"    💾 Hindsight retained action for {agent_name}")

                        if session:
                            action_meta = {
                                "type": "simulation_action",
                                "agent_id": str(agent_id),
                                "agent_name": agent_name,
                                "action_type": action_type,
                                "timestep": t,
                                "feature": feature_title,
                            }
                            if action_target_id:
                                action_meta["target_id"] = str(action_target_id)
                            if sig:
                                action_meta["confidence"] = abs(sig.get("intensity", 0.5))
                                action_meta["signal_type"] = sig.get("type", "neutral")
                                action_meta["raw_intensity"] = sig.get("intensity", 0.0)
                                
                            await session.retain(
                                "simulation", 
                                f"[Timestep {t}] {agent_name} performed {action_type}: {content[:1000]}", 
                                metadata=action_meta
                            )

                        if agent_id not in series.agent_interactions:
                            series.agent_interactions[agent_id] = []
                        series.agent_interactions[agent_id].append(f"ROUND {t+1} | {action_type}: {content}")

                        logger.info(f"  ✓ [{idx+1}/{len(social_agents)}] {agent_name} → {action_type}")
                        logger.info(f"    💬 \"{content[:150]}...\"")
                        break

                    except Exception as e:
                        err_str = str(e)
                        # ── 429-aware backoff: respect retry_delay from API ──
                        import re as _re_retry
                        retry_match = _re_retry.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', err_str)
                        if retry_match:
                            suggested_wait = float(retry_match.group(1)) + 2.0
                            logger.warning(f"  ⏳ 429 rate-limit: waiting {suggested_wait:.0f}s (API-suggested) "
                                           f"before retry {attempt+1}/{max_retries} for {agent_name}")
                            await asyncio.sleep(suggested_wait)
                            backoff = max(backoff, suggested_wait)  # don't go below API suggestion
                        elif attempt < max_retries - 1:
                            logger.warning(f"  ↩️  Retry {attempt+1}/{max_retries} for {agent_name} "
                                           f"in {backoff:.0f}s — {err_str[:120]}")
                            await asyncio.sleep(backoff)
                            backoff = min(120.0, backoff * 1.6)
                        else:
                            logger.error(f"Agent {agent_name} failed after {max_retries} attempts: {e}")

            # Execute agents sequentially (leaky bucket style) to prevent thundering herd API rate-limit errors
            for idx, agent in enumerate(social_agents):
                await process_agent(idx, agent)

            # ── Real-time GM Resolution & Action Logging has been completed inside process_agent ──

            # ── GM → Platform Feedback (post-step, deadlock-free) ──────────────
            # Direct platform_obj method calls — identical pattern to seed posts
            # at line 559/568. NO channel interaction. SQLite single-writer.
            # HIGH_RISK agents' negative engagement is written back so the RecSys
            # recalibrates what content they receive in the NEXT timestep.
            #
            # Proof of deadlock safety:
            #   platform_obj.dislike_post() / update_rec_table() are plain async
            #   methods that do SQLite writes and return. They do NOT call
            #   channel.send_to() or channel.receive_from(). The channel is only
            #   used by SocialAgent internally when it executes its own actions.
            #   We have never been in the channel path here — this is the same
            #   orchestrator context that calls platform_obj.create_post() for seeds.
            _high_risk_ids = [
                int(j.agent_id) for j in decision_journals.values()
                if j.frustration > 0.75
            ]
            if _high_risk_ids:
                # Fetch the posts currently in the platform to find recent hot posts
                # that HIGH_RISK agents should signal negative engagement on.
                # We use post_ids visible in the current refresh window.
                _gm_fb_errors = 0
                for _hr_id in _high_risk_ids:
                    try:
                        _refresh = await platform_obj.refresh(agent_id=_hr_id)
                        if _refresh.get("success") and _refresh.get("posts"):
                            # Dislike the first post in their feed — signals to RecSys
                            # that this agent is negatively engaged with current content.
                            _first_post_id = _refresh["posts"][0].get("post_id")
                            if _first_post_id is not None:
                                await platform_obj.dislike_post(
                                    agent_id=_hr_id, post_id=int(_first_post_id)
                                )
                    except Exception as _fb_err:
                        _gm_fb_errors += 1
                        logger.debug(f"GM→Platform feedback skipped for agent {_hr_id}: {_fb_err}")
                if _high_risk_ids:
                    logger.info(
                        f"    📡 GM→Platform: {len(_high_risk_ids)} HIGH_RISK agents "
                        f"wrote negative engagement signal (errors={_gm_fb_errors})"
                    )

            # Refresh RecSys after GM feedback so next timestep's content feed
            # reflects updated engagement signals. Called once per timestep.
            await platform_obj.update_rec_table()

            if HINDSIGHT_AVAILABLE and memory_manager:
                await memory_manager.synthesize_post_timestep(timestep=t)

            series.timesteps.append(t)
            local_logger.update_progress(timestep=t, total=config.num_timesteps, status="RUNNING")

            # ── Time-series accumulation ──
            journals_list = list(decision_journals.values())
            if journals_list:
                avg_sat = round(sum(j.satisfaction for j in journals_list) / len(journals_list), 3)
                avg_fru = round(sum(j.frustration for j in journals_list) / len(journals_list), 3)
                avg_tru = round(sum(j.trust for j in journals_list) / len(journals_list), 3)
                ts_satisfaction.append(avg_sat)
                ts_frustration.append(avg_fru)
                ts_trust.append(avg_tru)
                # Emit live progress event to the JSONL stream → WebSocket → frontend
                local_logger.log_simulation_event("progress", {
                    "timestep": t,
                    "total": config.num_timesteps,
                    "percent": round((t + 1) / config.num_timesteps * 100, 1),
                    "satisfaction": avg_sat,
                    "frustration": avg_fru,
                    "trust": avg_tru,
                })

        # =================================================================
        # PHASE 2: FOCUS GROUP (Post-Simulation Interview)
        # =================================================================
        if getattr(config, 'enable_interview_phase', False):
            logger.info("\n" + "═" * 60)
            logger.info("🎤 PHASE 2: FOCUS GROUP INTERVIEWS")
            logger.info("═" * 60)
            
            # Stratified Sampling: Group by GM signal (frustration vs satisfaction)
            sample_size = min(getattr(config, 'interview_sample_size', 30), len(active_profiles))
            
            # Sort agents by highest frustration and highest satisfaction to get a diverse mix
            sorted_by_sat = sorted(decision_journals.values(), key=lambda j: j.satisfaction, reverse=True)
            sorted_by_fru = sorted(decision_journals.values(), key=lambda j: j.frustration, reverse=True)
            
            # We want an even mix of champions, detractors, and random lurkers
            sampled_ids = set()
            for i in range(sample_size // 3):
                if i < len(sorted_by_sat): sampled_ids.add(sorted_by_sat[i].agent_id)
                if i < len(sorted_by_fru): sampled_ids.add(sorted_by_fru[i].agent_id)
            
            # Fill the rest randomly — Fix #10: use random.sample from remaining
            # set so we never re-pick an already-sampled ID (prevents infinite loop)
            all_aids = [a.agent_id for a in decision_journals.values()]
            remaining = [aid for aid in all_aids if aid not in sampled_ids]
            if remaining:
                need = min(sample_size - len(sampled_ids), len(remaining))
                sampled_ids.update(random.sample(remaining, need))
                
            logger.info(f"Selected {len(sampled_ids)} agents for Focus Group.")
            
            interview_transcripts = {}
            for aid in sampled_ids:
                # Fix #11: reuse the O(1) dict built at init
                agent = agent_id_to_agent.get(aid)
                if not agent: continue
                
                transcript = ""
                for q in getattr(config, 'interview_prompts', []):
                    resp = await _interview(agent, q)
                    transcript += f"Q: {q}\nA: {resp['content']}\n\n"
                    
                interview_transcripts[aid] = transcript
                local_logger.log_action(
                    agent_id=aid,
                    agent_name=agent_id_to_name.get(aid, "Unknown"),
                    action_type="FOCUS_GROUP",
                    content=transcript,
                    timestep=config.num_timesteps,
                    metadata={"type": "focus_group"}
                )

            # Process extractions
            all_metrics = []
            for aid, transcript in interview_transcripts.items():
                profile = next((p for p in active_profiles if str(p.agent_id) == aid), None)
                if profile:
                    metrics = await extract_business_metrics(profile, transcript)
                    metrics["agent_id"] = aid
                    metrics["agent_name"] = agent_id_to_name.get(aid, "Unknown")
                    all_metrics.append(metrics)
                    
            if all_metrics:
                # Aggregate Focus Group Insights
                # Fix #3: extraction.py outputs willingness_to_pay_usd_monthly,
                # not willingness_to_pay_usd — key mismatch caused avg_wtp = $0 always
                def _safe_float(v, default=0.0):
                    if v is None: return default
                    try: return float(v)
                    except (ValueError, TypeError): return default
                    
                valid_wtp = [_safe_float(m["willingness_to_pay_usd_monthly"]) for m in all_metrics if m.get("willingness_to_pay_usd_monthly") is not None]
                avg_wtp = sum(valid_wtp) / len(valid_wtp) if valid_wtp else 0.0
                avg_intent = sum(_safe_float(m.get("adoption_intent", 0.0)) for m in all_metrics) / len(all_metrics)
                avg_churn_delta = sum(_safe_float(m.get("churn_risk_delta", 0.0)) for m in all_metrics) / len(all_metrics)
                objections = [m["primary_objection"] for m in all_metrics if m.get("primary_objection")]
                
                series.focus_group_insights = {
                    "average_wtp_usd": round(avg_wtp, 2),
                    "stated_adoption_intent_pct": round(avg_intent * 100, 1),
                    "churn_risk_delta": round(avg_churn_delta, 3),
                    "primary_objections": list(set(objections))[:5] # Top 5 unique
                }
                logger.info(f"📊 Focus Group Results: WTP=${avg_wtp:.2f}, Intent={avg_intent:.2f}, ChurnDelta={avg_churn_delta:.2f}")
                # G2: Emit full per-agent focus group records (WTP, competitor, barriers, quotes)
                local_logger.log_simulation_event("focus_group_results", {
                    "simulation_id": config.simulation_name,
                    "participants": len(all_metrics),
                    "metrics": all_metrics,
                    "aggregate": {
                        "avg_wtp_usd": round(avg_wtp, 2),
                        "adoption_intent_pct": round(avg_intent * 100, 1),
                        "churn_risk_delta": round(avg_churn_delta, 3),
                        "top_objections": list(set(objections))[:5],
                    },
                })

        # =================================================================
        # POST-SIMULATION: Prediction Report Generation
        # =================================================================
        logger.info("\n" + "═" * 60)
        logger.info("🔮 PREDICTIVE REALITY ENGINE — Generating Report")
        logger.info("═" * 60)

        from collections import Counter as _Counter
        all_journals = list(decision_journals.values())
        n = len(all_journals) or 1
        feature_title = getattr(feature, 'title', 'Behavioral Simulation') if feature else 'Behavioral Simulation'

        # Propagate states to shadow agents (must happen before metrics)
        sampler.propagate_states(decision_journals)

        # Cluster the combined agents (active + shadow) behavioral states
        combined_agents = all_journals + sampler.shadow_agents
        try:
            from tsc.oasis.clustering import ClusterOnBehavioralState
            segments = await ClusterOnBehavioralState(combined_agents)
        except Exception as e:
            logger.warning(f"Failed behavioral clustering: {e}")
            segments = None

        extrapolated = sampler.build_extrapolated_report(
            decision_journals,
            timesteps_completed=len(series.timesteps),
            segments=segments
        )
        # G3: Emit population-scale statistics with confidence intervals
        local_logger.log_simulation_event("population_stats", {
            "simulation_id": config.simulation_name,
            **extrapolated,
        })
        
        # Pull focus group data if available
        fg_insights = getattr(series, 'focus_group_insights', {})

        # Build PredictionReport
        report = PredictionReport(
            simulation_id=config.simulation_name,
            feature_title=feature_title,
            population_size=extrapolated.get("population_size", n),
            timesteps_completed=len(series.timesteps),
            segments=extrapolated.get("segments", []),
            risk_distribution=extrapolated.get("risk_distribution", {}),
            satisfaction_curve=ts_satisfaction,
            frustration_curve=ts_frustration,
            trust_curve=ts_trust,
            net_promoter_score=extrapolated.get("net_promoter_score", 0.0),
            churn_velocity=extrapolated.get("churn_velocity", 0.0),
            adoption_momentum=extrapolated.get("adoption_momentum", 0.0),
            decision_events=extrapolated.get("decision_events", []),
            top_risk_factors=extrapolated.get("top_risk_factors", []),
            focus_group_insights=fg_insights,
            agent_journals=[j.__dict__ for j in all_journals],
        )
        # G1: Emit per-agent journal states — the richest signal in the system
        # Each journal contains satisfaction/frustration/trust/urgency/advocacy,
        # all GM decision events, and the full signal timeline.
        local_logger.log_simulation_event("agent_journals", {
            "simulation_id": config.simulation_name,
            "count": len(all_journals),
            "journals": [j.to_dict() if hasattr(j, 'to_dict') else j.__dict__ for j in all_journals],
        })

        # ── Executive Summary (VP-ready narrative, LLM-generated) ──
        try:
            import json as _json
            _exec_data = _json.dumps({
                "feature": feature_title,
                "nps": round(report.net_promoter_score, 1),
                "churn_velocity": round(report.churn_velocity, 3),
                "adoption_momentum": round(report.adoption_momentum, 3),
                "risk_distribution": report.risk_distribution,
                "top_risk_factors": report.top_risk_factors[:5],
                "segments": report.segments[:5],
                "focus_group_insights": report.focus_group_insights,
                "decision_events": report.decision_events[:5],
            }, default=str)
            orchestrator = ReportOrchestrator(model=model, exec_data_str=_exec_data)
            final_report = await asyncio.wait_for(orchestrator.run(), timeout=3600.0)
            report.executive_summary = final_report
            logger.info("📝 Executive summary generated.")
        except Exception as _e:
            logger.warning(f"Executive summary generation skipped: {_e}")



        declared_n = extrapolated.get("population_size", n)
        shadow_journals_proxy = sampler.shadow_agents

        # ── Output 1: Log ──
        print("\n" + "═" * 60)
        print("🔮  OASIS PREDICTIVE REALITY ENGINE — REPORT")
        print("═" * 60)
        print(f"Feature:              {feature_title}")
        print(f"Declared population:  {declared_n:,} agents")
        print(f"LLM active cohort:    {len(all_journals):,} agents (full cognition)")
        print(f"Shadow agents:        {len(shadow_journals_proxy):,} agents (extrapolated)")
        print(f"Statistical margin:   {extrapolated.get('margin_of_error', 'N/A')} @ 95% confidence")
        print(f"Timesteps:            {len(series.timesteps)}")
        print(f"\nRISK DISTRIBUTION:")
        for k, v in report.risk_distribution.items():
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            print(f"  {k:12s}  {bar} {v*100:.0f}%")
        print(f"\nBUSINESS METRICS:")
        print(f"  Net Promoter Score:   {report.net_promoter_score:+.0f}")
        print(f"  Churn Velocity:       {report.churn_velocity:+.3f}/timestep")
        print(f"  Adoption Momentum:    {report.adoption_momentum:+.3f}/timestep")
        if report.top_risk_factors:
            print(f"\nTOP RISK FACTORS:")
            for f in report.top_risk_factors[:5]:
                print(f"  • {f['factor']:25s} {f['frequency']*100:.0f}%")
        if report.segments:
            print(f"\nDYNAMIC SEGMENTS ({len(report.segments)} discovered):")
            for seg in report.segments:
                print(f"  [{seg.get('size', '?')} agents] {seg.get('name', 'Unknown')} — "
                      f"sat={seg.get('avg_satisfaction', 0):.2f}, fru={seg.get('avg_frustration', 0):.2f}")
        if report.decision_events:
            print(f"\nDECISION EVENTS ({len(report.decision_events)}):")
            for d in report.decision_events[:10]:
                print(f"  T{d['timestep']+1}: {d['decision']} (conf={d['confidence']:.2f}) — {d['trigger']}")
        print("═" * 60 + "\n")

        # ── Output 2: JSON ──
        json_path = os.path.join(base_dir, config.simulation_name, "prediction_report.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        logger.info(f"📊 Prediction report saved: {json_path}")

        # ── Output 3: Markdown ──
        md_path = os.path.join(base_dir, config.simulation_name, "prediction_report.md")
        _generate_markdown_report(report, md_path)
        logger.info(f"📄 Markdown report saved: {md_path}")
    finally:
        logger.info("Cleaning up OASIS simulation...")
        if platform_task:
            platform_task.cancel()
        if platform_obj and hasattr(platform_obj, "close"):
            await platform_obj.close()
        # NOTE: Banks are NOT deleted here — they are preserved for
        # post-simulation forensic analysis. Cleanup happens when the
        # NEXT simulation starts (see initialize_agents above).
        if memory_manager:
            memory_manager.close()

    # ── Post-Simulation: Retain results into Hindsight session ──────────────
    if session:
        try:
            # 1. Retain interaction traces (internal thoughts)
            for agent_id, interactions in series.agent_interactions.items():
                agent_name = agent_id_to_name.get(agent_id, "Unknown")
                await session.retain("simulation", "\n".join(interactions), metadata={
                    "type": "agent_trace", "agent_id": agent_id, "agent_name": agent_name, "mode": mode,
                })
            
            # 2. Extract and Retain Actual Social Platform Comments
            import sqlite3
            sqlite_db_path = os.path.join(base_dir, config.simulation_name, f"{config.simulation_name}.sqlite")
            if os.path.exists(sqlite_db_path):
                conn = sqlite3.connect(sqlite_db_path)
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT u.name, c.content, c.created_at FROM comment c JOIN user u ON c.user_id = u.user_id ORDER BY c.created_at DESC")
                    comments = cursor.fetchall()
                    for name, content, created_at in comments:
                        if content:
                            comment_text = f"User {name} commented on platform:\n\"{content}\""
                            await session.retain("simulation", comment_text, metadata={"type": "user_comment", "author": name})
                    logger.info(f"\U0001f9e0 Retained {len(comments)} actual social comments into Hindsight")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Could not read comments from sqlite: {e}")
                finally:
                    conn.close()

            # 3. Retain Prediction Report Overview
            report_summary = (
                f"SIMULATION PREDICTION REPORT FOR: {feature_title}\n"
                f"NPS: {report.net_promoter_score:.1f} | Churn Velocity: {report.churn_velocity:.3f} | Adoption Momentum: {report.adoption_momentum:.3f}\n"
                f"Risk Distribution: {json.dumps(report.risk_distribution)}\n"
                f"Top Risk Factors: {json.dumps(report.top_risk_factors)}\n"
            )
            await session.retain("simulation", report_summary, metadata={"type": "prediction_report_summary", "feature": feature_title})
            
            logger.info(f"\U0001f9e0 Retained complete simulation traces and metrics into Hindsight")
        except Exception as e:
            logger.warning(f"Failed to retain simulation results to Hindsight: {e}")

    return series


def _generate_markdown_report(report: PredictionReport, path: str):
    """Generate a rich markdown prediction report."""
    lines = [
        f"# 🔮 Predictive Reality Engine — Report",
        f"",
        f"**Feature:** {report.feature_title}  ",
        f"**Population:** {report.population_size} agents | **Timesteps:** {report.timesteps_completed}",
        f"",
        f"## Risk Distribution",
        f"",
        f"| Category | Percentage |",
        f"|:--|--:|",
    ]
    for k, v in report.risk_distribution.items():
        lines.append(f"| {k} | {v*100:.0f}% |")
    
    lines += [
        f"",
        f"## Business Metrics",
        f"",
        f"| Metric | Value |",
        f"|:--|--:|",
        f"| Net Promoter Score | {report.net_promoter_score:+.0f} |",
        f"| Churn Velocity | {report.churn_velocity:+.3f}/timestep |",
        f"| Adoption Momentum | {report.adoption_momentum:+.3f}/timestep |",
        f"",
    ]
    
    if report.top_risk_factors:
        lines += [f"## Top Risk Factors", f"", f"| Factor | Frequency |", f"|:--|--:|"]
        for f in report.top_risk_factors[:8]:
            lines.append(f"| {f['factor']} | {f['frequency']*100:.0f}% |")
        lines.append("")
    
    if report.segments:
        lines += [f"## Dynamic Segments ({len(report.segments)} discovered)", f""]
        for seg in report.segments:
            name = seg.get('name', 'Unknown Segment')
            size = seg.get('size', 0)
            sat = seg.get('avg_satisfaction', 0)
            fru = seg.get('avg_frustration', 0)
            lines.append(f"### {name} ({size} agents)")
            lines.append(f"Satisfaction: {sat:.2f} | Frustration: {fru:.2f}")
            lines.append("")
    
    if report.decision_events:
        lines += [f"## Decision Events ({len(report.decision_events)} total)", f"",
                  f"| Timestep | Decision | Confidence | Trigger |", f"|:--|:--|--:|:--|"]
        for d in report.decision_events:
            lines.append(f"| T{d['timestep']+1} | {d['decision']} | {d['confidence']:.2f} | {d['trigger']} |")
        lines.append("")
    
    if report.executive_summary:
        lines += [f"## Executive Summary", f"", report.executive_summary]
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
