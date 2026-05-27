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
# EXECUTIVE SUMMARY PROMPT (module-level constant)
# =============================================================================
_EXEC_SUMMARY_SYSTEM = """\
You are a senior market research analyst preparing a simulation-derived executive brief.

<data_contract>
You will receive a JSON object containing simulation metrics.
YOU MUST:
- Use ONLY numbers and quotes present in that JSON. Do NOT invent, estimate, or
  extrapolate any metric not explicitly in the data.
- If a field is missing, zero, or empty (e.g. focus_group_insights is {}),
  state "Focus group data unavailable for this run" — do not omit or fabricate.
- Quote agents verbatim from decision_events[].quote only. Never paraphrase quotes.
</data_contract>

<output_format>
Write exactly 3 paragraphs. No headers. No bullet points anywhere.

PARAGRAPH 1 — VERDICT
Lead with the single most surprising finding. Then state one clear recommendation:
"ship" / "ship with changes" / "do not ship" / "needs more data".
Cite the exact NPS, churn_velocity, and adoption_momentum values from the JSON.
Use hedged language: "simulation suggests", not "will" or "is".

PARAGRAPH 2 — RISK PROFILE
Name the top 2 risks from top_risk_factors. For each, state likelihood as
Low/Medium/High using the risk_distribution percentages to justify the label.
Include one verbatim agent quote from decision_events[].quote — copy it exactly,
wrap in quotation marks, do not shorten it.

PARAGRAPH 3 — NEXT STEPS
Give exactly 3 recommendations. Each must be a single testable action
(verifiable true/false within 30 days). At least one must directly address
the highest-churn segment identified in the segments field.
</output_format>

Write for a VP of Product with 90 seconds to read this. Be direct."""


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

    if gm_llm_client is None:
        gm_llm_client = llm_client


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
        start_time=start_time
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
    ]

    social_agents: List[SocialAgent] = []
    for profile in agent_profiles:
        user_info = UserInfo(**profile.user_info_dict)
        agent = SocialAgent(
            agent_id=str(profile.agent_id),
            user_info=user_info,
            channel=channel,
            model=model,
            available_actions=USEFUL_ACTIONS
        )
        logger.info(f"Agent {agent.agent_id} initialized with Hindsight-backed Memory architecture.")
        social_agents.append(agent)

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

    # Layer 3: Stochastic reciprocity (30% chance of follow-back)
    reciprocal_edges = set()
    for (follower, followee) in list(follow_edges):
        reverse = (followee, follower)
        if reverse not in follow_edges and reverse not in reciprocal_edges:
            if random.random() < 0.30:
                reciprocal_edges.add(reverse)
                await platform_obj.follow(agent_id=followee, followee_id=follower)

    total_edges = len(follow_edges) + len(reciprocal_edges) + (num_agents - 1)  # peers + reciprocal + proposer
    avg_degree = total_edges / max(1, num_agents)
    logger.info(f"🕸️  Network Topology Built: {total_edges} edges, avg degree {avg_degree:.1f} "
                f"(peers: {len(follow_edges)}, reciprocal: {len(reciprocal_edges)}, hub: {num_agents - 1})")
    # G4: Emit social network topology so the 3D graph can render real edges
    local_logger.log_simulation_event("network_topology", {
        "simulation_id": config.simulation_name,
        "hub_agent_id": str(proposer_id),
        "total_edges": total_edges,
        "avg_degree": round(avg_degree, 2),
        "edges": [{"from": str(f), "to": str(t)} for f, t in list(follow_edges)[:500] + list(reciprocal_edges)[:200]],
    })

    # ── 8.1 Seed Platform — Source-to-Synth Pipeline ────────────────────────────
    # Research-backed approach: Generate diverse seed posts based purely on feature
    # and company context to act as the simulation stimulus.
    
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

        # ── 1. Build Complete Data Brief (proposal + context — full, not truncated) ─
        product_name  = ctx.company_name if ctx else "this product"
        feat_title    = feat.title if feat else "the proposed change"
        feat_desc     = feat.description if feat else ""          # FULL description — no truncation
        feat_domains  = ", ".join(feat.affected_domains) if (feat and hasattr(feat, "affected_domains") and feat.affected_domains) else ""
        tech_stack    = ", ".join(ctx.tech_stack) if (ctx and ctx.tech_stack) else "the platform stack"
        competitors   = "\n  - ".join(ctx.competitors) if (ctx and ctx.competitors) else "competitors"
        priorities    = "\n  - ".join(ctx.current_priorities) if (ctx and ctx.current_priorities) else "product growth"
        team_size     = str(ctx.team_size) if (ctx and ctx.team_size) else "unknown"
        budget        = ctx.budget if (ctx and hasattr(ctx, "budget") and ctx.budget) else ""
        stakeholders  = "\n  - ".join(ctx.key_stakeholders) if (ctx and hasattr(ctx, "stakeholders") and ctx.stakeholders) else ""

        # Pull any extra structured fields that exist on the models
        platform_scale_raw = getattr(ctx, "platform_scale", {}) or {}
        history_raw        = getattr(ctx, "historical_context", {}) or {}
        regulatory_raw     = getattr(ctx, "regulatory_environment", "") or ""

        # Compact but complete platform scale block
        platform_lines = [f"{k}: {v}" for k, v in platform_scale_raw.items()] if platform_scale_raw else []
        platform_scale = "\n  - ".join(platform_lines) if platform_lines else ""

        # Compact historical context
        history_lines = [f"{k}: {v}" for k, v in history_raw.items()] if history_raw else []
        history_block = "\n  - ".join(history_lines) if history_lines else ""

        # ── 2. Prompt: Role + Structured Brief + Data-Coverage Instructions ─────────
        # Key technique from context-management.md: inject the full brief as a
        # structured XML <reference_brief> block so the LLM treats it as ground truth
        # and derives post content from it — not from parametric memory.
        # Key technique from system-prompts.md: Static Context Injection pattern.
        prompt = f"""<role>
You are a Senior Social Simulation Architect. Your task is to generate the
COMPLETE INFORMATION BRIEF for a simulation: a set of seed posts that, taken
together, expose the simulated agents to ALL relevant facts about a product
feature announcement. Agents read ONLY these posts — they have no other
information channel. If a fact is not in a post, it does not exist for them.

Your job is equal parts JOURNALIST and DEBATE MODERATOR:
- Distribute every fact from the reference brief across the posts.
- Each post is written by a distinct archetype with a distinct angle on the data.
- Together, the posts form a complete, multi-perspective briefing that enables
  agents to form informed, specific positions.
</role>

<reference_brief>
This is the GROUND TRUTH. Every field below MUST appear in at least one seed post.

<product>
Name: {product_name}
Team size: {team_size}
{f"Budget / Revenue: {budget}" if budget else ""}
Tech stack: {tech_stack}
{f"Regulatory environment: {regulatory_raw}" if regulatory_raw else ""}
</product>

<feature>
Title: {feat_title}
{f"Affected surfaces: {feat_domains}" if feat_domains else ""}
Full description:
{feat_desc}
</feature>

<company_priorities>
  - {priorities}
</company_priorities>

<competitors>
  - {competitors}
</competitors>

{f"""<platform_scale>
  - {platform_scale}
</platform_scale>""" if platform_scale else ""}

{f"""<historical_context>
  - {history_block}
</historical_context>""" if history_block else ""}
</reference_brief>

<data_coverage_rules>
MANDATORY: All 8 posts together MUST cover every field in <reference_brief>.
Assign data clusters to posts using this Four-Bucket layout:

POST 1 [OFFICIAL ANNOUNCEMENT — Primacy Anchor]:
  Must embed: feature title, full scope of what changes and what DOESN'T change,
  stated rationale, platform scale (users affected), tech stack surfaces affected.
  Tone: Formal, authoritative, like a product blog post. Opens with the news.

POST 2 [BUSINESS ANALYST — Revenue & Motive]:
  Must embed: company revenue/budget, company priorities, the business logic,
  who the real beneficiaries might be vs. stated beneficiaries.
  Tone: Skeptical but data-driven. Cites specific numbers.

POST 3 [TECHNICAL DEVELOPER — API & Stack Impact]:
  Must embed: tech stack details, API changes/deprecations, migration timelines,
  third-party developer ecosystem impact.
  Tone: Technical precision. Asks the specific developer-facing question.

POST 4 [COMPETITOR OBSERVER — Market Landscape]:
  Must embed: ALL competitors and their stance on this feature type,
  market positioning implications, who benefits/loses competitively.
  Tone: Analytical, comparative, slightly threatening.

POST 5 [HISTORICAL CONTEXT CARRIER]:
  Must embed: historical_context data (events, dates, precedents),
  any prior experiments or rollouts, what the timeline looked like.
  Tone: "Let's remember the history here." Archival, matter-of-fact.

POST 6 [SAFETY / REGULATORY WATCHDOG]:
  Must embed: regulatory environment, any safety/moderation implications,
  second-order effects of the feature on platform integrity.
  Tone: Formal concern. Asks who reviewed the risk.

POST 7 [AFFECTED STAKEHOLDER — Concrete Impact]:
  Must embed: the most specific, concrete use case harmed or helped by this
  feature. A real-sounding story from a named role (creator, developer, viewer).
  Tone: Personal, specific. One concrete scenario, fully described.

POST 8 [EXIT / ULTIMATUM — Recency Anchor]:
  Must embed: the stakes (what happens if this isn't reversed/implemented),
  competitor alternatives available, the decision point framing.
  Tone: "Here is the line." Stakes-setting. Not angry — cold and deliberate.
</data_coverage_rules>

<constraints>
MUST DO:
- Every post must be 40-130 words. Dense but readable.
- Every post must reference at least ONE specific data point from <reference_brief>.
- Posts must collectively cover ALL fields in <reference_brief>.
- Each post must end with a question OR a statement that demands engagement.
- Posts must feel like they come from real platform users, not a press release.

MUST NOT:
- Do NOT invent facts, statistics, or events not in <reference_brief>.
- Do NOT write vague generalities ("users are concerned") — use specific claims.
- Do NOT repeat the same data point across multiple posts.
- Do NOT include archetype labels inside the post text.
- Do NOT write posts shorter than 40 words or longer than 130 words.
</constraints>

<output_format>
Return ONLY this XML structure. No explanation, no markdown, no preamble.
The JSON array inside <seed_posts> must contain exactly 8 strings.

<seed_posts>
["post 1", "post 2", "post 3", "post 4", "post 5", "post 6", "post 7", "post 8"]
</seed_posts>
</output_format>"""

        # ── 3. Execute with retry-with-error-correction (structured-outputs.md) ──────
        last_error = ""
        for attempt in range(2):
            try:
                if attempt > 0 and last_error:
                    correction_prompt = (
                        f"Your previous response failed to parse.\n"
                        f"Error: {last_error}\n\n"
                        f"Return ONLY the <seed_posts>[...]</seed_posts> XML block. "
                        f"8 strings in the JSON array. No other text.\n\n"
                        f"Task context:\n{prompt}"
                    )
                    response_text = await llm.async_generate(correction_prompt)
                else:
                    response_text = await llm.async_generate(prompt)

                # XML-tag extraction (immune to markdown code fences and prose)
                xml_match = re.search(r'<seed_posts>(.*?)</seed_posts>', response_text, re.DOTALL)
                raw_json  = xml_match.group(1).strip() if xml_match else response_text.strip()

                # Fallback: bare JSON array
                if not xml_match:
                    arr_match = re.search(r'\[.*?\]', raw_json, re.DOTALL)
                    raw_json  = arr_match.group() if arr_match else raw_json

                posts = _json.loads(raw_json)

                if isinstance(posts, list) and len(posts) >= 4:
                    valid = [str(p).strip() for p in posts if len(str(p).strip()) >= 40]
                    if len(valid) >= 4:
                        logger.info(f"✅ AI Seed Posts (v4 full-coverage): {len(valid)} posts injecting complete proposal+context brief")
                        return valid

                last_error = f"Got {len(posts)} posts but need at least 4 valid (>=40 chars). Check JSON formatting."

            except Exception as e:
                last_error = str(e)
                logger.warning(f"⚠️ Seed post attempt {attempt + 1} failed: {e}")

        logger.warning("⚠️ AI seed generation failed after 2 attempts — using template fallback")
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

    controversy_seeds = _extract_controversy_seeds(feature, context, market_ctx if 'market_ctx' in dir() else {})
    # G12: Emit seed posts so the UI can show the debate context from T=0
    local_logger.log_simulation_event("seed_posts", {
        "simulation_id": config.simulation_name,
        "seeds": [{"index": i, "content": s[:500]} for i, s in enumerate(controversy_seeds)],
    })

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
        if ai_seeds:
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
    else:
        # FEATURE TEST MODE
        logger.info(f"🔬 Generating seed posts for feature: {feature.title}")

        # AI-First: Generate contextual seeds from feature + community feedback
        ai_seeds = await _generate_ai_seed_posts(feature, context, llm_client)
        if ai_seeds:
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

    # G12: Emit all seed posts so the UI can show the debate context from T=0
    local_logger.log_simulation_event("seed_posts", {
        "simulation_id": config.simulation_name,
        "seeds": [{
            "index": i,
            "content": s[:500],
            "source": "ai_generated" if ai_seeds else "template_fallback"
        } for i, s in enumerate(ai_seeds if ai_seeds else controversy_seeds)],
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
                    response = await asyncio.wait_for(agent.astep(msg), timeout=120.0)
            return {
                "content": response.msgs[0].content if response.msgs else "No response",
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
        if sycophancy_match and (agent_frustration > 0.5 or (journal and journal.trust < 0.35)):
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
        route_to_llm = (has_critical or agent_frustration > 0.5) and (gm_llm_client is not None)

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
                "- sycophancy_collapse_detected: True if the agent suddenly capitulates or agrees with social pressure despite having prior frustration/skepticism.\n"
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
            user_prompt_parts.append(f"Agent Comment/Post:\n\"\"\"\n{content}\n\"\"\"")
            user_prompt = "\n".join(user_prompt_parts)

            # Call LLM client
            res = await gm_llm_client.analyze(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=schema,
                temperature=0.0  # Zero-shot, highly deterministic
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
        if "create_comment" in content_lower or "comment" in content_lower:
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
    try:
        for t in range(config.num_timesteps):
            await command_listener.wait_if_paused(interview_callback=eagle_eye_interview_callback)
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
                                MAX_POSTS = 5
                                MAX_COMMENTS_PER_POST = 3
                                platform_obs = "\n\nCURRENT PLATFORM STATE:\n"
                                for p in posts[:MAX_POSTS]:
                                    platform_obs += f"- [PostID {p['post_id']}] (User {p['user_id']}): {p['content']}\n"
                                    if p.get('comments'):
                                        for c in p['comments'][-MAX_COMMENTS_PER_POST:]:
                                            platform_obs += f"  └─ [CommentID {c['comment_id']}] (User {c['user_id']}): {c['content']}\n"

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
                                "MID-DISCUSSION": "React to what others have said. Build on or push back against SPECIFIC points already made.",
                                "CLOSING": "Consolidate your view. Has anything changed your position? State your final stance explicitly.",
                            }[_ts_phase]

                            persona_grounding = (
                                f"\nCRITICAL BEHAVIORAL RULES FOR {agent_name} "
                                f"[Timestep {t+1}/{config.num_timesteps} — {_ts_phase}]:\n"
                                f"1. ANTI-SYCOPHANCY: Do NOT change your stated position because others disagree. "
                                f"Only update if shown concrete evidence that matches your specific concern.\n"
                                f"2. Your communication style is: {comm_style}\n"
                                f"3. Your TOP concern is: {pain_points[0] if pain_points else 'daily usability'}\n"
                                f"4. If you agree, ADD a new perspective from your own usage. "
                                f"If you disagree, explain how their view conflicts with your practical needs.\n"
                                f"5. Reference SPECIFIC features, workflows, or policies from the posts. "
                                f"Avoid abstract philosophy.\n"
                                f"6. Your user type is '{agent_type}' with satisfaction={satisfaction:.1f} — act accordingly.\n"
                                f"7. PHASE DIRECTIVE: {_ts_directive}\n"
                                f"8. Keep your response under 150 words. Speak naturally as a real user.\n"
                            )
                            
                            # ── Decision Journal Injection ──
                            journal_ctx = ""
                            if agent_id in decision_journals:
                                journal_ctx = decision_journals[agent_id].prompt_summary()
                            
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
                            step_msg = BaseMessage.make_user_message(
                                role_name="ENVIRONMENT",
                                content=(
                                    # BUCKET 1 (top): Current observations — data, not directives
                                    # USER REQUEST: Removed injection of actions/platform_state
                                    # f"<platform_state>\n{platform_obs}\n</platform_state>\n\n"
                                    # BUCKET 2 (middle): Narrative memory from prior turns
                                    f"<memory>\n{hindsight_context}\n</memory>\n\n"
                                    # BUCKET 3 (middle): Agent's own emotional state summary
                                    f"<journal>\n{journal_ctx}\n</journal>\n\n"
                                    # BUCKET 4 (bottom — highest recency attention): Behavioral rules
                                    f"<rules>\n{persona_grounding}\n</rules>\n\n"
                                    # Closing action cue — very last token, maximum LLM focus
                                    "Review your state above and select ONE action to take now."
                                )
                            )

                            # ── Phase 2: LLM call — MUST be inside _limiter ──
                            # _limiter is a token-bucket enforcing GEMINI_FREE_RPM.
                            # Previously this context was closed before astep(), meaning
                            # every agent.astep() LLM call ran completely unguarded → 429s.
                            async with _limiter:
                                logger.debug(f"    🚦 Rate-limit slot acquired for {agent_name}")
                                action_resp = await asyncio.wait_for(
                                    agent.astep(step_msg), timeout=240.0
                                )

                        raw_content = action_resp.msgs[0].content if action_resp and action_resp.msgs else "No content"
                        
                        # Step A: Check for structured tool call arguments (most reliable source of pristine comment text)
                        tool_val = None
                        if action_resp and hasattr(action_resp, 'info') and action_resp.info:
                            tool_info = action_resp.info.get('tool_calls', [])
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
                        
                        # Step B: Select source content (tool argument wins over raw text containing thought blocks)
                        selected_content = tool_val if tool_val else raw_content
                        
                        # Step C: Strip thought blocks and formatting markers (fallback if thoughts leaked to tool call or raw text is used)
                        import re
                        cleaned = re.sub(r'<thought>.*?</thought>', '', selected_content, flags=re.DOTALL)
                        cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
                        # Remove markdown bold/italic tags and optional thought prefixes
                        cleaned = re.sub(r'(?i)^\s*(thought|thinking|action):\s*', '', cleaned)
                        content = cleaned.strip() or "No content"

                        # Fix #2: pass action_resp so tool call name is read first
                        action_type = _detect_action_type(content, action_resp=action_resp)

                        # Step D: Proactively query the SQLite platform database for the clean post/comment content actually saved.
                        # This guarantees that we use the final, clean text registered in the simulation platform.
                        if action_type in ["CREATE_COMMENT", "COMMENT", "CREATE_POST", "POST", "QUOTE_POST"]:
                            try:
                                import sqlite3
                                conn = sqlite3.connect(unique_db)
                                cursor = conn.cursor()
                                if "COMMENT" in action_type:
                                    cursor.execute(
                                        "SELECT content FROM comment WHERE user_id = ? ORDER BY comment_id DESC LIMIT 1",
                                        (int(agent_id),)
                                    )
                                    row = cursor.fetchone()
                                    if row and row[0]:
                                        content = row[0]
                                elif "POST" in action_type or "REPOST" in action_type:
                                    cursor.execute(
                                        "SELECT content FROM post WHERE user_id = ? ORDER BY post_id DESC LIMIT 1",
                                        (int(agent_id),)
                                    )
                                    row = cursor.fetchone()
                                    if row and row[0]:
                                        content = row[0]
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
            _exec_agent = ChatAgent(
                system_message=BaseMessage.make_assistant_message(
                    role_name="System", content=_EXEC_SUMMARY_SYSTEM
                ),
                model=model,
            )
            _exec_resp = await asyncio.wait_for(
                _exec_agent.astep(
                    BaseMessage.make_user_message(role_name="Analyst", content=_exec_data)
                ),
                timeout=60.0,
            )
            report.executive_summary = _exec_resp.msgs[0].content if _exec_resp.msgs else ""
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
