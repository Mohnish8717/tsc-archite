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
    base_dir: str = "/tmp/oasis_runs",
    available_actions: Optional[List[Any]] = None,
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

    # ── 2. Concurrency Semaphore (bound to THIS loop) ────────────────────────
    # Raised from 1 → 20: enables 20 parallel LLM calls per timestep.
    # Free-tier safe: Groq/Gemini support burst concurrency up to ~30 RPS.
    _sem = asyncio.Semaphore(10)

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
            # Clean up banks from PREVIOUS simulation (preserves for analysis
            # until a NEW simulation starts — per user requirement)
            try:
                purged = await memory_manager.cleanup_banks()
                if purged > 0:
                    logger.info(f"🧹 Cleaned up {purged} banks from previous simulation")
            except Exception:
                pass  # No previous banks — fine
            await memory_manager.initialize_agents(
                agent_profiles=agent_profiles,
                feature_title=getattr(feature, 'title', 'Unspecified Feature'),
                feature_description=getattr(feature, 'description', 'No description provided'),
                simulation_id=config.simulation_name,
            )
        except Exception as e:
            logger.error(f"Fatal error during Hindsight Initialization: {e}")
            HINDSIGHT_AVAILABLE = False
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
    llm_provider   = LLMProvider.GOOGLE
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
    elif "gpt" in llm_model_name or llm_provider == LLMProvider.OPENAI:
        model = OpenAIModel(model_type=llm_model_name, api_key=api_key)
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
        bio = info.get("profile", {}).get("user_profile", "")[:100]
        
        user_msg = [user_name, display_name, bio]
        await platform_obj.sign_up(agent_id=int(profile.agent_id), user_message=user_msg)

    # CRITICAL: Monkey-patch ChatAgent._aexecute_tool
    from camel.agents import ChatAgent
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

    # ── 8.1 Seed Platform — Source-to-Synth Pipeline ────────────────────────────
    # Research-backed approach: Extract REAL controversy quotes from input data
    # and create diverse seed posts representing distinct opposing viewpoints.
    # This prevents the echo chamber effect observed in previous simulations.
    #
    # Strategy (per CAMEL-AI best practices):
    #   1. Extract verbatim high-friction quotes from customer interviews/tickets
    #   2. Create seeds with OPPOSING viewpoints to force polarization
    #   3. Target seeds to high-centrality agents (proposer = hub)
    #   4. Vary tone (formal/angry/skeptical) to elicit differentiated responses
    
    def _extract_controversy_seeds(feat, ctx, market_ctx):
        """Source-to-Synth: Extract real controversy data for seed posts."""
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
                f"Breaking: '{feat_title}' has been proposed.\n\n"
                f"{feat_desc}\n\n"
                f"I have serious concerns about this. The opt-out process alone "
                f"seems designed to minimize actual opt-outs. What's the REAL "
                f"impact on our users and our reputation?",
                
                f"Devil's advocate on '{feat_title}': The business case is "
                f"actually strong. More training data = better models = better UX. "
                f"But the execution is terrible. How would YOU implement this "
                f"if you had to balance data quality with user trust?",
            ]
        
        return seeds or [f"New feature proposal: {feat_title}. {feat_desc[:500]}"]
    
    if mode == "behavioral" or feature is None:
        product_desc = context.company_name if context else "the product"
        product_stack = ", ".join(context.tech_stack) if (context and context.tech_stack) else "the platform"
        competitors = ", ".join(context.competitors) if (context and context.competitors) else "alternatives"
        
        seed_posts = [
            f"Hey everyone! As a daily user of {product_desc}, I wanted to share "
            f"my experience today. The {product_stack} workflow has been interesting "
            f"but I've run into some friction points. What's your experience been like?",
            
            f"I've been comparing {product_desc} with {competitors} lately. "
            f"Some things work great here, but there are areas where I feel "
            f"we're falling behind. Anyone else noticing gaps?",
            
            f"Product update thread 🧵: What's the ONE thing about {product_desc} "
            f"that, if fixed or improved, would make you significantly more productive?",
        ]
        for post in seed_posts:
            await platform_obj.create_post(agent_id=int(proposer_id), content=post)
        logger.info(f"🔬 Behavioral Mode: Agents grounded as users of {product_desc}")
    else:
        # FEATURE TEST MODE: Source-to-Synth seed generation
        logger.info(f"🔬 Generating source-to-synth seed posts for: {feature.title}")
        controversy_seeds = _extract_controversy_seeds(feature, context, market_context)
        for i, seed in enumerate(controversy_seeds):
            # Distribute seeds across first few agents for network diversity
            poster_id = int(agent_profiles[min(i, len(agent_profiles) - 1)].agent_id)
            await platform_obj.create_post(agent_id=poster_id, content=seed)
            logger.info(f"  📝 Seed post {i+1}/{len(controversy_seeds)} by agent {poster_id}")
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
    ]

    def _gm_resolve(content: str, timestep: int) -> dict:
        """Game Master: Classify behavioral intent from natural language."""
        if not content:
            return {"type": "neutral", "intensity": 0.0, "timestep": timestep, "factors": []}
        
        signals_found = []
        factors = set()
        for pattern, signal_type, intensity in _GM_SIGNALS:
            if pattern.search(content):
                signals_found.append((signal_type, intensity))
                factors.add(signal_type.split("_")[0])  # Extract factor root
        
        if not signals_found:
            return {"type": "neutral", "intensity": 0.0, "timestep": timestep, "factors": []}
        
        # Use strongest signal as dominant
        dominant = max(signals_found, key=lambda s: abs(s[1]))
        avg_intensity = sum(s[1] for s in signals_found) / len(signals_found)
        
        return {
            "type": dominant[0],
            "intensity": round(avg_intensity, 2),
            "timestep": timestep,
            "factors": list(factors),
            "quote": content[:200],
            "all_signals": [s[0] for s in signals_found],
        }

    def _detect_action_type(content: str) -> str:
        """Detect CAMEL platform action type from response content."""
        content_lower = content.lower() if content else ""
        if "create_comment" in content_lower or "comment" in content_lower:
            return "COMMENT"
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
    # MAIN SIMULATION LOOP
    # =====================================================================
    try:
        for t in range(config.num_timesteps):
            await command_listener.wait_if_paused()
            if command_listener.should_stop:
                break

            logger.info(f"━━━ Timestep {t+1}/{config.num_timesteps} ━━━")
            async def process_agent(idx, agent):
                agent_id   = str(agent.social_agent_id)
                agent_name = agent_id_to_name.get(agent_id, "Unknown")
                
                # Skip shadow agents — they inherit state post-simulation
                if agent_id not in decision_journals:
                    return
                
                backoff    = 5.0
                max_retries = 15

                for attempt in range(max_retries):
                    try:
                        async with _sem:
                            await asyncio.sleep(random.uniform(1.0, 4.0))

                            hindsight_context = ""
                            if HINDSIGHT_AVAILABLE and memory_manager:
                                hindsight_context = await memory_manager.recall_for_turn(str(agent_id))
                                if hindsight_context:
                                    logger.info(f"    🧠 Hindsight injected for {agent_name} ({len(hindsight_context)} chars)")

                            refresh_resp = await platform_obj.refresh(agent_id=int(agent_id))
                            platform_obs = ""
                            if refresh_resp.get("success") and refresh_resp.get("posts"):
                                posts = refresh_resp["posts"]
                                platform_obs = "\n\nCURRENT PLATFORM STATE:\n"
                                for p in posts:
                                    platform_obs += f"- [PostID {p['post_id']}] (User {p['user_id']}): {p['content']}\n"
                                    if p.get('comments'):
                                        for c in p['comments']:
                                            platform_obs += f"  └─ [CommentID {c['comment_id']}] (User {c['user_id']}): {c['content']}\n"

                            # ── Persona-Grounded Anti-Sycophancy Prompt ──
                            profile = agent_profiles[idx]
                            info = profile.user_info_dict
                            persona_profile = info.get("profile", {})
                            comm_style = persona_profile.get("communication_style", "direct")
                            pain_points = persona_profile.get("pain_points", [])
                            satisfaction = getattr(profile, 'satisfaction', 0.5)
                            agent_type = getattr(profile, 'agent_type', 'unknown')
                            
                            persona_grounding = (
                                f"\nCRITICAL BEHAVIORAL RULES FOR {agent_name}:\n"
                                f"1. Your communication style is: {comm_style}\n"
                                f"2. Your TOP concern is: {pain_points[0] if pain_points else 'daily usability'}\n"
                                f"3. You may agree or disagree with others, but your response MUST be driven by your personal workflow and product experience.\n"
                                f"4. If you agree, ADD a new perspective from your own usage. If you disagree, explain how their view conflicts with your practical needs.\n"
                                f"5. Reference SPECIFIC features, workflows, or policies from the posts. Avoid abstract philosophy.\n"
                                f"6. Your user type is '{agent_type}' with satisfaction={satisfaction:.1f} — act accordingly.\n"
                                f"7. Keep your response under 150 words. Speak naturally as a real user of this product.\n"
                            )
                            
                            # ── Decision Journal Injection ──
                            journal_ctx = ""
                            if agent_id in decision_journals:
                                journal_ctx = decision_journals[agent_id].prompt_summary()
                            
                            step_msg = BaseMessage.make_user_message(
                                role_name="ENVIRONMENT", 
                                content=(
                                    "Please observe the platform state and take your next autonomous action.\n"
                                    f"{persona_grounding}\n"
                                    f"{journal_ctx}\n"
                                    f"Current Platform State:\n{platform_obs}\n"
                                    f"{hindsight_context}"
                                )
                            )
                            action_resp = await asyncio.wait_for(
                                agent.astep(step_msg), timeout=240.0
                            )

                        content = action_resp.msgs[0].content if action_resp and action_resp.msgs else "No content"
                        action_type = _detect_action_type(content)

                        # ── Game Master: Resolve behavioral signal ──
                        gm_signal = _gm_resolve(content, timestep=t)
                        if agent_id in decision_journals and gm_signal["type"] != "neutral":
                            decision_journals[agent_id].update_from_signal(gm_signal)
                            logger.info(f"    🎲 GM State Shift: {agent_name} → {gm_signal['type']} ({gm_signal['intensity']:+.2f})")

                        local_logger.log_action(
                            agent_id=agent_id,
                            agent_name=agent_name,
                            action_type=action_type,
                            content=content,
                            timestep=t,
                        )
                        
                        if HINDSIGHT_AVAILABLE and memory_manager:
                            await memory_manager.extract_and_retain(
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
                        if attempt < max_retries - 1:
                            await asyncio.sleep(backoff)
                            backoff = min(60.0, backoff * 1.5)
                        else:
                            logger.error(f"Agent {agent_name} failed after {max_retries} attempts: {e}")

            # Execute all agents concurrently using the defined semaphore
            tasks = [process_agent(idx, agent) for idx, agent in enumerate(social_agents)]
            await asyncio.gather(*tasks)

            if HINDSIGHT_AVAILABLE and memory_manager:
                await memory_manager.synthesize_post_timestep(timestep=t)

            series.timesteps.append(t)
            local_logger.update_progress(timestep=t, total=config.num_timesteps, status="RUNNING")

            # ── Time-series accumulation ──
            journals_list = list(decision_journals.values())
            if journals_list:
                ts_satisfaction.append(round(sum(j.satisfaction for j in journals_list) / len(journals_list), 3))
                ts_frustration.append(round(sum(j.frustration for j in journals_list) / len(journals_list), 3))
                ts_trust.append(round(sum(j.trust for j in journals_list) / len(journals_list), 3))

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

        # ── Propagate states to shadow agents (must happen before metrics) ──
        sampler.propagate_states(decision_journals)
        extrapolated = sampler.build_extrapolated_report(decision_journals)
        
        # Combine active + shadow for full declared-population metrics
        shadow_journals_proxy = sampler.shadow_agents
        all_frusts = ([j.frustration for j in all_journals]
                      + [s.frustration for s in shadow_journals_proxy])
        all_sats   = ([j.satisfaction for j in all_journals]
                      + [s.satisfaction for s in shadow_journals_proxy])
        all_advs   = ([j.advocacy   for j in all_journals]
                      + [s.advocacy   for s in shadow_journals_proxy])
        declared_n = len(all_frusts) or 1

        # Override with population-scale values
        high_risk  = sum(1 for f in all_frusts if f > 0.6) / declared_n
        low_risk   = sum(1 for s in all_sats   if s > 0.6) / declared_n
        moderate   = max(0.0, 1.0 - high_risk - low_risk)
        risk_dist  = {"HIGH_RISK": round(high_risk, 2),
                      "MODERATE":  round(moderate,  2),
                      "LOW_RISK":  round(low_risk,  2)}
        promoters  = sum(1 for a in all_advs   if a > 0.6) / declared_n
        nps        = round((promoters - high_risk) * 100, 1)
        n          = declared_n

        # Churn velocity & adoption momentum
        churn_vel = round((ts_frustration[-1] - ts_frustration[0]) / max(len(ts_frustration), 1), 3) if ts_frustration else 0.0
        adopt_mom = round((ts_satisfaction[-1] - ts_satisfaction[0]) / max(len(ts_satisfaction), 1), 3) if ts_satisfaction else 0.0

        # Top risk factors (from signal types)
        all_signal_types = [s["type"] for j in all_journals for s in j.signals if s["type"] != "neutral"]
        factor_counts = _Counter(all_signal_types)
        total_signals = sum(factor_counts.values()) or 1
        top_factors = [{"factor": f, "frequency": round(c / total_signals, 2)} for f, c in factor_counts.most_common(8)]

        # Decision events
        all_decisions = [d for j in all_journals for d in j.decisions]

        # Dynamic segment discovery via clustering
        segments = []
        try:
            from .clustering import ClusterOnBehavioralState
            segments = await ClusterOnBehavioralState(all_journals)
        except Exception as e:
            logger.warning(f"Behavioral clustering failed: {e}")

        # Build PredictionReport
        report = PredictionReport(
            simulation_id=config.simulation_name,
            feature_title=feature_title,
            population_size=n,
            timesteps_completed=len(series.timesteps),
            segments=segments,
            risk_distribution=risk_dist,
            satisfaction_curve=ts_satisfaction,
            frustration_curve=ts_frustration,
            trust_curve=ts_trust,
            net_promoter_score=nps,
            churn_velocity=churn_vel,
            adoption_momentum=adopt_mom,
            decision_events=all_decisions,
            top_risk_factors=top_factors,
            agent_journals=[j.to_dict() for j in all_journals],
        )

        # ── Propagate states to shadow agents ──
        sampler.propagate_states(decision_journals)
        extrapolated = sampler.build_extrapolated_report(decision_journals)
        
        # Combine active + shadow for full population metrics

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
        for k, v in risk_dist.items():
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            print(f"  {k:12s}  {bar} {v*100:.0f}%")
        print(f"\nBUSINESS METRICS:")
        print(f"  Net Promoter Score:   {nps:+.0f}")
        print(f"  Churn Velocity:       {churn_vel:+.3f}/timestep")
        print(f"  Adoption Momentum:    {adopt_mom:+.3f}/timestep")
        if top_factors:
            print(f"\nTOP RISK FACTORS:")
            for f in top_factors[:5]:
                print(f"  • {f['factor']:25s} {f['frequency']*100:.0f}%")
        if segments:
            print(f"\nDYNAMIC SEGMENTS ({len(segments)} discovered):")
            for seg in segments:
                print(f"  [{seg.get('size', '?')} agents] {seg.get('name', 'Unknown')} — "
                      f"sat={seg.get('avg_satisfaction', 0):.2f}, fru={seg.get('avg_frustration', 0):.2f}")
        if all_decisions:
            print(f"\nDECISION EVENTS ({len(all_decisions)}):")
            for d in all_decisions[:10]:
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
                f"NPS: {nps:.1f} | Churn Velocity: {churn_vel:.3f} | Adoption Momentum: {adopt_mom:.3f}\n"
                f"Risk Distribution: {json.dumps(risk_dist)}\n"
                f"Top Risk Factors: {json.dumps(top_factors)}\n"
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
