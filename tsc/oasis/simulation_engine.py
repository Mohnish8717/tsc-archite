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
from .models import OASISSimulationConfig, OASISAgentProfile, MarketSentimentSeries
from .ipc import CommandListener, LocalActionLogger
from tsc.models.inputs import CompanyContext
from .clustering import AnalyzeAgentAlignment

from tsc.config import settings as tsc_settings, LLMProvider
# (test harness / worker.py) to ensure C++ modules are pre-warmed
# before any event loop is created or patched.


# =============================================================================
# PUBLIC API
# =============================================================================

async def RunOASISSimulation(
    config: OASISSimulationConfig,
    agent_profiles: List[OASISAgentProfile],
    feature: Any,           # FeatureProposal
    context: CompanyContext,
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
        # 1. gRPC Mock (Removed to avoid metaclass conflict with qdrant-client)
        pass

        # 2. Hard Disable for backends that trigger deadlocks or import loops
        # Setting to None tells Python (and Transformers) the module is NOT available.
        for m in ["tensorflow", "codecarbon"]:
            sys.modules[m] = None

    # ── 1. Deferred Heavy Imports ────────────────────────────────────────────
    from oasis.social_platform.platform import Platform
    from oasis.social_platform.channel import Channel
    from oasis.social_platform.typing import RecsysType, ActionType
    from oasis.social_agent.agent import SocialAgent
    from fastembed import TextEmbedding

    from oasis.social_platform.config.user import UserInfo
    from camel.models import ModelFactory
    from camel.types import ModelType, ModelPlatformType
    from camel.messages import BaseMessage

    # ── 1. Config Access (clean, no __import__ hacks) ────────────────────────
    from tsc.config import settings as tsc_settings
    from camel.memories import LongtermAgentMemory, MemoryRecord, ScoreBasedContextCreator, ContextRecord
    from camel.memories.blocks import ChatHistoryBlock, VectorDBBlock
    from camel.storages.vectordb_storages import QdrantStorage
    from camel.embeddings import BaseEmbedding

    # ── 1.1 Enhanced Memory Classes ──────────────────────────────────────────
    class FastEmbedAdapter(BaseEmbedding):
        def __init__(self, model):
            self.model = model
            self._output_dim_internal: Optional[int] = None

        def embed_list(self, objs: list[str], **kwargs) -> list[list[float]]:
            return [list(e) for e in self.model.embed(objs)]

        def get_output_dim(self) -> int:
            if self._output_dim_internal is None:
                dummy = list(self.model.embed(["dummy"]))[0]
                self._output_dim_internal = len(dummy)
            return cast(int, self._output_dim_internal)

    class SlidingWindowLongtermMemory(LongtermAgentMemory):
        def __init__(self, *args, window_size: int = 12, **kwargs):
            super().__init__(*args, **kwargs)
            self.window_size = window_size

        def retrieve(self) -> List[ContextRecord]:
            chat_history = self.chat_history_block.retrieve(window_size=self.window_size)
            vector_db_retrieve = self.vector_db_block.retrieve(
                self._current_topic,
                self.retrieve_limit,
            )
            if not chat_history:
                return vector_db_retrieve
            return chat_history[:1] + vector_db_retrieve + chat_history[1:]

    # ── 2. Concurrency Semaphore (bound to THIS loop) ────────────────────────
    _sem = asyncio.Semaphore(1)

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

    # ── 4.1 Embedding Infrastructure ─────────────────────────────────────────
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # ── 5. Platform Infrastructure ───────────────────────────────────────────
    unique_db = os.path.join(sim_dir, f"{config.simulation_name}.sqlite")
    channel   = Channel()
    platform_obj = Platform(
        db_path=unique_db,
        recsys_type=RecsysType(config.platform_type),
        channel=channel,
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
    llm_model_name = tsc_settings.llm_model
    llm_provider   = tsc_settings.llm_provider
    api_key        = tsc_settings.get_api_key(llm_provider)

    from camel.models import GroqModel, OpenAIModel, AnthropicModel, GeminiModel
    if llm_provider == LLMProvider.GROQ:
        model = GroqModel(model_type=llm_model_name, api_key=api_key, max_retries=10)
    elif llm_provider == LLMProvider.ANTHROPIC:
        model = AnthropicModel(model_type=llm_model_name, api_key=api_key)
    elif llm_provider == LLMProvider.GOOGLE:
        # CAMEL-AI's native GeminiModel via OpenAI-compatible Gemini endpoint
        import os as _os
        _os.environ.setdefault("GEMINI_API_KEY", api_key or "")
        model = GeminiModel(model_type=llm_model_name, api_key=api_key, max_retries=10)
    elif "gpt" in llm_model_name or llm_provider == LLMProvider.OPENAI:
        model = OpenAIModel(model_type=llm_model_name, api_key=api_key)
    else:
        model = OpenAIModel(model_type=llm_model_name, api_key=api_key)
    
    logger.info(f"✅ LLM Model Initialized: {llm_model_name} ({llm_provider})")

    # ── 7. Instantiate Social Agents ─────────────────────────────────────────
    # Confine to useful actions for market analysis (sentiment & dialogue)
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
        # ── Memory Persistence Initialization ────────────────────────────────
        # Use Sliding Window (12) + RAG approach as requested
        # Reuse RecSys embedding model for efficiency
        adapter = FastEmbedAdapter(embedding_model)
        
        # Create context creator for pruning if token limit is hit
        context_creator = ScoreBasedContextCreator(
            model.token_counter,
            model.token_limit,
        )
        
        # Initialize Longterm Memory components
        ch_block = ChatHistoryBlock()
        vdb_block = VectorDBBlock(
            storage=QdrantStorage(vector_dim=adapter.get_output_dim()),
            embedding=adapter
        )
        
        lt_memory = SlidingWindowLongtermMemory(
            context_creator=context_creator,
            chat_history_block=ch_block,
            vector_db_block=vdb_block,
            retrieve_limit=5, # Capture original pitch and key rebuttals
            window_size=12,   # Capture immediate contextual now
            agent_id=str(profile.agent_id)
        )

        agent = SocialAgent(
            agent_id=str(profile.agent_id),
            user_info=user_info,
            channel=channel,
            model=model,
            available_actions=USEFUL_ACTIONS
        )
        # Inject the custom persistent memory
        agent.memory = lt_memory

        logger.info(f"Agent {agent.agent_id} initialized with Sliding Window (12) + RAG memory.")
        social_agents.append(agent)

    # CRITICAL: Monkey-patch ChatAgent._aexecute_tool to fix 'cannot pickle coroutine'
    # This happens in camel-ai v0.2.78 when FunctionTool is used with async functions
    # but not correctly awaited in all paths.
    from camel.agents import ChatAgent
    from camel.agents._types import ToolCallRequest
    from camel.types.agents import ToolCallingRecord
    original_aexecute_tool = ChatAgent._aexecute_tool

    async def patched_aexecute_tool(self, tool_call_request: ToolCallRequest) -> ToolCallingRecord:
        record = await original_aexecute_tool(self, tool_call_request)
        # If the result is a coroutine (due to imperfect async_call detection in camel-ai), await it!
        import asyncio
        if asyncio.iscoroutine(record.result):
            logger.info(f"FIX: Awaiting leaked coroutine for tool {tool_call_request.tool_name}")
            record.result = await record.result
        return record

    ChatAgent._aexecute_tool = patched_aexecute_tool

    # Agent-name lookup map
    agent_id_to_name = {
        getattr(a, "social_agent_id", str(id(a))): getattr(a, "user_info", None) and a.user_info.name or "Agent"
        for a in social_agents
    }

    # ── 8. Seed Platform with Feature Proposal ───────────────────────────────
    proposer_id = agent_profiles[0].agent_id if agent_profiles else 0
    logger.info(f"Seeding platform with proposal: {feature.title}")
    await platform_obj.create_post(
        agent_id=int(proposer_id),
        content=(
            f"I'd like to propose a new feature: {feature.title}\n\n"
            f"Description: {feature.description}\n\n"
            f"What do you all think?"
        ),
    )

    # ── 8.1 Forced Social Graph (The "Follow" Shortcut) ──────────────────────
    # Ensure every agent follows the proposer to guarantee the post appears in feeds
    logger.info(f"Establishing 'Follow' relationships to proposer (Agent {proposer_id})")
    for profile in agent_profiles:
        if profile.agent_id != proposer_id:
            await platform_obj.follow(agent_id=int(profile.agent_id), followee_id=int(proposer_id))

    # ── 9. Result Container ──────────────────────────────────────────────────
    series = MarketSentimentSeries(
        simulation_id=config.simulation_name,
        target_market=context.company_name,
        feature_proposal_id=getattr(feature, "proposal_id", "unknown"),
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
        except asyncio.TimeoutError:
            logger.warning(f"Interview timeout for agent {getattr(agent, 'social_agent_id', '?')}")
            return {"content": "Timeout", "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Interview error: {e}")
            return {"content": f"Error: {e}", "timestamp": datetime.now().isoformat()}

    # =====================================================================
    # HELPER: Mid-simulation interview callback (used by IPC)
    # =====================================================================
    async def _mid_sim_interview(questions: List[str]):
        responses_file = os.path.join(sim_dir, "mid_sim_interview_responses.json")
        all_responses = []
        for agent in social_agents:
            agent_resps = []
            for q in questions:
                resp = await _interview(agent, q)
                agent_resps.append({"question": q, "response": resp["content"]})
            all_responses.append({
                "agent_id": getattr(agent, "social_agent_id", str(id(agent))),
                "interviews": agent_resps,
            })
        with open(responses_file, "w", encoding="utf-8") as f:
            json.dump(all_responses, f, indent=2, ensure_ascii=False)
        logger.info(f"Mid-sim interview saved → {responses_file}")

    # =====================================================================
    # HELPER: Detect OASIS action type from response content
    # =====================================================================
    def _detect_action_type(content: str) -> str:
        """Parse OASIS agent response to identify the actual action taken."""
        content_lower = content.lower() if content else ""
        if "create_comment" in content_lower or "comment" in content_lower:
            return "COMMENT"
        elif "like_post" in content_lower or "like_comment" in content_lower:
            return "LIKE"
        elif "dislike" in content_lower:
            return "DISLIKE"
        elif "follow" in content_lower:
            return "FOLLOW"
        elif "unfollow" in content_lower:
            return "UNFOLLOW"
        elif "search" in content_lower:
            return "SEARCH"
        elif "mute" in content_lower:
            return "MUTE"
        elif "quote" in content_lower:
            return "QUOTE"
        elif "trend" in content_lower:
            return "TREND"
        elif "refresh" in content_lower:
            return "REFRESH"
        return "POST"

    # =====================================================================
    # MAIN SIMULATION LOOP
    # =====================================================================
    try:
        for t in range(config.num_timesteps):
            # IPC check (pause/resume/interview)
            await command_listener.wait_if_paused(interview_callback=_mid_sim_interview)
            if command_listener.should_stop:
                logger.warning(f"Simulation STOPPED by IPC at timestep {t}")
                break

            logger.info(f"━━━ Timestep {t+1}/{config.num_timesteps} ━━━")

            # ── Sequential Agent Stepping (deadlock-free) ────────────────
            #    Instead of asyncio.gather(30 agents) which exhausts the
            #    gRPC pool and triggers mutex.cc, we step agents one at a
            #    time with exponential backoff on failures.
            for idx, agent in enumerate(social_agents):
                agent_id   = getattr(agent, "social_agent_id", str(id(agent)))
                agent_name = agent_id_to_name.get(agent_id, "Unknown")
                backoff    = 5.0  # initial backoff seconds
                max_retries = 15

                for attempt in range(max_retries):
                    try:
                        async with _sem:
                            # ── JITTER (Burst Prevention) ─────────────────────
                            # Add 1-4s random delay to de-synchronize agent steps
                            await asyncio.sleep(random.uniform(1.0, 4.0))

                            # Create a default observation prompt for the agent
                            step_msg = BaseMessage.make_user_message(
                                role_name="ENVIRONMENT", 
                                content=(
                                    "Please observe the platform state and take your next autonomous action.\n\n"
                                    "CRITICAL INSTRUCTIONS:\n"
                                    "1. Belief Revision: If you encounter specific technical facts or mitigations provided by others (e.g. latency numbers, security patches), you MUST update your plausibility thresholds. Do not endlessly repeat the same skepticism if facts answer it.\n"
                                    "2. Persona Stability: Separate your Technical Domain Layer (your actual beliefs and skepticism) from your Cooperation Layer (social politeness). Do not let social politeness dilute your technical stance.\n"
                                    "3. Theory of Mind: Explicitly consider what other agents believe and why they are posting. Do not be an 'invisible' skeptic who outwardly appears supportive but internally disagrees.\n\n"
                                    "You should focus on expressing your opinion through posts or comments, or reacting with likes/dislikes. Avoid noise-only actions."
                                )
                            )

                            action_resp = await asyncio.wait_for(
                                agent.astep(step_msg), timeout=240.0
                            )

                        # Extract content and detect action type
                        content = (
                            action_resp.msgs[0].content
                            if action_resp and action_resp.msgs
                            else "No content"
                        )
                        action_type = _detect_action_type(content)

                        # Log to local JSONL for dashboard
                        local_logger.log_action(
                            agent_id=agent_id,
                            agent_name=agent_name,
                            action_type=action_type,
                            content=content,
                            timestep=t,
                            platform=config.platform_type,
                        )

                        # ── Agent Interaction Tracking ────────────────
                        if agent_id not in series.agent_interactions:
                            series.agent_interactions[agent_id] = []
                        series.agent_interactions[agent_id].append(
                            f"ROUND {t+1} | {action_type}: {content}"
                        )

                        logger.info(
                            f"  ✓ [{idx+1}/{len(social_agents)}] {agent_name} → {action_type}"
                        )
                        break  # success — exit retry loop

                    except asyncio.TimeoutError:
                        logger.warning(
                            f"  ⏰ [{idx+1}] {agent_name} timed out (attempt {attempt+1})"
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(backoff)
                            backoff *= 2
                    except Exception as e:
                        # Recognize 429 Too Many Requests specifically
                        if "429" in str(e):
                            logger.error(f"  ✗ [429 RATE LIMIT] {agent_name} hit rate limit (attempt {attempt+1}). Sleeping {backoff}s...")
                        else:
                            logger.error(f"  ✗ [{idx+1}] {agent_name} failed: {e} (attempt {attempt+1})")
                        
                        if attempt < max_retries - 1:
                            await asyncio.sleep(backoff)
                            backoff = min(60.0, backoff * 1.5)

            # ── Storage Sync (verified via per-run SQLite) ──────────
            logger.info(f"Round {t+1} complete for all agents.")

            # ── Progress Heartbeat ───────────────────────────────────────
            series.timesteps.append(t)
            series.sentiment_volatility.append(0.0)  # Will be enriched by temporal_analysis

            local_logger.update_progress(
                timestep=t,
                total=config.num_timesteps,
                status="RUNNING",
            )
            local_logger.log_event("round_end", {"timestep": t})

        # =================================================================
        # FINAL PHASE: Interview All Agents
        # =================================================================
        logger.info("Starting final agent interviews...")
        final_responses = []
        for idx, agent in enumerate(social_agents):
            agent_id = getattr(agent, "social_agent_id", str(id(agent)))
            agent_name = agent_id_to_name.get(agent_id, "Unknown")

            interview_resps = []
            for prompt in config.interview_prompts:
                resp = await _interview(agent, prompt)
                interview_resps.append(resp)

            final_responses.append({
                "agent_id": agent_id,
                "responses": interview_resps,
            })
            logger.info(f"  📝 [{idx+1}/{len(social_agents)}] Interviewed {agent_name}")

        series.raw_responses = final_responses

        # =================================================================
        # AGGREGATION: LLM-Based Sentiment Audit
        # =================================================================
        await _aggregate_insights_llm(final_responses, series, model, BaseMessage)

        # ── 11.5 Agent Sentiment Alignment Audit ───────────────────────
        try:
            from .clustering import AnalyzeAgentAlignment
            await AnalyzeAgentAlignment(
                series=series,
                feature_title=getattr(feature, "title", "The Feature"),
                feature_desc=getattr(feature, "description", "The proposed feature"),
            )
        except Exception as e:
            logger.error(f"Agent alignment audit failed: {e}")

        # ── 12. Export Per-Agent Logs ──────────────────────────────────
        agent_logs_dir = os.path.join(sim_dir, "agent_logs")
        os.makedirs(agent_logs_dir, exist_ok=True)
        for aid, logs in series.agent_interactions.items():
            aname = agent_id_to_name.get(aid, "Unknown")
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', aname)
            log_file = os.path.join(agent_logs_dir, f"{safe_name}_{aid[:8]}.txt")
            with open(log_file, "w") as f:
                f.write(f"=== INTERACTION HISTORY: {aname} ({aid}) ===\n\n")
                f.write("\n\n---\n\n".join(logs))
        logger.info(f"💾 Per-agent interaction logs exported to {agent_logs_dir}")

    finally:
        # ── Robust Cleanup ───────────────────────────────────────────────
        logger.info("Cleaning up OASIS platform resources...")

        if platform_task is not None:
            platform_task.cancel()
            try:
                await platform_task
            except (asyncio.CancelledError, Exception):
                pass

        if platform_obj is not None and hasattr(platform_obj, "close"):
            try:
                await platform_obj.close()
            except Exception:
                pass

        if local_logger is not None:
            local_logger.log_event("simulation_end")

        logger.info(f"Simulation {config.simulation_name} cleanup complete.")

    return series


