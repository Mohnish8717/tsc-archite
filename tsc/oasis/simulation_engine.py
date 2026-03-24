import os
import uuid
import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# External OASIS Imports moved inside RunOASISSimulation to prevent import-time deadlocks
Platform = Any
Channel = Any
RecsysType = Any
SocialAgent = Any
UserInfo = Any
ModelFactory = Any
ModelType = Any
ModelPlatformType = Any
BaseMessage = Any

from .models import (
    OASISAgentProfile,
    OASISSimulationConfig,
    MarketSentimentSeries,
    UserInfoAdapter,
    OpinionVector,
    BeliefCluster
)
from .ipc import CommandListener, LocalActionLogger
from tsc.models.inputs import CompanyContext

logger = logging.getLogger("tsc.oasis.engine")

async def RunOASISSimulation(
    config: OASISSimulationConfig,
    agent_profiles: List[OASISAgentProfile],
    feature: Any, # FeatureProposal
    context: CompanyContext,
    market_context: Optional[Dict[str, Any]] = None,
    zep_client: Optional[Any] = None,
    base_dir: str = "/tmp/oasis_runs"
) -> MarketSentimentSeries:
    """
    Run actual CAMEL-AI OASIS simulation using Platform and SocialAgents.
    """
    # Deferred Heavy Imports to avoid C++ deadlocks on macOS
    global Platform, Channel, RecsysType, SocialAgent, UserInfo, ModelFactory, ModelType, ModelPlatformType, BaseMessage
    from oasis.social_platform.platform import Platform
    from oasis.social_platform.channel import Channel
    from oasis.social_platform.typing import RecsysType
    from oasis.social_agent.agent import SocialAgent
    from oasis.social_platform.config.user import UserInfo
    from camel.models import ModelFactory
    from camel.types import ModelType, ModelPlatformType
    from camel.messages import BaseMessage

    async def limited_interview(agent: "SocialAgent", question: str) -> Dict[str, Any]:
        """Helper to query a SocialAgent with a timeout/retry or concurrency limit."""
        try:
            msg = BaseMessage.make_user_message(role_name="INTERVIEWER", content=question)
            response = await agent.step(msg) 
            return {
                "content": response.msgs[0].content if response.msgs else "No response",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Interview failed: {e}")
            return {"content": f"Error: {e}", "timestamp": datetime.now().isoformat()}

    async def perform_mid_sim_interview(questions: List[str]):
        """Callback for mid-simulation querying via IPC."""
        responses_file = os.path.join(sim_dir, "mid_sim_interview_responses.json")
        all_responses = []
        
        for agent in social_agents:
            agent_responses = []
            for q in questions:
                resp = await limited_interview(agent, q)
                agent_responses.append({"question": q, "response": resp["content"]})
            
            all_responses.append({
                "agent_id": getattr(agent, "social_agent_id", str(id(agent))),
                "interviews": agent_responses
            })
        
        with open(responses_file, 'w', encoding='utf-8') as f:
            json.dump(all_responses, f, indent=2, ensure_ascii=False)
        logger.info(f"Mid-simulation interview responses saved to {responses_file}")

    logger.info(f"Starting actual OASIS simulation: {config.simulation_name}")

    # 0. Setup IPC and Local Logging
    sim_dir = os.path.join(base_dir, config.simulation_name)
    os.makedirs(sim_dir, exist_ok=True)
    command_listener = CommandListener(config.simulation_name, sim_dir)
    local_logger = LocalActionLogger(sim_dir)

    # 1. Deferred Heavy Imports to avoid C++ deadlocks on macOS
    global Platform, Channel, RecsysType, SocialAgent, UserInfo, ModelFactory, ModelType, ModelPlatformType, BaseMessage
    from oasis.social_platform.platform import Platform
    from oasis.social_platform.channel import Channel
    from oasis.social_platform.typing import RecsysType
    from oasis.social_agent.agent import SocialAgent
    from oasis.social_platform.config.user import UserInfo
    from camel.models import ModelFactory
    from camel.types import ModelType, ModelPlatformType
    from camel.messages import BaseMessage

    # 1b. Setup Platform Infrastructure (Unique DB Path)
    unique_db = os.path.join(sim_dir, f"{config.simulation_name}.sqlite")
    channel = Channel()
    platform = Platform(
        db_path=unique_db,
        recsys_type=RecsysType(config.platform_type),
        channel=channel
    )
    
    # Start platform in background
    platform_task = asyncio.create_task(platform.running())
    
    # 2. Instantiate LLM Model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.GROQ,
        model_type=ModelType.GROQ_LLAMA_3_1_8B,
    )

    # 3. Instantiate Social Agents from Profiles
    social_agents = []
    for profile in agent_profiles:
        user_info = UserInfo(**profile.user_info_dict)
        agent = SocialAgent(
            agent_id=str(profile.agent_id),
            user_info=user_info,
            model=model
        )
        social_agents.append(agent)
        
    # 4. Seed Platform with Feature Proposal
    proposer_id = agent_profiles[0].agent_id if agent_profiles else 0
    logger.info(f"Seeding platform with proposal: {feature.title} by agent {proposer_id}")
    await platform.create_post(
        agent_id=int(proposer_id),
        content=f"I'd like to propose a new feature: {feature.title}\n\nDescription: {feature.description}\n\nWhat do you all think?"
    )

    series = MarketSentimentSeries(
        simulation_id=config.simulation_name,
        target_market=context.company_name,
        feature_proposal_id=getattr(feature, "proposal_id", "unknown")
    )
    
    # Map for agent name resolution
    agent_id_to_name = { 
        getattr(a, "social_agent_id", str(id(a))): a.user_info.name 
        for a in social_agents 
    }

    try:
        # 5. Core Simulation Loop
        for t in range(config.num_timesteps):
            # Mid-Simulation Interview & IPC Check
            await command_listener.wait_if_paused(interview_callback=perform_mid_sim_interview)
            if command_listener.should_stop:
                logger.warning(f"Simulation {config.simulation_name} STOPPED by IPC at timestep {t}")
                break
                
            logger.info(f"Starting Timestep {t}/{config.num_timesteps}...")
            
            # Agent Interaction Logic
            async def limited_action(agent):
                try:
                    # Sync to platform
                    action_resp = await agent.step()
                    # Log locally for dashboard with granular metadata
                    agent_id = getattr(agent, "social_agent_id", str(id(agent)))
                    agent_name = agent_id_to_name.get(agent_id, "Unknown Agent")
                    
                    local_logger.log_action(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        action_type="POST", # OASIS agents usually post
                        content=action_resp.msgs[0].content if action_resp.msgs else "No content",
                        timestep=t,
                        platform=config.platform_type
                    )
                    return action_resp
                except Exception as e:
                    logger.error(f"Action failed for agent: {e}")
                    return None

            action_tasks = [limited_action(agent) for agent in social_agents]
            actions = await asyncio.gather(*action_tasks, return_exceptions=True)
            
            # 5b. Periodic Zep Sync (Batching for cost efficiency)
            if zep_client and t % 5 == 0:
                all_fact_data = []
                for i, action in enumerate(actions):
                    if action and not isinstance(action, Exception):
                        fact = await _prepare_fact_data(
                            getattr(social_agents[i], "social_agent_id", str(id(social_agents[i]))),
                            action, t
                        )
                        if fact: all_fact_data.append(fact)
                
                if all_fact_data:
                    await zep_client.ingest_facts(all_fact_data)
                    logger.info(f"Batch-synced {len(all_fact_data)} agent actions to Zep for timestep {t}")
            
            from tsc.oasis.temporal_analysis import CalculateVolatility
            
            # Progress Heartbeat (Post-Timestep)
            # Track instantaneous timeline context
            volatility = CalculateVolatility(agent_profiles)
            series.timesteps.append(t)
            series.sentiment_volatility.append(volatility)
            
            local_logger.update_progress(
                timestep=t,
                total=config.num_timesteps,
                status="RUNNING"
            )
            local_logger.log_event("round_end", {"timestep": t, "volatility": volatility})
            
        # 6. Final Recommendation Phase: Interviewing Agents
        logger.info("Starting prediction reports via agent interviews...")
        final_responses = []
        for agent in social_agents:
            interview_tasks = [
                limited_interview(agent, prompt) 
                for prompt in config.interview_prompts
            ]
            responses = await asyncio.gather(*interview_tasks)
            final_responses.append({
                "agent_id": getattr(agent, "social_agent_id", str(id(agent))),
                "responses": responses
            })
        from tsc.oasis.clustering import PerformBehavioralClustering
        
        # 6b. Behavioral Clustering (MiroFish standard)
        logger.info("Performing behavioral clustering on agent responses...")
        series.segment_breakdown = await PerformBehavioralClustering(
            agents=agent_profiles,
            simulation_results=final_responses
        )
            
        # 7. Aggregate Insights (Structured LLM Audit)
        await _aggregate_insights_llm(final_responses, series, model)
        
    finally:
        # Robust Cleanup (Prevent Zombie Processes)
        logger.info(f"Cleaning up OASIS platform resources (DB: {unique_db})...")
        platform_task.cancel()
        try:
            await platform_task
        except asyncio.CancelledError:
            pass
        if hasattr(platform, "close"):
            await platform.close()
            
        local_logger.log_event("simulation_end")
            
    return series

async def _prepare_fact_data(agent_id: str, action: Any, timestep: int) -> Optional[Dict[str, Any]]:
    """Helper to format agent action for Zep ingestion."""
    content = ""
    if isinstance(action, str):
        content = action
    elif isinstance(action, dict):
        content = action.get("content") or action.get("text") or str(action)
    else:
        content = str(action)
        
    if not content:
        return None

    return {
        "fact": f"Agent {agent_id} performed action: {content}",
        "created_at": datetime.now().isoformat(),
        "metadata": {
            "source": "OASIS_SIMULATION",
            "agent_id": agent_id,
            "timestep": timestep,
            "type": "SIMULATION_ACTION"
        }
    }

async def _aggregate_insights_llm(
    responses: List[Dict[str, Any]], 
    series: MarketSentimentSeries,
    model: Any
):
    """
    Derive market fit metrics using a Stratified LLM Auditor instead of heuristics.
    """
    logger.info("Performing stratified sentiment audit on simulation results...")
    
    # Stratified Sampling: Distinguish between Bullish/Bearish proxies locally
    bullish_bucket = []
    bearish_bucket = []
    neutral_bucket = []

    pos_keywords = re.compile(r"(love|great|good|excellent|needed|helpful|yes|awesome)", re.I)
    neg_keywords = re.compile(r"(expensive|hate|bad|confusing|redundant|no|useless|flaw)", re.I)

    for r in responses:
        content_blob = " ".join([resp.get("content", "") for resp in r["responses"]])
        if pos_keywords.search(content_blob):
            bullish_bucket.append(r)
        elif neg_keywords.search(content_blob):
            bearish_bucket.append(r)
        else:
            neutral_bucket.append(r)

    # Sample equally from each bucket to ensure spectrum visibility
    sample = []
    per_bucket = 7
    sample.extend(bullish_bucket[:per_bucket])
    sample.extend(bearish_bucket[:per_bucket])
    sample.extend(neutral_bucket[:per_bucket])

    # Fallback to simple slice if buckets are empty or too small
    if not sample:
        sample = responses[:20]
    
    audit_context = ""
    for r in sample:
        answers = [resp.get("content", "") for resp in r["responses"]]
        audit_context += f"Agent {r['agent_id']}: {' / '.join(answers)}\n\n"

    audit_prompt = f"""
    Analyze the following market simulation interviews for a new feature proposal.
    Format your response as a JSON object with:
    - adoption_score: float (0.0 to 1.0)
    - consensus_verdict: string ("BULLISH", "BEARISH", "NEUTRAL")
    - key_objections: list of strings
    
    Interviews (Stratified Sample):
    {audit_context}
    """

    try:
        msg = BaseMessage.make_user_message(role_name="Market Auditor", content=audit_prompt)
        response = await model.run(msg)
        
        content = response.msgs[0].content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            series.final_adoption_score = data.get("adoption_score", 0.5)
            series.consensus_verdict = data.get("consensus_verdict", "NEUTRAL")
            series.key_objections = data.get("key_objections", [])
        else:
            logger.warning("LLM Auditor failed to return JSON.")
            
    except Exception as e:
        logger.error(f"Audit failed: {e}")

    logger.info(f"Simulation finalized with LLM Audit verdict: {series.consensus_verdict}")