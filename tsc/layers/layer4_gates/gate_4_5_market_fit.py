import asyncio
import logging
import time
from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

from tsc.gates.base import BaseGate
from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.models.personas import FinalPersona
from tsc.models.graph import KnowledgeGraph
from tsc.models.chunks import ProblemContextBundle
from tsc.models.gates import GateResult, GateVerdict, RiskEntry

# OASIS Imports
from tsc.oasis.models import OASISSimulationConfig, MarketSentimentSeries
from tsc.oasis.simulation_runner import SimulationRunner
from tsc.oasis.clustering import PerformBehavioralClustering, DetectConsensus

# Legacy Fallback
from tsc.mesa.simulation import RunMesaSimulation

logger = logging.getLogger(__name__)

class MarketFitMode(str, Enum):
    OASIS_ONLY = "OASIS_ONLY"
    MESA_ONLY = "MESA_ONLY"
    HYBRID_DUAL = "HYBRID_DUAL"

class MarketFitGate(BaseGate):
    """
    Gate 4.5: Market Fit (Actual CAMEL-AI OASIS Integration)
    Orchestrates the high-fidelity social simulation with behavioral analysis.
    """
    gate_id = "4.5"
    gate_name = "Market Fit (OASIS Social Simulation)"
    gate_domain = "market analysis, social simulation, adoption forecasting"

    def __init__(
        self, 
        llm_client: Any,
        graph_store: Optional[Any] = None,
        mode: MarketFitMode = MarketFitMode.HYBRID_DUAL,
        num_agents: int = 150, # Optimized for actual OASIS runs
        enable_parallel: bool = True
    ):
        super().__init__(llm_client)
        self.mode = mode
        self.num_agents = num_agents
        self.enable_parallel = enable_parallel
        self._zep_client = graph_store # Store Zep client for simulation sync
        self.config = OASISSimulationConfig(
            num_agents=num_agents,
            timesteps=10, # Balanced for performance/fidelity
            parallel_execution=enable_parallel
        )

    async def evaluate(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: List[FinalPersona]
    ) -> GateResult:
        """Execute the actual OASIS market fit simulation."""
        t0 = time.time()
        logger.info(f"Starting Market Fit Gate (Actual OASIS Mode: {self.mode.value})")

        oasis_result = None
        mesa_result = None
        verdict = GateVerdict.PASS
        score = 0.5
        reasoning = ""
        details = {}

        try:
            if self.mode in [MarketFitMode.OASIS_ONLY, MarketFitMode.HYBRID_DUAL]:
                logger.info("Running Actual OASIS Simulation (Primary Path)")
                try:
                    # 1. Initialize agents from personas (mapping to UserInfo)
                    from tsc.oasis.profile_builder import InitializeOASISAgents
                    agent_profiles, _ = await InitializeOASISAgents(
                        personas=personas,
                        feature=feature,
                        context=company,
                        config=self.config,
                        kg=graph,
                        market_context=bundle.market_context # MiroFish: Ground agents in market context
                    )

                    # 2. Run Actual OASIS (Process Isolated via SimulationRunner)
                    simulation_id = f"oasis_{int(time.time())}"
                    self.config.simulation_name = simulation_id # Ensure name consistency
                    runner = SimulationRunner(simulation_id)
                    
                    logger.info(f"Spawning OASIS subprocess: {simulation_id}")
                    runner.start_simulation(
                        config=self.config,
                        agent_profiles=agent_profiles,
                        feature=feature,
                        context=company,
                        market_context=bundle.market_context if hasattr(bundle, 'market_context') else None
                    )
                    
                    # Polling with Async Sleep (Main process remains responsive)
                    sentiment_series = None
                    while True:
                        status = runner.get_status()
                        if status["status"] == "COMPLETED":
                            sentiment_series = runner.get_result()
                            break
                        elif status["status"] == "FAILED":
                            logger.error(f"OASIS Simulation Failed: {status.get('error')}")
                            raise Exception(f"OASIS Subprocess Error: {status.get('error')}")
                        
                        await asyncio.sleep(2) # Poll frequency
                    
                    # 3. Behavioral Analysis & Consensus Detection
                    clusters = await PerformBehavioralClustering(agent_profiles, sentiment_series)
                    sentiment_series.belief_clusters = clusters
                    
                    is_consensus, strength, consensus_type = DetectConsensus(sentiment_series, self.config)
                    sentiment_series.consensus_strength = strength
                    sentiment_series.consensus_type = consensus_type
                    sentiment_series.convergence_reached = is_consensus
                    
                    oasis_result = {
                        "sentiment": sentiment_series.consensus_verdict,
                        "adoption_score": sentiment_series.final_adoption_score,
                        "consensus_strength": strength,
                        "consensus_type": consensus_type,
                        "segment_count": len(clusters),
                        "objections": sentiment_series.key_objections,
                        "status": "SUCCESS"
                    }
                    
                    # Score derived from adoption score and consensus
                    score = (strength * 0.3) + (sentiment_series.final_adoption_score * 0.7)
                    verdict = GateVerdict.STRONG_FIT if score > 0.75 else GateVerdict.PASS
                    if score < 0.4: verdict = GateVerdict.RISKY
                    
                    reasoning = f"Actual OASIS simulation reached {consensus_type} consensus with {sentiment_series.final_adoption_score*100:.1f}% positive market sentiment. Key objections: {', '.join(sentiment_series.key_objections[:3]) if sentiment_series.key_objections else 'None'}."
                    
                except Exception as e:
                    logger.error(f"OASIS Simulation failed: {e}", exc_info=True)
                    if self.mode == MarketFitMode.OASIS_ONLY:
                        raise
                    logger.warning("Falling back to legacy Mesa simulation...")

            if (self.mode == MarketFitMode.MESA_ONLY) or \
               (self.mode == MarketFitMode.HYBRID_DUAL and oasis_result is None):
                logger.info("Running Mesa Simulation (Fallback Path)")
                mesa_raw = await RunMesaSimulation(feature, personas, company, self.num_agents)
                mesa_result = {
                    "adoption_rate": mesa_raw.final_adoption_rate,
                    "convergence": mesa_raw.convergence_steps,
                    "status": "SUCCESS"
                }
                
                score = mesa_raw.final_adoption_rate
                verdict = GateVerdict.PASS if score > 0.5 else GateVerdict.RISKY
                if not reasoning:
                    reasoning = f"Fallback Mesa simulation predicted {score*100:.1f}% adoption rate."

            # Consolidate results
            details = {
                "oasis": oasis_result,
                "mesa": mesa_result,
                "mode_used": self.mode.value,
                "execution_time": time.time() - t0
            }

            return GateResult(
                gate_id=self.gate_id,
                gate_name=self.gate_name,
                verdict=verdict,
                score=min(1.0, max(0.0, score)),
                details=details,
                reasoning=reasoning
            )

        except Exception as e:
            logger.error(f"MarketFitGate critical failure: {e}")
            return GateResult(
                gate_id=self.gate_id,
                gate_name=self.gate_name,
                verdict=GateVerdict.RISKY,
                score=0.2,
                details={"error": str(e)},
                reasoning=f"Market Fit evaluation failed: {e}"
            )

    def _build_context(self, *args, **kwargs) -> str: return ""
    def _build_questions(self, *args, **kwargs) -> str: return ""
