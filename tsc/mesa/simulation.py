import asyncio
import logging
from typing import List, Optional, Any
from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.models.personas import FinalPersona
from tsc.mesa.market_fit_gate_legacy import MonteCarloMarketFitGate

logger = logging.getLogger(__name__)

async def RunMesaSimulation(
    proposal: FeatureProposal,
    personas: List[FinalPersona],
    context: CompanyContext,
    num_agents: int = 300,
    timesteps: int = 12
) -> Any:
    """
    Wrapper for the legacy Mesa simulation logic.
    """
    logger.info(f"Running legacy Mesa simulation for {proposal.title}")
    
    # We instantiate the legacy gate. 
    # Note: It needs an LLMClient, but for a pure simulation fallback 
    # we might need to pass one or mock it if only internal logic is used.
    # In the original gate, it uses the LLM for 'inject_llm_baselines'.
    
    # For now, we'll assume the caller provides an LLM client or we use a global one.
    # Since this is a wrapper, let's assume we need the LLM.
    from tsc.llm.groq_provider import GroqLLMClient # Default
    import os
    
    llm = GroqLLMClient(api_key=os.environ.get("GROQ_API_KEY", ""))
    
    gate = MonteCarloMarketFitGate(
        llm_client=llm,
        num_agents=num_agents
    )
    
    # The legacy gate evaluation returns a GateResult
    # We'll need a ProblemContextBundle and KnowledgeGraph too as per signature
    # but the legacy gate barely uses them for the simulation loop except for context building.
    
    from tsc.models.chunks import ProblemContextBundle
    from tsc.models.graph import KnowledgeGraph
    
    dummy_bundle = ProblemContextBundle(chunks=[])
    dummy_graph = KnowledgeGraph(nodes={}, edges={})
    
    result = await gate.evaluate(
        feature=proposal,
        company=context,
        graph=dummy_graph,
        bundle=dummy_bundle,
        personas=personas
    )
    
    # Wrap result in a simple object that the new gate expects
    class MesaResultWrapper:
        def __init__(self, res):
            self.final_adoption_rate = res.details.get("adoption_rate", 0.0)
            self.convergence_steps = res.details.get("simulations", 12) # Approximation
            
    return MesaResultWrapper(result)
