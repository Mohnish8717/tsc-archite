import asyncio
import logging
from typing import List, Optional, Any, Dict
from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.models.personas import FinalPersona
from tsc.models.graph import KnowledgeGraph
from tsc.models.chunks import ProblemContextBundle
from tsc.mesa.red_team_gate_legacy import RedTeamAdversarialGate

logger = logging.getLogger(__name__)

async def RunRedTeamSimulation(
    proposal: FeatureProposal,
    context: CompanyContext,
    personas: List[FinalPersona],
    graph: KnowledgeGraph,
    bundle: ProblemContextBundle
) -> Any:
    """
    Wrapper for the legacy Mesa Red Team simulation.
    """
    logger.info(f"Running legacy Red Team simulation for {proposal.title}")
    
    from tsc.llm.groq_provider import GroqLLMClient
    import os
    
    llm = GroqLLMClient(api_key=os.environ.get("GROQ_API_KEY", ""))
    
    gate = RedTeamAdversarialGate(llm_client=llm)
    
    result = await gate.evaluate(
        feature=proposal,
        company=context,
        graph=graph,
        bundle=bundle,
        personas=personas
    )
    
    return result
