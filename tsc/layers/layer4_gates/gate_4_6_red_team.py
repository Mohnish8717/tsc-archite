import asyncio
import logging
import time
from typing import List, Optional, Any, Dict

from tsc.gates.base import BaseGate
from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.models.personas import FinalPersona
from tsc.models.graph import KnowledgeGraph
from tsc.models.chunks import ProblemContextBundle
from tsc.models.gates import GateResult, GateVerdict

# Legacy Fallback
from tsc.mesa.red_team import RunRedTeamSimulation

logger = logging.getLogger(__name__)

class RedTeamGate(BaseGate):
    """
    Gate 4.6: Red-Team Adversarial Analysis (Hybrid)
    Uses OASIS insights for stakeholder-specific risks + Mesa for cascading failures.
    """
    gate_id = "4.6"
    gate_name = "Red-Team Adversarial Analysis"
    gate_domain = "security, safety, and adversarial impact"

    async def evaluate(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: List[FinalPersona]
    ) -> GateResult:
        """Execute adversarial analysis."""
        t0 = time.time()
        logger.info("Executing Red Team Analysis")

        # 1. Primary Path: LLM-based Red Teaming targeted at OASIS clusters
        # (For this MVP, we use the legacy Mesa model as the primary heavy-lifter)
        # but we could add more targeted logic here.
        
        try:
            # We use the legacy implementation which is already quite robust
            # in terms of cascading risk logic.
            result = await RunRedTeamSimulation(feature, company, personas, graph, bundle)
            
            # Record execution time correctly
            result.execution_time_seconds = round(time.time() - t0, 1)
            
            return result

        except Exception as e:
            logger.error(f"Red Team analysis failed: {e}")
            return GateResult(
                gate_id=self.gate_id,
                gate_name=self.gate_name,
                verdict=GateVerdict.RISKY,
                score=0.2,
                reasoning=f"Red Team analysis failed: {e}"
            )

    def _build_context(self, *args, **kwargs) -> str: return ""
    def _build_questions(self, *args, **kwargs) -> str: return ""
