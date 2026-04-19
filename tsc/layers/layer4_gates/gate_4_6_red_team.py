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

from enum import Enum

class RedTeamMode(Enum):
    OASIS_ONLY = "oasis_only"
    MESA_ONLY = "mesa_only"
    HYBRID = "hybrid"

logger = logging.getLogger(__name__)

class RedTeamGate(BaseGate):
    """
    Gate 4.6: Red-Team Adversarial Analysis (Hybrid/OASIS)
    Uses OASIS insights for stakeholder-specific risks + Mesa for cascading failures.
    """
    gate_id = "4.6"
    gate_name = "Red-Team Adversarial Analysis"
    gate_domain = "security, safety, and adversarial impact"

    def __init__(
        self,
        llm_client: Any,
        graph_store: Optional[Any] = None,
        mode: RedTeamMode = RedTeamMode.OASIS_ONLY,
    ):
        super().__init__(llm_client)
        self._graph_store = graph_store
        self._mode = mode

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
        logger.info(f"Executing Red Team Analysis (Mode: {self._mode.value})")

        if self._mode == RedTeamMode.OASIS_ONLY:
            return await self._evaluate_oasis(feature, company, graph, bundle, personas, t0)
        
        try:
            # Legacy Mesa path
            result = await RunRedTeamSimulation(feature, company, personas, graph, bundle)
            result.execution_time_seconds = float(round(time.time() - t0, 1))
            return result

        except Exception as e:
            logger.error(f"Mesa Red Team analysis failed: {e}")
            return await self._evaluate_oasis(feature, company, graph, bundle, personas, t0)

    async def _evaluate_oasis(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: List[FinalPersona],
        start_time: float
    ) -> GateResult:
        """Pure OASIS/LLM adversarial analysis."""
        logger.info("Running OASIS-based Red Team Analysis")
        
        # Build adversarial context
        context = self._build_adversarial_context(feature, company, graph, personas)
        
        prompt = f"""
        {context}

        As a Red Team lead, perform a deep adversarial analysis of this feature.
        Identify the top 3 FATAL RISKS (Technical, Market, or Adoption).
        For each risk, calculate probability (0.0-1.0) and impact (high/medium/low).
        
        Focus on:
        1. Cascading failures (how one failure triggers others).
        2. Corner cases that typical testing misses.
        3. Strategic "poisoning" of the user base (bad actors).

        Return a summary that includes a recommended verdict and an overall score (0.0-10.0).
        High risk = Low score.
        """

        try:
            response = await self._llm.analyze(
                system_prompt="You are a senior red-team security and market strategist. Your goal is to find why this WILL FAIL.",
                user_prompt=prompt,
                temperature=0.7
            )
            
            # Extract score and verdict using a helper or basic parsing
            score = 7.5 # Fallback
            if "score" in response.lower():
                import re
                match = re.search(r"score:\s*([\d\.]+)", response.lower())
                if match: score = float(match.group(1))

            verdict = GateVerdict.PASS_WITH_MITIGATION
            if score < 5.0: verdict = GateVerdict.RISKY
            if score < 3.0: verdict = GateVerdict.FAIL

            return GateResult(
                gate_id=self.gate_id,
                gate_name=self.gate_name,
                verdict=verdict,
                score=score,
                reasoning=response,
                execution_time_seconds=float(round(time.time() - start_time, 1))
            )
        except Exception as e:
            return GateResult(
                gate_id=self.gate_id,
                gate_name=self.gate_name,
                verdict=GateVerdict.RISKY,
                score=0.2,
                reasoning=f"OASIS Red Team analysis failed: {e}"
            )

    def _build_adversarial_context(self, feature, company, graph, personas) -> str:
        return f"""
        FEATURE: {feature.title} - {feature.description}
        COMPANY: {company.company_name}
        STAKEHOLDERS: {len(personas)} distinct personas represented.
        GRAPH NODES: {len(graph.nodes)} technical/business entities.
        """

    def _build_questions(self, *args, **kwargs) -> str: return ""
