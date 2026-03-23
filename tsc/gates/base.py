"""Base gate class — all 8 gates inherit from this."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from tsc.llm.base import LLMClient
from tsc.llm.prompts import GATE_SYSTEM, GATE_USER
from tsc.models.chunks import ProblemContextBundle
from tsc.models.gates import GateResult, GateVerdict, RiskEntry
from tsc.models.graph import GraphEntity, KnowledgeGraph
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.personas import FinalPersona

logger = logging.getLogger(__name__)


class BaseGate(ABC):
    """Abstract base class for evaluation gates."""

    gate_id: str = ""
    gate_name: str = ""
    gate_domain: str = ""
    verdict_options: str = "PASS, PASS_WITH_MITIGATION, FEASIBLE_WITH_DEBT, RISKY, FAIL"

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    async def evaluate(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: list[FinalPersona],
    ) -> GateResult:
        """Run the gate evaluation."""
        t0 = time.time()
        logger.info("Gate %s: %s — starting", self.gate_id, self.gate_name)

        top_entities = sorted(
            graph.nodes.values(), key=lambda e: e.mentions, reverse=True
        )[:15]

        # Build gate-specific context
        gate_context = self._build_context(feature, company, graph, bundle, personas)
        questions = self._build_questions()

        prompt = GATE_USER.render(
            feature=feature,
            company=company,
            top_entities=[
                {
                    "name": e.name,
                    "type": e.type,
                    "mentions": e.mentions,
                    "average_urgency": e.average_urgency,
                }
                for e in top_entities
            ],
            gate_specific_context=gate_context,
            gate_questions=questions,
            verdict_options=self.verdict_options,
            gate_id=self.gate_id,
            gate_name=self.gate_name,
        )

        system = GATE_SYSTEM.render(gate_domain=self.gate_domain)

        result_data = await self._llm.analyze(
            system_prompt=system,
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=3000,
        )

        # Parse into GateResult
        result = self._parse_result(result_data, time.time() - t0)
        logger.info(
            "Gate %s: %s — %s (score: %.2f, %.1fs)",
            self.gate_id,
            self.gate_name,
            result.verdict.value,
            result.score,
            result.execution_time_seconds,
        )
        return result

    @abstractmethod
    def _build_context(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
        personas: list[FinalPersona],
    ) -> str:
        """Build gate-specific context string."""
        ...

    @abstractmethod
    def _build_questions(self) -> str:
        """Build questions for this gate to answer."""
        ...

    def _parse_result(self, data: dict[str, Any], elapsed: float) -> GateResult:
        """Parse LLM response into GateResult."""
        verdict_str = data.get("verdict", "PASS")
        try:
            verdict = GateVerdict(verdict_str)
        except ValueError:
            # Try mapping common variants
            mapping = {
                "FEASIBLE": GateVerdict.PASS,
                "GO": GateVerdict.PASS,
                "NO_GO": GateVerdict.FAIL,
                "STRONG_FIT": GateVerdict.STRONG_FIT,
                "EXISTS_NEEDS_ADAPTATION": GateVerdict.EXISTS_NEEDS_ADAPTATION,
                "MANAGEABLE": GateVerdict.MANAGEABLE,
            }
            verdict = mapping.get(verdict_str, GateVerdict.PASS_WITH_MITIGATION)

        risks = [
            RiskEntry(
                risk_category=r.get("risk_category", ""),
                description=r.get("description", ""),
                probability=r.get("probability", 0.0),
                impact=str(r.get("impact", "")),
                mitigation=r.get("mitigation", ""),
            )
            for r in data.get("risks", [])
        ]

        return GateResult(
            gate_id=data.get("gate_id", self.gate_id),
            gate_name=data.get("gate_name", self.gate_name),
            verdict=verdict,
            score=min(1.0, max(0.0, data.get("score", 0.5))),
            details=data.get("details", {}),
            risks=risks,
            recommendations=data.get("recommendations", []),
            execution_time_seconds=round(elapsed, 1),
        )
