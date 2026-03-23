"""Layer 8: Cursor Handoff & Monitoring.

Packages the final recommendation with monitoring framework and next steps.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from tsc.llm.base import LLMClient
from tsc.llm.prompts import SUMMARY_SYSTEM, SUMMARY_USER
from tsc.models.debate import ConsensusResult
from tsc.models.gates import GatesSummary
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.personas import FinalPersona
from tsc.models.recommendation import (
    EvaluationMetadata,
    FinalRecommendation,
    MonitoringFramework,
    MonitoringMetrics,
    NextStep,
    PillarVerdict,
)
from tsc.models.spec import FeatureSpecification

logger = logging.getLogger(__name__)


class HandoffGenerator:
    """Layer 8: Package final recommendation."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    async def process(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        personas: list[FinalPersona],
        gates_summary: GatesSummary,
        consensus: ConsensusResult,
        spec: FeatureSpecification,
        start_time: float,
    ) -> FinalRecommendation:
        """Generate the final recommendation."""
        t0 = time.time()
        logger.info("Layer 8: Generating final recommendation")

        # Build verdicts by pillar
        verdicts = self._build_pillar_verdicts(gates_summary, consensus)

        # Build monitoring framework
        monitoring = self._build_monitoring(feature)

        # Build next steps
        next_steps = self._build_next_steps(consensus)

        # Generate leadership summary
        summary = await self._generate_summary(
            feature, consensus, gates_summary
        )

        # Top risks (from gates)
        top_risks = []
        for gate in gates_summary.results:
            top_risks.extend(gate.risks)
        top_risks = sorted(
            top_risks, key=lambda r: r.probability, reverse=True
        )[:5]

        total_minutes = (time.time() - start_time) / 60

        recommendation = FinalRecommendation(
            feature_name=feature.title,
            evaluation_date=datetime.utcnow().strftime("%Y-%m-%d"),
            final_verdict=consensus.overall_verdict,
            overall_confidence=consensus.approval_confidence,
            verdicts_by_pillar=verdicts,
            phase_1=consensus.phase_1,
            success_criteria=consensus.success_criteria,
            phase_2_gate=consensus.phase_2_gate,
            top_risks=top_risks,
            stakeholder_approvals=consensus.approvals,
            development_tasks=spec.development_tasks,
            monitoring=monitoring,
            next_steps=next_steps,
            specification=spec,
            summary_for_leadership=summary,
            metadata=EvaluationMetadata(
                total_time_minutes=round(total_minutes, 1),
                confidence_calculation=(
                    f"Gates: {gates_summary.overall_score:.2f} × 0.6 + "
                    f"Consensus: {consensus.approval_confidence:.2f} × 0.4 = "
                    f"{consensus.approval_confidence:.2f}"
                ),
                llm_provider=self._llm.__class__.__name__,
                llm_model=self._llm.model,
                total_tokens_used=self._llm.get_usage().total_tokens,
            ),
        )

        logger.info(
            "Layer 8 complete: %s (confidence: %.2f, %.1fs)",
            recommendation.final_verdict,
            recommendation.overall_confidence,
            time.time() - t0,
        )
        return recommendation

    def _build_pillar_verdicts(
        self, gates: GatesSummary, consensus: ConsensusResult
    ) -> dict[str, PillarVerdict]:
        # Map gates to pillars
        technical_gates = [g for g in gates.results if g.gate_id in ("4.1", "4.2", "4.3", "4.4")]
        market_gates = [g for g in gates.results if g.gate_id == "4.5"]
        risk_gates = [g for g in gates.results if g.gate_id == "4.6"]
        exec_gates = [g for g in gates.results if g.gate_id in ("4.7", "4.8")]

        def avg_score(gs):
            return sum(g.score for g in gs) / max(len(gs), 1)

        return {
            "technical": PillarVerdict(
                verdict=technical_gates[0].verdict.value if technical_gates else "UNKNOWN",
                score=round(avg_score(technical_gates), 2),
                rationale=", ".join(g.gate_name for g in technical_gates),
            ),
            "market": PillarVerdict(
                verdict=market_gates[0].verdict.value if market_gates else "UNKNOWN",
                score=round(avg_score(market_gates), 2),
                rationale="Market fit based on Monte Carlo simulation",
            ),
            "internal_stakeholder": PillarVerdict(
                verdict="CONSENSUS_REACHED" if consensus.overall_verdict != "REJECTED" else "NO_CONSENSUS",
                score=consensus.approval_confidence,
                rationale=f"{len(consensus.approvals)} stakeholders participated",
            ),
            "risk_assessment": PillarVerdict(
                verdict=risk_gates[0].verdict.value if risk_gates else "UNKNOWN",
                score=round(avg_score(risk_gates), 2),
                rationale="Red-team risk analysis",
            ),
        }

    def _build_monitoring(self, feature: FeatureProposal) -> MonitoringFramework:
        return MonitoringFramework(
            metrics=MonitoringMetrics(
                real_time=[
                    "System health (error rate, latency)",
                    "Feature usage / engagement rate",
                    "Performance impact (p95 latency)",
                ],
                weekly=[
                    "Adoption rate (% of target users)",
                    "Support ticket volume (feature-related)",
                    "User feedback sentiment",
                ],
                biweekly=[
                    "NPS / satisfaction trend",
                    "Feature engagement depth",
                    "Business metric impact",
                ],
            ),
            gates_and_checkpoints={
                "week_1": "Design review, technical spike complete",
                "week_2": "Core implementation working, integration started",
                "week_4": "Feature complete, QA sign-off, launch readiness",
                "week_6": "Adoption measurement, Phase 2 go/no-go",
            },
            escalation_triggers={
                "critical_incident": "Immediate: stop feature, investigate, fix",
                "adoption_below_threshold": "Week 6: Phase 2 NOT approved, UX review",
                "performance_degradation": "Immediate: investigate, optimize or rollback",
            },
            success_definition={
                "full_success": "All criteria met + adoption target achieved",
                "partial_success": "Most criteria met, conditional Phase 2",
                "failure": "Below adoption threshold, post-mortem required",
            },
        )

    def _build_next_steps(self, consensus: ConsensusResult) -> list[NextStep]:
        steps = [
            NextStep(step=1, action="Engineering kickoff meeting", owner="Engineering Lead", timeline="Day 1"),
            NextStep(step=2, action="Technical design review", owner="Engineering Lead", timeline="Day 2"),
            NextStep(step=3, action="Development sprint start", owner="Team", timeline="Week 1"),
            NextStep(step=4, action="Week 1 gate review", owner="Product Manager", timeline="End of Week 1"),
            NextStep(step=5, action="Development + QA", owner="Team", timeline="Weeks 2-4"),
            NextStep(step=6, action="Beta launch", owner="Product", timeline="Week 4"),
            NextStep(step=7, action="Adoption measurement", owner="Analytics", timeline="Week 6"),
            NextStep(step=8, action="Phase 2 gate decision", owner="Finance Lead", timeline="Week 6"),
        ]
        return steps

    async def _generate_summary(
        self,
        feature: FeatureProposal,
        consensus: ConsensusResult,
        gates: GatesSummary,
    ) -> str:
        try:
            prompt = SUMMARY_USER.render(
                feature=feature,
                verdict=consensus.overall_verdict,
                confidence=consensus.approval_confidence,
                key_metrics=f"{gates.overall_score:.0%} gate pass rate",
                roi="Based on adoption forecasts",
                timeline=consensus.phase_1.timeline or "TBD",
                top_risk=consensus.mitigations[0] if consensus.mitigations else "None identified",
            )
            return await self._llm.generate(
                system_prompt=SUMMARY_SYSTEM,
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=200,
            )
        except Exception as e:
            logger.warning("Summary generation failed: %s", e)
            return (
                f"{feature.title} has been evaluated with {consensus.overall_verdict} verdict "
                f"(confidence: {consensus.approval_confidence:.0%}). "
                f"Gate pass rate: {gates.overall_score:.0%}."
            )
