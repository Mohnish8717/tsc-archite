"""Final recommendation output model — the ultimate pipeline output."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from tsc.models.debate import (
    ConsensusResult,
    PhaseGate,
    PhaseSpecification,
    StakeholderApproval,
    SuccessCriteria,
)
from tsc.models.gates import GateResult, RiskEntry
from tsc.models.spec import DevelopmentTask, FeatureSpecification


class PillarVerdict(BaseModel):
    """Verdict for one evaluation pillar."""

    verdict: str
    score: float = 0.0
    rationale: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class MonitoringMetrics(BaseModel):
    real_time: list[str] = Field(default_factory=list)
    weekly: list[str] = Field(default_factory=list)
    biweekly: list[str] = Field(default_factory=list)


class MonitoringFramework(BaseModel):
    metrics: MonitoringMetrics = Field(default_factory=MonitoringMetrics)
    gates_and_checkpoints: dict[str, str] = Field(default_factory=dict)
    escalation_triggers: dict[str, str] = Field(default_factory=dict)
    success_definition: dict[str, str] = Field(default_factory=dict)


class NextStep(BaseModel):
    step: int
    action: str
    owner: str = ""
    timeline: str = ""


class EvaluationMetadata(BaseModel):
    evaluated_by: str = "TSC v2.0 Evaluation Pipeline"
    layers_executed: int = 8
    total_time_minutes: float = 0.0
    confidence_calculation: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    llm_provider: str = ""
    llm_model: str = ""
    total_tokens_used: int = 0


class FinalRecommendation(BaseModel):
    """The complete output of the TSC v2.0 pipeline."""

    # Core verdict
    feature_name: str
    evaluation_date: str = ""
    final_verdict: str = ""  # APPROVED, REJECTED, CONDITIONAL
    overall_confidence: float = 0.0

    # Verdicts by pillar
    verdicts_by_pillar: dict[str, PillarVerdict] = Field(default_factory=dict)

    # Phase specifications
    phase_1: PhaseSpecification = Field(default_factory=PhaseSpecification)
    success_criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)
    phase_2_gate: Optional[PhaseGate] = None

    # Top risks
    top_risks: list[RiskEntry] = Field(default_factory=list)

    # Stakeholder approvals
    stakeholder_approvals: list[StakeholderApproval] = Field(default_factory=list)

    # Development tasks
    development_tasks: list[DevelopmentTask] = Field(default_factory=list)

    # Monitoring
    monitoring: MonitoringFramework = Field(default_factory=MonitoringFramework)

    # Next steps
    next_steps: list[NextStep] = Field(default_factory=list)

    # Full specification document
    specification: Optional[FeatureSpecification] = None

    # Summary
    summary_for_leadership: str = ""

    @property
    def executive_summary(self) -> str:
        """Alias for summary_for_leadership."""
        return self.summary_for_leadership

    # Metadata
    metadata: EvaluationMetadata = Field(default_factory=EvaluationMetadata)
