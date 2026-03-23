"""Gate verdict and result models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class GateVerdict(str, Enum):
    PASS = "PASS"
    PASS_WITH_MITIGATION = "PASS_WITH_MITIGATION"
    FEASIBLE_WITH_DEBT = "FEASIBLE_WITH_DEBT"
    EXISTS_NEEDS_ADAPTATION = "EXISTS_NEEDS_ADAPTATION"
    STRONG_FIT = "STRONG_FIT"
    MANAGEABLE = "MANAGEABLE"
    RISKY = "RISKY"
    FAIL = "FAIL"


class RiskEntry(BaseModel):
    """A single identified risk with mitigation."""

    risk_category: str
    description: str
    probability: float = 0.0
    impact: str = ""
    weighted_score: float = 0.0
    mitigation: str = ""


class GateResult(BaseModel):
    """Output of a single gate evaluation."""

    gate_id: str
    gate_name: str
    verdict: GateVerdict
    score: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)
    risks: list[RiskEntry] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evidence_chunks: list[str] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MonteCarloResults(BaseModel):
    """Results from Gate 4.5 Monte Carlo market simulation."""

    simulations: int = 3000
    adoption_mean: float = 0.0
    adoption_std_dev: float = 0.0
    adoption_p25: float = 0.0
    adoption_p75: float = 0.0
    revenue_mean: str = ""
    revenue_min: str = ""
    revenue_max: str = ""
    cost_to_build: str = ""
    mean_roi: str = ""
    adoption_probabilities: dict[str, float] = Field(default_factory=dict)
    persona_breakdown: list[dict[str, Any]] = Field(default_factory=list)


class GatesSummary(BaseModel):
    """Summary of all 8 gate results."""

    results: list[GateResult] = Field(default_factory=list)
    overall_score: float = 0.0
    all_passed: bool = False
    failed_gates: list[str] = Field(default_factory=list)
    passed_gates: list[str] = Field(default_factory=list)
    needs_refinement: bool = False
    recommendation: str = ""
    recommendation_reason: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="allow")  # allows _diagnostics injection

    @property
    def gate_count(self) -> int:
        return len(self.results)

    def get_gate(self, gate_id: str) -> Optional[GateResult]:
        for r in self.results:
            if r.gate_id == gate_id:
                return r
        return None
