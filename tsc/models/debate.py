"""Debate and consensus models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DebatePosition(BaseModel):
    """A stakeholder's position in one debate round."""

    stakeholder_name: str
    role: str
    statement: str
    verdict: str = ""  # APPROVE, REJECT, CONDITIONAL_APPROVE
    confidence: float = 0.0
    key_concerns: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)


class DebateRound(BaseModel):
    """A single round of the stakeholder debate."""

    round_number: int
    round_name: str  # e.g. "Initial Positions", "Negotiation", "Final Consensus"
    positions: list[DebatePosition] = Field(default_factory=list)
    synthesis: str = ""  # Summary of the round
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StakeholderApproval(BaseModel):
    """Final approval record from one stakeholder."""

    stakeholder: str
    role: str
    verdict: str
    confidence: float = 0.0
    key_concerns: str = ""
    conditions: list[str] = Field(default_factory=list)
    conditions_accepted: list[str] = Field(default_factory=list)


class PhaseSpecification(BaseModel):
    """Specification for a single phase (e.g., Phase 1 MVP)."""

    name: str = ""
    timeline: str = ""
    scope: list[str] = Field(default_factory=list)
    deferred: list[str] = Field(default_factory=list)
    cost_estimate: str = ""
    team_allocation: str = ""


class SuccessCriteria(BaseModel):
    """Measurable success criteria for the feature."""

    criteria: dict[str, Any] = Field(default_factory=dict)


class PhaseGate(BaseModel):
    """Gate condition for proceeding to next phase."""

    condition: str = ""
    measurement_date: str = ""
    phase_scope: list[str] = Field(default_factory=list)
    estimated_timeline: str = ""
    estimated_cost: str = ""


class ConsensusResult(BaseModel):
    """Output of the full debate process."""

    feature_name: str
    overall_verdict: str  # APPROVED, REJECTED, CONDITIONAL
    approval_confidence: float = 0.0
    stakeholder_verdicts: dict[str, str] = Field(default_factory=dict)
    approvals: list[StakeholderApproval] = Field(default_factory=list)

    phase_1: PhaseSpecification = Field(default_factory=PhaseSpecification)
    success_criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)
    phase_2_gate: Optional[PhaseGate] = None

    mitigations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    tension_shifts: dict[str, float] = Field(default_factory=dict)
    overall_summary: str = ""

    debate_rounds: list[DebateRound] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Fix forward ref
from typing import Optional  # noqa: E402
