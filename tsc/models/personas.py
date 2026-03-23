"""Stakeholder persona models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class EmotionalTriggers(BaseModel):
    excited_by: list[str] = Field(default_factory=list)
    frustrated_by: list[str] = Field(default_factory=list)
    scared_of: list[str] = Field(default_factory=list)


class CommunicationStyle(BaseModel):
    default: str = "Direct"
    formality: str = "Semi-formal"
    conflict_handling: str = "Pragmatic"
    preferred_channels: list[str] = Field(default_factory=list)


class DecisionPattern(BaseModel):
    speed: str = "Moderate"
    preference: str = "Data-driven"
    influencers: list[str] = Field(default_factory=list)
    justification: str = ""
    risk_tolerance: str = "Medium"


class PredictedStance(BaseModel):
    feature: str = ""
    prediction: str = ""  # APPROVE, REJECT, CONDITIONAL_APPROVE
    confidence: float = 0.0
    likely_conditions: list[str] = Field(default_factory=list)
    potential_objections: list[str] = Field(default_factory=list)


class PsychologicalProfile(BaseModel):
    """Detailed psychological profile of a stakeholder."""

    mbti: str = ""
    mbti_description: str = ""
    key_traits: list[str] = Field(default_factory=list)
    emotional_triggers: EmotionalTriggers = Field(default_factory=EmotionalTriggers)
    communication_style: CommunicationStyle = Field(
        default_factory=CommunicationStyle
    )
    decision_pattern: DecisionPattern = Field(default_factory=DecisionPattern)
    predicted_stance: PredictedStance = Field(default_factory=PredictedStance)
    questions_they_will_ask: list[str] = Field(default_factory=list)
    full_profile_text: str = ""  # The complete ~2000-word narrative


class Stakeholder(BaseModel):
    """A selected stakeholder before profiling."""

    name: str
    role: str
    title: str = ""
    relevance_score: float = 0.0
    domain_relevance: str = ""
    decision_authority: str = "medium"
    persona_type: str = "INTERNAL"


class StakeholderContextBundle(BaseModel):
    """All context retrieved from memory for a given stakeholder."""

    stakeholder: Stakeholder
    personal_facts: list[Any] = Field(default_factory=list)
    past_decisions: list[dict[str, str]] = Field(default_factory=list)
    org_context: list[str] = Field(default_factory=list)
    constraint_context: list[str] = Field(default_factory=list)
    relevance_summary: dict[str, float] = Field(default_factory=dict)


class FinalPersona(BaseModel):
    """Complete persona with profile and evidence."""

    name: str
    role: str
    psychological_profile: PsychologicalProfile = Field(
        default_factory=PsychologicalProfile
    )
    evidence_sources: list[str] = Field(default_factory=list)
    profile_timestamp: datetime = Field(default_factory=datetime.utcnow)
    profile_word_count: int = 0
    profile_confidence: float = 0.0
    grounding_quality: float = 1.0
    persona_type: str = "INTERNAL"
