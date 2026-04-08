"""Stakeholder persona models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

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


# ─────────────────────────────────────────────────────────────────────
# Market / Buyer Context (EXTERNAL personas only)
# Domain-agnostic — works for any business, product, or strategic feature.
# ─────────────────────────────────────────────────────────────────────

class MarketContext(BaseModel):
    """
    Economic and organisational attributes of an external buyer/customer.
    Fully domain-agnostic: applies to SaaS tools, physical products, policy
    changes, service launches, B2B/B2C market entries, etc.
    """
    company_size_band: str = "mid-market"
    # micro (<10 employees) | small (10–99) | mid-market (100–999) | enterprise (1000+)

    buyer_role: str = "influencer"
    # decision-maker | influencer | end-user | champion

    annual_solution_budget_usd: int = 50_000
    # Rough annual spend authority for solutions in this category

    pricing_sensitivity: str = "medium"
    # high (budget-drives everything) | medium | low (value-drives)

    sales_cycle_weeks: int = 8
    # Typical weeks from first contact to signed agreement

    deployment_preference: str = "cloud"
    # cloud | on-prem | hybrid | managed-service | physical | policy-mandate

    industry_vertical: str = "technology"
    # finance | healthcare | retail | technology | government |
    # manufacturing | education | real-estate | media | other

    regulatory_burden: str = "light"
    # none | light | moderate | heavy


class BuyerJourney(BaseModel):
    """
    How this persona moves from problem awareness through to post-adoption.
    Fully domain-agnostic: maps to any buying process regardless of product type.
    """
    awareness_channel: str = "peer-recommendation"
    # peer-recommendation | analyst-report | vendor-outreach |
    # organic-search | internal-mandate | conference | social-proof

    evaluation_trigger: str = ""
    # The specific business event or pain that forces them to seek a solution

    key_proof_points: list[str] = Field(default_factory=list)
    # What they need to see/hear before they believe the solution works

    deal_breakers: list[str] = Field(default_factory=list)
    # Conditions that stop the evaluation immediately

    success_metric: str = ""
    # How they would quantify success after adoption

    roi_threshold_months: int = 12
    # Payback period they consider acceptable (shorter = more urgent buyer)

    willingness_to_pay_band: str = "moderate"
    # low | moderate | high | very-high (relative to solution list price)


# ─────────────────────────────────────────────────────────────────────
# Stakeholder Models
# ─────────────────────────────────────────────────────────────────────

class Stakeholder(BaseModel):
    """A selected stakeholder before profiling."""

    name: str
    role: str
    title: str = ""
    relevance_score: float = 0.0
    domain_relevance: str = ""
    decision_authority: str = "medium"
    persona_type: str = "INTERNAL"

    # Economic metadata (populated for EXTERNAL stakeholders from LLM selection)
    market_metadata: dict[str, Any] = Field(default_factory=dict)


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

    # OASIS-specific fields
    belief_vector: Optional[Any] = None  # OpinionVector
    network_position_hint: str = "peripheral"
    influence_strength: float = 0.5
    receptiveness: float = 0.5

    # Market/Buyer context — populated only for EXTERNAL personas
    market_context: Optional[MarketContext] = None
    buyer_journey: Optional[BuyerJourney] = None

    def to_oasis_profile(self, feature: Any) -> Any:
        """Convert persona to OASIS simulation agent."""
        from tsc.oasis.models import OASISAgentProfile
        from tsc.oasis.profile_builder import BuildBeliefVector
        import asyncio

        # Synchronous wrapper — actual build happens in Layer 3 process()
        return None

