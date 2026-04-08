"""Unit tests for domain-agnostic market persona enhancements.

Tests cover:
1. MarketContext and BuyerJourney model construction
2. Optional fields default to None on FinalPersona
3. _jitter_clone produces market frames for EXTERNAL personas
4. _jitter_clone produces internal frames for INTERNAL personas (backward compat)
5. UserInfoAdapter includes market fields in other_info for EXTERNAL personas
"""
import sys
from pathlib import Path
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── 1. Model construction ────────────────────────────────────────────────────

def test_market_context_defaults():
    from tsc.models.personas import MarketContext
    mc = MarketContext()
    assert mc.company_size_band == "mid-market"
    assert mc.buyer_role == "influencer"
    assert mc.pricing_sensitivity == "medium"
    assert mc.sales_cycle_weeks == 8
    assert mc.regulatory_burden == "light"


def test_buyer_journey_defaults():
    from tsc.models.personas import BuyerJourney
    bj = BuyerJourney()
    assert bj.awareness_channel == "peer-recommendation"
    assert bj.roi_threshold_months == 12
    assert bj.willingness_to_pay_band == "moderate"
    assert bj.key_proof_points == []
    assert bj.deal_breakers == []


def test_market_context_full_construction():
    from tsc.models.personas import MarketContext
    mc = MarketContext(
        company_size_band="enterprise",
        buyer_role="decision-maker",
        annual_solution_budget_usd=500_000,
        pricing_sensitivity="low",
        sales_cycle_weeks=20,
        deployment_preference="on-prem",
        industry_vertical="healthcare",
        regulatory_burden="heavy",
    )
    assert mc.company_size_band == "enterprise"
    assert mc.buyer_role == "decision-maker"
    assert mc.industry_vertical == "healthcare"
    assert mc.regulatory_burden == "heavy"


def test_buyer_journey_full_construction():
    from tsc.models.personas import BuyerJourney
    bj = BuyerJourney(
        awareness_channel="analyst-report",
        evaluation_trigger="Failed compliance audit",
        key_proof_points=["SOC2 cert", "HIPAA BAA"],
        deal_breakers=["No on-prem option", "Vendor lock-in"],
        success_metric="Zero compliance violations in 12 months",
        roi_threshold_months=18,
        willingness_to_pay_band="very-high",
    )
    assert bj.awareness_channel == "analyst-report"
    assert len(bj.key_proof_points) == 2
    assert bj.roi_threshold_months == 18


# ─── 2. Optional fields on FinalPersona ──────────────────────────────────────

def test_final_persona_market_fields_default_to_none():
    from tsc.models.personas import FinalPersona, PsychologicalProfile
    p = FinalPersona(
        name="Alice",
        role="Enterprise Buyer",
        psychological_profile=PsychologicalProfile(),
    )
    assert p.market_context is None
    assert p.buyer_journey is None


def test_final_persona_market_fields_assignable():
    from tsc.models.personas import FinalPersona, PsychologicalProfile, MarketContext, BuyerJourney
    mc = MarketContext(company_size_band="small", buyer_role="decision-maker", pricing_sensitivity="high")
    bj = BuyerJourney(roi_threshold_months=3, willingness_to_pay_band="low")
    p = FinalPersona(
        name="Bob",
        role="Startup Founder",
        persona_type="EXTERNAL",
        psychological_profile=PsychologicalProfile(),
        market_context=mc,
        buyer_journey=bj,
    )
    assert p.market_context.company_size_band == "small"
    assert p.buyer_journey.roi_threshold_months == 3


# ─── 3. EXTERNAL personas get market frames in _jitter_clone ─────────────────

def test_jitter_clone_external_populates_market_context():
    """EXTERNAL persona clones must have market_context != None after jittering."""
    import copy
    from tsc.models.personas import FinalPersona, PsychologicalProfile
    from tsc.selection.synthetic_expander import GMMSyntheticExpander

    p = FinalPersona(
        name="Ximena Ruiz",
        role="Chief Procurement Officer",
        persona_type="EXTERNAL",
        psychological_profile=PsychologicalProfile(mbti="INTJ"),
    )
    exp = GMMSyntheticExpander(rng_seed=42)
    clone = exp._jitter_clone(p, clone_index=0, sigma=0.3)

    assert clone.market_context is not None, "EXTERNAL clone should have market_context"
    assert clone.buyer_journey is not None, "EXTERNAL clone should have buyer_journey"
    assert clone.market_context.company_size_band in (
        "micro", "small", "mid-market", "enterprise"
    )


# ─── 4. INTERNAL personas get engineering frames (backward compatibility) ─────

def test_jitter_clone_internal_does_not_populate_market_context():
    """INTERNAL personas must NOT receive market_context from _jitter_clone."""
    from tsc.models.personas import FinalPersona, PsychologicalProfile
    from tsc.selection.synthetic_expander import GMMSyntheticExpander

    p = FinalPersona(
        name="Dev User",
        role="Senior Engineer",
        persona_type="INTERNAL",
        psychological_profile=PsychologicalProfile(mbti="INTJ"),
    )
    exp = GMMSyntheticExpander(rng_seed=42)
    clone = exp._jitter_clone(p, clone_index=0, sigma=0.3)

    assert clone.market_context is None, "INTERNAL clone must NOT have market_context"
    assert clone.buyer_journey is None, "INTERNAL clone must NOT have buyer_journey"


# ─── 5. UserInfoAdapter injects market context into other_info ────────────────

def test_user_info_adapter_includes_market_context():
    from tsc.models.personas import FinalPersona, PsychologicalProfile, MarketContext, BuyerJourney
    from tsc.oasis.models import UserInfoAdapter

    mc = MarketContext(
        company_size_band="enterprise",
        buyer_role="decision-maker",
        pricing_sensitivity="low",
        regulatory_burden="heavy",
    )
    bj = BuyerJourney(
        awareness_channel="internal-mandate",
        roi_threshold_months=18,
        willingness_to_pay_band="very-high",
    )
    p = FinalPersona(
        name="Carlos Mendes",
        role="VP Operations",
        persona_type="EXTERNAL",
        psychological_profile=PsychologicalProfile(mbti="ESTJ"),
        market_context=mc,
        buyer_journey=bj,
    )
    user_info = UserInfoAdapter.to_oasis_user_info(p)
    other_info = user_info["profile"]["other_info"]

    assert "market_context" in other_info, "market_context should be in other_info"
    assert "buyer_journey" in other_info, "buyer_journey should be in other_info"
    assert other_info["market_context"]["buyer_role"] == "decision-maker"
    assert other_info["buyer_journey"]["awareness_channel"] == "internal-mandate"


def test_user_info_adapter_no_market_context_for_internal():
    from tsc.models.personas import FinalPersona, PsychologicalProfile
    from tsc.oasis.models import UserInfoAdapter

    p = FinalPersona(
        name="Jane Smith",
        role="CTO",
        persona_type="INTERNAL",
        psychological_profile=PsychologicalProfile(mbti="INTJ"),
    )
    user_info = UserInfoAdapter.to_oasis_user_info(p)
    other_info = user_info["profile"]["other_info"]

    assert "market_context" not in other_info, "INTERNAL personas should not have market_context in other_info"
    assert "buyer_journey" not in other_info, "INTERNAL personas should not have buyer_journey in other_info"
