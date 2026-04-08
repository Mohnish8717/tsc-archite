"""Smoke test for market-aware belief vector derivation."""
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path("/Users/mohnish/Downloads/tsc architecture")
sys.path.insert(0, str(PROJECT_ROOT))

from tsc.models.personas import FinalPersona, PsychologicalProfile, MarketContext, BuyerJourney, EmotionalTriggers
from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.oasis.profile_builder import BuildBeliefVector

async def smoke_test_belief_vectors():
    feature = FeatureProposal(title="Test Feature", description="Test Description")
    company = CompanyContext(company_name="TestCo", team_size=100, tech_stack=["Python"])

    # 1. High-value Enterprise Decision Maker (should have high demand and velocity)
    p1 = FinalPersona(
        name="Enterprise DM",
        role="CPO",
        persona_type="EXTERNAL",
        psychological_profile=PsychologicalProfile(
            mbti="ENTJ",
            emotional_triggers=EmotionalTriggers(excited_by=["Efficiency", "ROI"], frustrated_by=[])
        ),
        market_context=MarketContext(
            buyer_role="decision-maker",
            pricing_sensitivity="low",
            sales_cycle_weeks=4,
            annual_solution_budget_usd=500_000
        ),
        buyer_journey=BuyerJourney(roi_threshold_months=6, awareness_channel="internal-mandate")
    )

    # 2. Price-sensitive Mid-market Influencer (should have lower demand and velocity)
    p2 = FinalPersona(
        name="MM Influencer",
        role="Manager",
        persona_type="EXTERNAL",
        psychological_profile=PsychologicalProfile(
            mbti="ISFJ",
            emotional_triggers=EmotionalTriggers(excited_by=["Simplicity"], frustrated_by=["Complexity"])
        ),
        market_context=MarketContext(
            buyer_role="influencer",
            pricing_sensitivity="high",
            sales_cycle_weeks=12,
            annual_solution_budget_usd=20_000
        ),
        buyer_journey=BuyerJourney(roi_threshold_months=24, awareness_channel="organic-search")
    )

    # Unpack the (OpinionVector, edge_case_label) tuple
    bv1, _ = await BuildBeliefVector(p1, feature, company)
    bv2, _ = await BuildBeliefVector(p2, feature, company)

    print(f"\n--- Belief Vector Comparison ---")
    print(f"Persona: {p1.name}")
    print(f"  Market Demand:     {bv1.market_demand:.2f}")
    print(f"  Adoption Velocity: {bv1.adoption_velocity:.2f}")
    print(f"  Resource Alignment: {bv1.resource_alignment:.2f}")


    print(f"\nPersona: {p2.name}")
    print(f"  Market Demand:     {bv2.market_demand:.2f}")
    print(f"  Adoption Velocity: {bv2.adoption_velocity:.2f}")
    print(f"  Resource Alignment: {bv2.resource_alignment:.2f}")

    # Assertions
    assert bv1.market_demand > bv2.market_demand
    assert bv1.adoption_velocity > bv2.adoption_velocity
    assert bv1.resource_alignment > bv2.resource_alignment
    print("\n✅ Belief vector signals verified!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(smoke_test_belief_vectors())
