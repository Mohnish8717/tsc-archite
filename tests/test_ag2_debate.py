import pytest
import os
import json
from unittest.mock import MagicMock, patch

from tsc.layers.layer6_ag2_debate import AG2DebateEngine
from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.models.personas import FinalPersona, PsychologicalProfile
from tsc.models.gates import GatesSummary
from tsc.models.graph import KnowledgeGraph

@pytest.fixture
def mock_feature():
    return FeatureProposal(
        feature_id="ft-001",
        name="Predictive Revenue Generation",
        description="Uses AI to generate predictive revenue models.",
        target_audience="Enterprise",
        core_problem="Lack of predictive visibility",
        solution_hypothesis="AI models provide 95% accuracy.",
        metrics={"KPI": "ARR Growth"},
        constraints={"Budget": "$5M"}
    )

@pytest.fixture
def mock_company():
    return CompanyContext(
        company_name="Acme Corp",
        industry="SaaS",
        current_market_position="Challenger",
        core_value_proposition="We predict the future.",
        key_competitors=["Globex"],
        business_model="B2B Subscription",
        revenue_stage="Series B",
        strategic_goals=["Increase Enterprise MRR"]
    )

@pytest.fixture
def mock_personas():
    profile = PsychologicalProfile(
        openness=0.8,
        conscientiousness=0.9,
        extraversion=0.5,
        agreeableness=0.2,
        neuroticism=0.4,
        dark_triad_traits={"Machiavellianism": 0.8},
        cognitive_biases=[],
        full_profile_text="Highly analytical and slightly machiavellian CTO."
    )
    cto = FinalPersona(
        name="Alice CTO",
        role="Chief Technology Officer",
        domain_expertise=["Architecture", "AI"],
        goals=["Stability", "Innovation"],
        pain_points=["Tech Debt"],
        psychological_profile=profile
    )
    return [cto]

@pytest.mark.asyncio
async def test_ag2_debate_engine_initialization():
    engine = AG2DebateEngine(llm_client=MagicMock())
    assert engine.primary_config is not None
    assert engine.critic_config is not None
    
    tools = engine._create_tools()
    assert "run_pre_mortem_simulation" in tools
    assert "generate_vision_mockup" in tools
    
    # Verify the native execution dir maps out
    assert os.path.exists(engine.executor_dir)
