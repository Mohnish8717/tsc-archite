"""Unit tests for SOTA Re-Architecture of Tenant State Consensus (TSC) post-simulation layers.

This suite covers:
1. Layer 2 Feature Discovery: priority sorting of customer data chunks, batch map-reduce parsing, and Qdrant semantic deduplication.
2. Boardroom QA: Neo4j Belief Graph population, grounded context query, and parallel async LLM execution.
3. Layer 6 Debate: thread-safe DebateStateCoordinator state adjustments, CognitiveLedger task state transitions, and Quadratic Voting scaling.
4. Layer 7 Spec: Fibonacci SP estimation mapping and topological sorting + critical path DAG calculations.
5. Layer 8 Handoff: SRE Prometheus/Grafana telemetry config generation, fallback gates logic, and mirror simulation verification.
"""

import sys
import pytest
import asyncio
import time
import json
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, List, Any

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.debate import ConsensusResult, DebateRound
from tsc.models.spec import DevelopmentTask, FeatureSpecification
from tsc.oasis.models import MarketSentimentSeries
from tsc.layers.layer2_discovery import FeatureDiscoveryEngine
from tsc.layers.boardroom_qa import populate_neo4j_belief_graph, query_neo4j_beliefs, ask_the_board_async
from tsc.layers.debate_coordinator import DebateStateCoordinator, TensionPayload
from tsc.layers.debate_ledger import CognitiveLedger, VoteReceiptLedger, apply_quadratic_voting_constraints
from tsc.layers.layer7_spec import SpecGenerator
from tsc.layers.layer8_handoff import HandoffGenerator
from tsc.memory.hindsight_memory import HindsightBoardroom, LiveAgentMemory


# ─── Layer 2: Feature Discovery Tests ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_layer2_chunk_priority_sorting():
    """Verify that incoming customer data chunks are prioritized based on urgency and negative sentiment."""
    mock_llm = MagicMock()
    # Mock LLM calls for Map and Reduce phases
    mock_llm.analyze = AsyncMock(side_effect=[
        # Map phase response
        {
            "top_pain_points": [{"pain_point": "FP1", "frequency": "high", "severity": "critical", "customer_quotes": ["quote"], "affected_segments": ["Enterprise"]}],
            "proposed_features": [{"title": "F1", "description": "D1", "target_users": "Users", "justification": "Why", "customer_evidence": ["Evidence"], "affected_domains": ["D1"], "priority": "P0", "effort_estimate": "small", "competitive_advantage": "Comp"}]
        },
        # Reduce phase response
        {
            "analysis_summary": "Summary",
            "top_pain_points": [{"pain_point": "FP1", "frequency": "high", "severity": "critical", "customer_quotes": ["quote"], "affected_segments": ["Enterprise"]}],
            "proposed_features": [{"title": "F1", "description": "D1", "target_users": "Users", "justification": "Why", "customer_evidence": ["Evidence"], "affected_domains": ["D1"], "priority": "P0", "effort_estimate": "small", "competitive_advantage": "Comp"}]
        }
    ])

    engine = FeatureDiscoveryEngine(llm_client=mock_llm)
    
    class MockChunk:
        def __init__(self, text, urgency_level, sentiment_label):
            self.text = text
            self.urgency_level = urgency_level
            self.sentiment_label = sentiment_label

    chunks = [
        MockChunk(text="Chunk low-neutral", urgency_level="LOW", sentiment_label="NEUTRAL"),
        MockChunk(text="Chunk critical-negative", urgency_level="CRITICAL", sentiment_label="NEGATIVE"),
        MockChunk(text="Chunk high-neutral", urgency_level="HIGH", sentiment_label="NEUTRAL"),
        MockChunk(text="Chunk medium-anger", urgency_level="MEDIUM", sentiment_label="ANGER"),
    ]

    company = CompanyContext(
        company_name="Acme",
        industry="SaaS",
        current_market_position="Challenger",
        core_value_proposition="We predict the future.",
        key_competitors=["Globex"],
        business_model="B2B Subscription",
        revenue_stage="Series B",
        strategic_goals=["Increase Enterprise MRR"]
    )

    with patch("tsc.layers.layer2_discovery._get_qdrant", return_value=None), \
         patch("tsc.layers.layer2_discovery._embed", return_value=[[0.1]*1536]):
        await engine.process(company=company, raw_chunks=chunks)

    # Inspect the analyze calls to verify the chunks order sent to the Map agent
    calls = mock_llm.analyze.call_args_list
    assert len(calls) >= 2
    
    map_prompt = calls[0].kwargs["user_prompt"]
    # Verify critical-negative is mapped before others due to priority sorting (3+2=5 vs 1+2=3 vs 2+0=2 vs 0)
    assert "critical-negative" in map_prompt
    assert map_prompt.index("critical-negative") < map_prompt.index("medium-anger")
    assert map_prompt.index("medium-anger") < map_prompt.index("high-neutral")
    assert map_prompt.index("high-neutral") < map_prompt.index("low-neutral")


@pytest.mark.asyncio
async def test_layer2_deduplication_merge_match():
    """Verify that if new proposals match Qdrant records by > 0.85 similarity, they are merged via LLM."""
    mock_llm = MagicMock()
    mock_llm.analyze = AsyncMock(side_effect=[
        # Map phase
        {
            "top_pain_points": [],
            "proposed_features": [{"title": "F1", "description": "D1", "target_users": "Users", "justification": "Why", "customer_evidence": [], "affected_domains": ["D1"], "priority": "P0", "effort_estimate": "small", "competitive_advantage": "Comp"}]
        },
        # Reduce phase
        {
            "analysis_summary": "Summary",
            "top_pain_points": [],
            "proposed_features": [{"title": "F1", "description": "D1", "target_users": "Users", "justification": "Why", "customer_evidence": [], "affected_domains": ["D1"], "priority": "P0", "effort_estimate": "small", "competitive_advantage": "Comp"}]
        },
        # Merge phase (when similarity exceeds 0.85)
        {
            "title": "Merged Feature Title",
            "description": "Merged description",
            "target_users": "Merged target users",
            "affected_domains": ["D1", "D2"]
        }
    ])

    engine = FeatureDiscoveryEngine(llm_client=mock_llm)
    
    company = CompanyContext(company_name="Acme")

    mock_point = MagicMock()
    mock_point.score = 0.9  # Exceeds 0.85 similarity threshold
    mock_point.id = "matched_point_id"
    mock_point.payload = {
        "title": "Existing Feature Title",
        "text": "Existing description",
        "target_users": "Existing target users",
        "affected_domains": ["D1"]
    }
    
    mock_results = MagicMock()
    mock_results.points = [mock_point]
    
    mock_client = AsyncMock()
    mock_client.query_points = AsyncMock(return_value=mock_results)
    mock_client.upsert = AsyncMock()

    with patch("tsc.layers.layer2_discovery._get_qdrant", return_value=mock_client), \
         patch("tsc.layers.layer2_discovery._embed", return_value=[[0.1]*1536]):
        
        proposals = await engine.process(company=company, raw_chunks=[MagicMock(text="Customer feedback text", urgency_level="LOW", sentiment_label="NEUTRAL")])
        
        assert len(proposals) == 1
        assert proposals[0].title == "Merged Feature Title"
        assert proposals[0].description == "Merged description"
        
        # Verify upsert targeted the existing point ID
        mock_client.upsert.assert_called_once()
        points_arg = mock_client.upsert.call_args[1]["points"]
        assert points_arg[0].id == "matched_point_id"
        assert points_arg[0].payload["title"] == "Merged Feature Title"


@pytest.mark.asyncio
async def test_layer2_deduplication_no_merge_match():
    """Verify that if new proposals do not match Qdrant records by > 0.85, they are inserted normally."""
    mock_llm = MagicMock()
    mock_llm.analyze = AsyncMock(side_effect=[
        # Map phase
        {
            "top_pain_points": [],
            "proposed_features": [{"title": "F1", "description": "D1", "target_users": "Users", "justification": "Why", "customer_evidence": [], "affected_domains": ["D1"], "priority": "P0", "effort_estimate": "small", "competitive_advantage": "Comp"}]
        },
        # Reduce phase
        {
            "analysis_summary": "Summary",
            "top_pain_points": [],
            "proposed_features": [{"title": "F1", "description": "D1", "target_users": "Users", "justification": "Why", "customer_evidence": [], "affected_domains": ["D1"], "priority": "P0", "effort_estimate": "small", "competitive_advantage": "Comp"}]
        }
    ])

    engine = FeatureDiscoveryEngine(llm_client=mock_llm)
    
    company = CompanyContext(company_name="Acme")

    mock_point = MagicMock()
    mock_point.score = 0.7  # Below 0.85 similarity threshold
    mock_point.id = "different_point_id"
    mock_point.payload = {
        "title": "Different Feature Title",
        "text": "Different description",
        "target_users": "Different target users",
        "affected_domains": ["D1"]
    }
    
    mock_results = MagicMock()
    mock_results.points = [mock_point]
    
    mock_client = AsyncMock()
    mock_client.query_points = AsyncMock(return_value=mock_results)
    mock_client.upsert = AsyncMock()

    with patch("tsc.layers.layer2_discovery._get_qdrant", return_value=mock_client), \
         patch("tsc.layers.layer2_discovery._embed", return_value=[[0.1]*1536]):
        
        proposals = await engine.process(company=company, raw_chunks=[MagicMock(text="Customer feedback text", urgency_level="LOW", sentiment_label="NEUTRAL")])
        
        assert len(proposals) == 1
        assert proposals[0].title == "F1"
        
        # Verify upsert registered a new point ID (not the existing one)
        mock_client.upsert.assert_called_once()
        points_arg = mock_client.upsert.call_args[1]["points"]
        assert points_arg[0].id != "different_point_id"
        assert points_arg[0].payload["title"] == "F1"


# ─── Boardroom QA Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_boardroom_qa_populate_neo4j():
    """Verify boardroom debate memories populate Neo4j with semantic nodes and relations."""
    boardroom = HindsightBoardroom()
    
    mem = LiveAgentMemory(
        agent_name="Alice CTO",
        role="Chief Technology Officer",
        role_short="CTO",
        feature_title="Predictive Analytics",
        hindsight_bank_id="bank1"
    )
    mem._embedded_commitments = ["Commitment A"]
    mem._embedded_concessions = ["Concession A"]
    mem._embedded_proposals = ["Proposal A"]
    mem._embedded_concerns = ["Concern A"]
    
    boardroom.memories = {"Alice CTO": mem}
    
    mock_session = AsyncMock()
    mock_driver = MagicMock()
    
    class AsyncContextManagerMock:
        async def __aenter__(self):
            return mock_session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    mock_driver.session = MagicMock(return_value=AsyncContextManagerMock())
    
    with patch("tsc.layers.boardroom_qa._get_neo4j", return_value=mock_driver):
        await populate_neo4j_belief_graph(boardroom)
        
    mock_driver.session.assert_called_once()
    
    run_calls = mock_session.run.call_args_list
    assert len(run_calls) == 5  # 1 BoardMember, 1 Commitment, 1 Concession, 1 Proposal, 1 Concern
    
    queries = [call[0][0] for call in run_calls]
    assert any("MERGE (m:BoardMember" in q for q in queries)
    assert any("MERGE (c:Commitment" in q for q in queries)
    assert any("MERGE (cn:Concession" in q for q in queries)
    assert any("MERGE (pr:Proposal" in q for q in queries)
    assert any("MERGE (cr:Concern" in q for q in queries)


@pytest.mark.asyncio
async def test_boardroom_qa_query_neo4j_beliefs():
    """Verify querying Neo4j returns correctly parsed grounded evidence context."""
    mock_record = {
        "commitments": ["Commit 1"],
        "concessions": ["Concession 1"],
        "proposals": ["Proposal 1"],
        "concerns": ["Concern 1"]
    }
    
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)
    
    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    
    mock_driver = MagicMock()
    class AsyncContextManagerMock:
        async def __aenter__(self):
            return mock_session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    mock_driver.session = MagicMock(return_value=AsyncContextManagerMock())
    
    with patch("tsc.layers.boardroom_qa._get_neo4j", return_value=mock_driver):
        evidence = await query_neo4j_beliefs("Alice CTO")
        
        mock_session.run.assert_called_once()
        assert "[NEO4J BELIEF GRAPH GROUNDED EVIDENCE]" in evidence
        assert "Hard Commitments: Commit 1" in evidence
        assert "Accepted Concessions: Concession 1" in evidence
        assert "Championed Proposals: Proposal 1" in evidence
        assert "Expressed Concerns: Concern 1" in evidence


@pytest.mark.asyncio
async def test_boardroom_qa_ask_the_board_async():
    """Verify board members are queried concurrently in parallel threads using asyncio.gather."""
    boardroom = MagicMock()
    boardroom.get_agent_names.return_value = ["AgentA", "AgentB"]
    
    query_calls = []
    def mock_query_agent(agent_name, question, config):
        query_calls.append((agent_name, question))
        return f"Response from {agent_name}"
    
    boardroom.query_agent = mock_query_agent
    llm_config = {"config_list": []}
    
    with patch("tsc.layers.boardroom_qa.query_neo4j_beliefs", AsyncMock(return_value="[MOCK EVIDENCE]")):
        answers = await ask_the_board_async(boardroom, "Launch now?", llm_config)
        
    assert answers == {
        "AgentA": "Response from AgentA",
        "AgentB": "Response from AgentB"
    }
    assert len(query_calls) == 2
    for agent_name, q in query_calls:
        assert "[MOCK EVIDENCE]" in q
        assert "Launch now?" in q


# ─── Layer 6: Debate Coordination & Quadratic Voting Tests ────────────────────

def test_layer6_coordinator_thread_safety_and_vetos():
    """Verify that multiple threads can submit tension payloads safely and vetos/consensus are computed under lock."""
    from concurrent.futures import ThreadPoolExecutor
    
    ledger = MagicMock()
    state_machine = MagicMock()
    receipt_ledger = MagicMock()
    receipt_ledger.can_vote.return_value = (True, "VOTE_AUTHORIZED")
    
    coordinator = DebateStateCoordinator(ledger=ledger, state_machine=state_machine, receipt_ledger=receipt_ledger)
    
    # 20 agents voting concurrently, with 2 triggering a high-risk veto
    agents = [f"Agent_{i}" for i in range(20)]
    
    def run_vote(agent_name):
        is_veto = (agent_name in ["Agent_3", "Agent_7"])
        payload = TensionPayload(
            adjustments={"Security": 0.6 if not is_veto else 0.9, "Latency": 0.4},
            confidence=0.8,
            is_high_risk=is_veto,
            is_low_information=False
        )
        return coordinator.submit_tension_vector(agent_name, payload)
        
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(run_vote, agents))
        
    # Check thread-safe collections populated correctly
    assert len(coordinator.live_tension_registry) == 20
    assert len(coordinator._voted_agents) == 20
    assert coordinator._high_risk_flags["Agent_3"] is True
    assert coordinator._high_risk_flags["Agent_7"] is True
    
    metrics = coordinator.get_consensus_metrics()
    assert metrics["approval_confidence"] == pytest.approx(0.8)
    assert metrics["high_risk_vetos"] == 2
    assert metrics["voter_count"] == 20


def test_layer6_cognitive_ledger_task_states_and_frustration():
    """Verify CognitiveLedger manages task hierarchy (OPEN/RESOLVED) and frustration-based assertiveness escalation."""
    ledger = CognitiveLedger()
    
    # Verify open tasks block completion
    assert ledger.has_open_tasks() is True
    
    # Add a subtask to T1
    ledger.internal_add_micro_task("T1", "T1.1", "Verify SSL support")
    assert "T1.1" in ledger.tasks["T1"]["subtasks"]
    assert ledger.tasks["T1"]["subtasks"]["T1.1"]["status"] == "OPEN"
    
    # Resolve the subtask
    ledger.internal_update_task("T1.1", "RESOLVED", "SSL verified")
    assert ledger.tasks["T1"]["subtasks"]["T1.1"]["status"] == "RESOLVED"
    
    # Resolve all other top level tasks
    for tid in ["T1", "T2", "T3", "T4"]:
        ledger.internal_update_task(tid, "RESOLVED", "Resolved all constraints")
        
    assert ledger.has_open_tasks() is False
    
    # Check frustration scaling and prompt injection overrides
    assert ledger.get_assertiveness_injection("CISO") == ""
    
    ledger.increment_frustration("CISO", delta=0.6)
    assert "[ASSERTIVENESS ESCALATION]" in ledger.get_assertiveness_injection("CISO")
    
    ledger.increment_frustration("CISO", delta=0.3)
    assert "[PROCEDURAL OVERRIDE]" in ledger.get_assertiveness_injection("CISO")


def test_layer6_quadratic_voting_constraints():
    """Verify Quadratic Voting constraint math scaling: Cost = (dev * 20)^2. Total budget <= 100."""
    # Under limit: Devs: 0.1 and 0.2. Credits: (0.1*20)^2 = 4, (0.2*20)^2 = 16. Total = 20. Should not scale.
    under_limit = {"Cost": 0.6, "Security": 0.7}
    res_under = apply_quadratic_voting_constraints(under_limit, credit_budget=100.0)
    assert res_under == under_limit
    
    # Over limit: Devs: +0.4, -0.4, +0.3. Credits: 8^2 + 8^2 + 6^2 = 64 + 64 + 36 = 164. Should scale.
    over_limit = {"Cost": 0.9, "Security": 0.1, "Scale": 0.8}
    res_over = apply_quadratic_voting_constraints(over_limit, credit_budget=100.0)
    
    scale_factor = (100.0 / 164.0) ** 0.5
    assert res_over["Cost"] == pytest.approx(0.5 + 0.4 * scale_factor)
    assert res_over["Security"] == pytest.approx(0.5 - 0.4 * scale_factor)
    assert res_over["Scale"] == pytest.approx(0.5 + 0.3 * scale_factor)
    
    # Verify scaled values fall within 100 credit budget
    total_credits = sum((abs(v - 0.5) * 20.0) ** 2 for v in res_over.values())
    assert total_credits == pytest.approx(100.0)


# ─── Layer 7: Specification Generation Tests ───────────────────────────────────

@pytest.mark.asyncio
async def test_layer7_fibonacci_sp_estimation():
    """Verify that story points are mapped to the closest Fibonacci number and converted to effort days."""
    mock_llm = MagicMock()
    mock_spec_json = {
        "title": "Mock Spec",
        "executive_summary": "Summary",
        "justification": "Why",
        "tasks": [
            {"id": "TASK-1", "title": "Exact Fibonacci", "story_points": 5},
            {"id": "TASK-2", "title": "Rounding Fibonacci 4", "story_points": 4},
            {"id": "TASK-3", "title": "Rounding Fibonacci 10", "story_points": 10}
        ]
    }
    mock_llm.generate = AsyncMock(return_value=json.dumps(mock_spec_json))
    
    generator = SpecGenerator(llm_client=mock_llm)
    feature = FeatureProposal(title="FP", description="Desc")
    company = CompanyContext(company_name="Acme")
    consensus = ConsensusResult(feature_name="FP", overall_verdict="APPROVED", approval_confidence=0.9, overall_summary="Summary", debate_rounds=[])
    
    spec = await generator.process(feature, company, consensus)
    assert len(spec.development_tasks) == 3
    assert spec.development_tasks[0].effort_days == 5.0
    assert spec.development_tasks[1].effort_days == 3.0  # rounds to 3 (tie-breaker first index)
    assert spec.development_tasks[2].effort_days == 8.0  # rounds to 8


@pytest.mark.asyncio
async def test_layer7_critical_path_dag():
    """Verify topological sorting and longest-path critical path calculation on task dependency DAG."""
    mock_llm = MagicMock()
    mock_spec_json = {
        "title": "Mock DAG Spec",
        "executive_summary": "Summary",
        "justification": "Why",
        "tasks": [
            {"id": "TASK-1", "title": "Task 1", "story_points": 2, "dependencies": []},
            {"id": "TASK-2", "title": "Task 2", "story_points": 3, "dependencies": ["TASK-1"]},
            {"id": "TASK-3", "title": "Task 3", "story_points": 5, "dependencies": ["TASK-1"]},
            {"id": "TASK-4", "title": "Task 4", "story_points": 5, "dependencies": ["TASK-2", "TASK-3"]}
        ]
    }
    mock_llm.generate = AsyncMock(return_value=json.dumps(mock_spec_json))
    
    generator = SpecGenerator(llm_client=mock_llm)
    feature = FeatureProposal(title="FP", description="Desc")
    company = CompanyContext(company_name="Acme")
    consensus = ConsensusResult(feature_name="FP", overall_verdict="APPROVED", approval_confidence=0.9, overall_summary="Summary", debate_rounds=[])
    
    spec = await generator.process(feature, company, consensus)
    
    # Path 1: 1 -> 2 -> 4: 2 + 3 + 5 = 10
    # Path 2: 1 -> 3 -> 4: 2 + 5 + 5 = 12 (Critical Path)
    assert spec.critical_path == ["TASK-1", "TASK-3", "TASK-4"]


# ─── Layer 8: Telemetry & Handoff Tests ────────────────────────────────────────

@pytest.mark.asyncio
async def test_layer8_telemetry_generation_and_fallbacks():
    """Verify SRE telemetry generation works successfully with Prometheus/Grafana and has safe fallbacks."""
    # Case 1: Successful custom configuration generation
    mock_llm_success = MagicMock()
    mock_telemetry_res = {
        "prometheus_alerts_yaml": "custom_alerts",
        "prometheus_scrape_yaml": "custom_scrape",
        "grafana_dashboard_json": '{"panels":[]}'
    }
    mock_llm_success.analyze = AsyncMock(return_value=mock_telemetry_res)
    
    handoff_success = HandoffGenerator(llm_client=mock_llm_success)
    feature = FeatureProposal(title="FP", description="Desc", target_users="All")
    spec = FeatureSpecification(
        feature_name="FP", specification_markdown="MD",
        development_tasks=[DevelopmentTask(task_id="T1", name="Task 1", effort_days=3.0, priority="P0")],
        evidence_citations={}, total_effort_days=3.0, critical_path=["T1"]
    )
    
    monitoring_success = await handoff_success._build_monitoring(feature, spec)
    assert monitoring_success.prometheus_alerts_yaml == "custom_alerts"
    assert monitoring_success.prometheus_scrape_yaml == "custom_scrape"
    assert monitoring_success.grafana_dashboard_json == '{"panels":[]}'
    
    # Case 2: Fallback configs when LLM fails
    mock_llm_fail = MagicMock()
    mock_llm_fail.analyze = AsyncMock(side_effect=RuntimeError("LLM overload"))
    
    handoff_fail = HandoffGenerator(llm_client=mock_llm_fail)
    monitoring_fail = await handoff_fail._build_monitoring(feature, spec)
    
    assert "FeatureHighErrorRate" in monitoring_fail.prometheus_alerts_yaml
    assert "feature-service" in monitoring_fail.prometheus_scrape_yaml
    assert "System Performance Dashboard" in monitoring_fail.grafana_dashboard_json


@pytest.mark.asyncio
async def test_layer8_mirror_simulation_integration():
    """Verify mirror simulation execution is triggered during handoff processing and results are merged."""
    mock_llm = MagicMock()
    mock_llm.model = "mock-model"
    mock_llm.get_usage.return_value.total_tokens = 1000
    # First analyze call: mirror simulation verification
    mock_llm.analyze = AsyncMock(return_value={
        "verification_score": 0.95,
        "satisfied_pain_points": ["Pain point A resolved"],
        "unresolved_pain_points": [],
        "cohort_verdict": "PASSED",
        "detailed_feedback": "Perfect specs"
    })
    # Second generate call: leadership summary
    mock_llm.generate = AsyncMock(return_value="Summary for leadership")
    
    handoff = HandoffGenerator(llm_client=mock_llm)
    feature = FeatureProposal(title="FP", description="Desc", target_users="All")
    spec = FeatureSpecification(
        feature_name="FP", specification_markdown="MD",
        development_tasks=[DevelopmentTask(task_id="T1", name="Task 1", effort_days=3.0, priority="P0")],
        evidence_citations={}, total_effort_days=3.0, critical_path=["T1"]
    )
    consensus = ConsensusResult(
        feature_name="FP",
        overall_verdict="APPROVED", approval_confidence=0.95, overall_summary="Summary", debate_rounds=[],
        mitigations=["Mitigation A"]
    )
    sim_results = MarketSentimentSeries(simulation_id="sim1", feature_proposal_id="fp1")
    
    recommendation = await handoff.process(
        feature=feature,
        company=CompanyContext(company_name="Acme"),
        personas=[],
        consensus=consensus,
        spec=spec,
        simulation_results=sim_results,
        start_time=time.time()
    )
    
    # Verify mirror simulation results injected in verdicts
    mv_verdict = recommendation.verdicts_by_pillar["market_validation"]
    assert "mirror_simulation" in mv_verdict.details
    assert mv_verdict.details["mirror_simulation"]["verification_score"] == 0.95
    assert mv_verdict.details["mirror_simulation"]["cohort_verdict"] == "PASSED"
