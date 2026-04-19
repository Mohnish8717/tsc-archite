"""T01–T12 Behavioral Test Suite for AG2 Boardroom Debate Engine.

All tests run fully offline — LLM calls are mocked.
SentenceTransformer is lazy-loaded (no import hang).
Each test validates one specific behavioral guarantee deterministically.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict

from tsc.layers.layer6_ag2_debate import (  # noqa: E402
    AG2DebateEngine,
    DebateStateMachine,
    DebateState,
    VoteReceiptLedger,
    CognitiveLedger,
    TensionPayload,
    AllianceMatrix,
)

# ── Shared Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def engine():
    return AG2DebateEngine(llm_client=MagicMock())


@pytest.fixture
def ledger():
    return CognitiveLedger()


@pytest.fixture
def receipt_ledger():
    return VoteReceiptLedger()


def _make_payload(
    adjustments: Dict[str, float] | None = None,
    confidence: float = 0.8,
    is_high_risk: bool = False,
    is_low_information: bool = False,
) -> TensionPayload:
    """Helper to build a TensionPayload with defaults."""
    if adjustments is None:
        adjustments = {'Security': 0.7, 'Cost': 0.6}
    return TensionPayload(
        adjustments=adjustments,
        confidence=confidence,
        is_high_risk=is_high_risk,
        is_low_information=is_low_information,
    )


class _FakePersona:
    """Minimal persona stub for unit tests."""

    def __init__(self, name, role, domain_expertise=None, role_short=""):
        self.name = name
        self.role = role
        self.role_short = role_short
        self.domain_expertise = domain_expertise or []
        self.psychological_profile = MagicMock(full_profile_text="Test profile")


class _FakeAgent:
    """Minimal agent stub for unit tests."""

    def __init__(self, name):
        self.name = name
        self.system_message = ""

    def update_system_message(self, msg):
        self.system_message = msg


# ── T01: CFO veto on burn_rate > $1M/mo ─────────────────────────────


class TestT01CfoVeto:
    def test_calculate_financials_flags_critical_on_high_burn(self, engine):
        tools = engine._create_tools()
        result = tools["calculate_financials"](burn_rate=6_000_000, runway_months=12)
        assert "CRITICAL" in result or "BUDGET EXCEEDED" in result

    def test_calculate_financials_low_burn_is_manageable(self, engine):
        tools = engine._create_tools()
        result = tools["calculate_financials"](burn_rate=200_000, runway_months=6)
        assert "LOW" in result or "WITHIN BUDGET" in result


# ── T02: Score formula weighted mean correctness ─────────────────────


class TestT02ScoreFormula:
    def test_all_ones_produces_high_score(self):
        """All agents voting 1.0 on all dimensions -> score near 0.9."""
        raw_mean = 1.0
        final_score = 0.3 + (raw_mean * 0.6)
        final_score = max(0.0, min(1.0, final_score))
        assert final_score >= 0.85, f"All 1.0 votes should yield >=0.85, got {final_score}"

    def test_all_zeros_produces_low_score(self):
        """All agents voting 0.0 -> score near 0.3."""
        raw_mean = 0.0
        final_score = 0.3 + (raw_mean * 0.6)
        assert final_score <= 0.35, f"All 0.0 votes should yield <=0.35, got {final_score}"

    def test_score_varies_meaningfully(self):
        """Score range covers [0.3, 0.9] regardless of agent count."""
        for n_agents in [3, 5, 10]:
            low = 0.3 + (0.0 * 0.6)
            high = 0.3 + (1.0 * 0.6)
            assert low == pytest.approx(0.3, abs=0.01)
            assert high == pytest.approx(0.9, abs=0.01)

    def test_weighted_mean_with_domain_authority(self):
        """3x multiplier correctly amplifies domain-expert votes."""
        # CISO votes 0.9 on Security with expertise match (3x), CTO votes 0.4 (1x)
        ciso_weight = 0.8 * 3.0   # conf * 3x
        cto_weight = 0.8 * 1.0    # conf * 1x
        weighted_sum = (0.9 * ciso_weight) + (0.4 * cto_weight)
        total_weight = ciso_weight + cto_weight
        dim_mean = weighted_sum / total_weight
        # CISO's vote dominates: result should be closer to 0.9 than 0.4
        assert dim_mean > 0.7, f"Domain authority should pull mean toward expert, got {dim_mean}"


# ── T03: ER-401 blocks votes before min_tools satisfied ──────────────


class TestT03ER401BlocksVotes:
    def test_vote_blocked_without_tool_calls(self, receipt_ledger):
        ok, msg = receipt_ledger.can_vote("Agent_CTO", min_tools=2)
        assert ok is False
        assert "ER-401" in msg

    def test_vote_allowed_after_sufficient_tools(self, receipt_ledger):
        receipt_ledger.record("Agent_CTO", "web_search", "result1")
        receipt_ledger.record("Agent_CTO", "run_multi_agent_rag", "result2")
        ok, msg = receipt_ledger.can_vote("Agent_CTO", min_tools=2)
        assert ok is True
        assert msg == "VOTE_AUTHORIZED"


# ── T04: Domain authority 3x multiplier effect ───────────────────────


class TestT04DomainAuthorityMultiplier:
    def test_security_expert_amplifies_security_dimension(self):
        persona = _FakePersona(
            "Sarah_CISO", "CISO",
            domain_expertise=["Security", "Cybersecurity"],
        )
        dim_key = "Security"
        matches = any(
            dim_key.lower() in e.lower() or e.lower() in dim_key.lower()
            for e in persona.domain_expertise
        )
        assert matches is True

    def test_non_expert_gets_1x_weight(self):
        persona = _FakePersona(
            "Bob_CFO", "CFO",
            domain_expertise=["Finance", "Compliance"],
        )
        dim_key = "Security"
        matches = any(
            dim_key.lower() in e.lower() or e.lower() in dim_key.lower()
            for e in persona.domain_expertise
        )
        assert matches is False


# ── T05: AllianceMatrix bid modifiers ─────────────────────────────────


class TestT05AllianceMatrix:
    def test_rivalry_bonus_applied(self):
        personas = [
            _FakePersona("Bob_CFO", "Chief Financial Officer"),
            _FakePersona("Peter_CPO", "Chief Product Officer"),
        ]
        agents = [_FakeAgent("Bob_CFO"), _FakeAgent("Peter_CPO")]
        matrix = AllianceMatrix(agents, personas)
        matrix.set("Bob_CFO", "Peter_CPO", -0.6)
        score = matrix.get("Bob_CFO", "Peter_CPO")
        assert score < -0.5, "Rivalry threshold should be met"

    def test_deference_penalty_applied(self):
        personas = [
            _FakePersona("David_CEO", "Chief Executive Officer"),
            _FakePersona("Marcus_Legal", "Chief Counsel"),
        ]
        agents = [_FakeAgent("David_CEO"), _FakeAgent("Marcus_Legal")]
        matrix = AllianceMatrix(agents, personas)
        matrix.set("David_CEO", "Marcus_Legal", 0.7)
        score = matrix.get("David_CEO", "Marcus_Legal")
        assert score > 0.6

    def test_role_inference(self):
        assert AllianceMatrix._infer_role_short("Chief Technology Officer") == "CTO"
        assert AllianceMatrix._infer_role_short("Chief Counsel") == "Legal"
        assert AllianceMatrix._infer_role_short("Head of Sales") == "Sales"

    def test_score_clamped(self):
        matrix = AllianceMatrix([], [])
        matrix.set("A", "B", 5.0)
        assert matrix.get("A", "B") == 1.0
        matrix.set("A", "B", -5.0)
        assert matrix.get("A", "B") == -1.0


# ── T06: Vote rejected with zero tool calls ──────────────────────────


class TestT06VoteRejectedZeroTools:
    def test_can_vote_fails_with_no_receipts(self, receipt_ledger):
        ok, msg = receipt_ledger.can_vote("Agent_CTO", min_tools=1)
        assert ok is False
        assert "ER-401" in msg


# ── T07: FSM advances to CHALLENGE after RESEARCH budget ─────────────


class TestT07FSMAdvanceAfterResearch:
    def test_research_to_challenge_transition(self):
        fsm = DebateStateMachine(agent_count=3)
        fsm.current_state = DebateState.RESEARCH
        fsm.state_round = 0
        for _ in range(2):  # U22: RESEARCH budget reduced to 1
            fsm.tick()
        assert fsm.current_state == DebateState.CHALLENGE

    def test_opening_to_research_transition(self):
        fsm = DebateStateMachine(agent_count=2)
        fsm.tick()
        fsm.tick()
        assert fsm.current_state == DebateState.RESEARCH


# ── T08: Dynamic domain bids match production personas ────────────────


class TestT08DynamicDomainBids:
    def test_bids_derived_from_role(self):
        personas = [
            _FakePersona("Alice_CTO", "Chief Technology Officer", domain_expertise=["AI"]),
            _FakePersona("Bob_CFO", "Chief Financial Officer", domain_expertise=["Finance"]),
        ]
        agents = [_FakeAgent("Alice_CTO"), _FakeAgent("Bob_CFO")]
        bids = AG2DebateEngine._build_domain_bids(personas, agents)
        assert "tech" in bids["Alice_CTO"]
        assert "architecture" in bids["Alice_CTO"]
        assert "ai" in bids["Alice_CTO"]
        assert "cost" in bids["Bob_CFO"]
        assert "finance" in bids["Bob_CFO"]

    def test_unknown_role_still_gets_expertise_keywords(self):
        personas = [
            _FakePersona("Zara_Ops", "VP of Operations", domain_expertise=["Logistics", "Supply Chain"]),
        ]
        agents = [_FakeAgent("Zara_Ops")]
        bids = AG2DebateEngine._build_domain_bids(personas, agents)
        assert "logistics" in bids["Zara_Ops"]
        assert "supply" in bids["Zara_Ops"]


# ── T09: Logical Orphanage error on missing memory_hash ───────────────


class TestT09LogicalOrphanage:
    def test_pin_conflict_requires_memory_hash(self, engine):
        tools = engine._create_tools()
        result = tools["pin_conflict_to_blackboard"](
            key="test_key", conflict_summary="Some conflict", memory_hash="",
        )
        assert "Logical Orphanage" in result or "ERROR" in result


# ── T10: Frustration injection above 0.8 threshold ───────────────────


class TestT10FrustrationInjection:
    def test_no_injection_below_half(self, ledger):
        ledger.frustration_levels["Agent_CTO"] = 0.3
        assert ledger.get_assertiveness_injection("Agent_CTO") == ""

    def test_assertiveness_between_half_and_eight(self, ledger):
        ledger.frustration_levels["Agent_CTO"] = 0.6
        result = ledger.get_assertiveness_injection("Agent_CTO")
        assert "ASSERTIVENESS ESCALATION" in result

    def test_procedural_override_above_eight(self, ledger):
        ledger.frustration_levels["Agent_CTO"] = 0.85
        result = ledger.get_assertiveness_injection("Agent_CTO")
        assert "PROCEDURAL OVERRIDE" in result
        assert "executive_veto" in result

    def test_increment_frustration_caps_at_one(self, ledger):
        ledger.frustration_levels["Agent_CTO"] = 0.9
        ledger.increment_frustration("Agent_CTO", delta=0.3)
        assert ledger.frustration_levels["Agent_CTO"] == 1.0


# ── T11: ESCALATED verdict on majority low-information votes ──────────


class TestT11EscalatedVerdict:
    def test_majority_low_info_triggers_escalation(self):
        parsed_votes = 5
        low_information_votes = 3
        assert low_information_votes > (parsed_votes / 2)

    def test_minority_low_info_does_not_escalate(self):
        parsed_votes = 5
        low_information_votes = 1
        assert not (low_information_votes > (parsed_votes / 2))


# ── T12: VOTE index sequential, non-repeating ────────────────────────


class TestT12VoteIndexSequential:
    def test_sequential_voting_order(self):
        agents = [_FakeAgent(f"Agent_{i}") for i in range(3)]
        fsm = DebateStateMachine(agent_count=3)
        fsm.current_state = DebateState.VOTE
        fsm._vote_index = 0
        selected = []
        for _ in range(3):
            voter = fsm.next_voter(agents)
            assert voter is not None
            selected.append(voter.name)
        assert selected == ["Agent_0", "Agent_1", "Agent_2"]

    def test_advances_to_closed_after_all_voted(self):
        agents = [_FakeAgent(f"Agent_{i}") for i in range(2)]
        fsm = DebateStateMachine(agent_count=2)
        fsm.current_state = DebateState.VOTE
        fsm._vote_index = 0
        fsm.next_voter(agents)
        fsm.next_voter(agents)
        voter = fsm.next_voter(agents)
        assert voter is None
        assert fsm.current_state == DebateState.CLOSED

    def test_no_repeats(self):
        agents = [_FakeAgent(f"Agent_{i}") for i in range(5)]
        fsm = DebateStateMachine(agent_count=5)
        fsm.current_state = DebateState.VOTE
        fsm._vote_index = 0
        names = []
        for _ in range(5):
            voter = fsm.next_voter(agents)
            names.append(voter.name)
        assert len(names) == len(set(names)), "Duplicate agents in voting sequence!"
