from __future__ import annotations
import logging
from enum import Enum, auto
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class DebateState(Enum):
    OPENING    = auto()  # Initial framing, 1 turn per agent
    RESEARCH   = auto()  # Mandatory Discovery phase before any stance
    CHALLENGE  = auto()  # Red team + contrarian adversarial phase
    MITIGATION = auto()  # Proposed solutions and risk mitigations
    VOTE       = auto()  # Final vote collection, no new arguments
    CLOSED     = auto()  # Termination sentinel

class DebateStateMachine:
    TRANSITIONS = {
        DebateState.OPENING:    [DebateState.RESEARCH],
        DebateState.RESEARCH:   [DebateState.CHALLENGE, DebateState.VOTE],
        DebateState.CHALLENGE:  [DebateState.MITIGATION, DebateState.VOTE],
        DebateState.MITIGATION: [DebateState.VOTE],
        DebateState.VOTE:       [DebateState.CLOSED],
    }
    STATE_ROUND_BUDGETS = {
        DebateState.OPENING:    2,   # U22: Broadcast handles initial stances
        DebateState.RESEARCH:   1,   # U22: Prefetch eliminates this phase
        DebateState.CHALLENGE:  3,   # U22: Top 3 conflicts only
        DebateState.MITIGATION: 2,
        DebateState.VOTE:       4,   # U22: Batch voting
    }

    def __init__(self, agent_count: int):
        self.current_state = DebateState.OPENING
        self.state_round   = 0
        self._vote_index   = 0
        self._agent_count  = agent_count

    def tick(self) -> DebateState:
        self.state_round += 1
        budget = self.STATE_ROUND_BUDGETS.get(self.current_state, 99)
        if self.state_round >= budget:
            self.advance()
        return self.current_state

    def advance(self, override: Optional[DebateState] = None) -> None:
        valid = self.TRANSITIONS.get(self.current_state, [])
        if override and override in valid:
            self.current_state = override
        elif valid:
            self.current_state = valid[0]
        self.state_round = 0
        if self.current_state == DebateState.VOTE:
            self._vote_index = 0

    def next_voter(self, agents: list) -> Optional[object]:
        """Return the next agent to vote, or None if all have voted."""
        if self._vote_index >= len(agents):
            self.advance(override=DebateState.CLOSED)
            return None
        agent = agents[self._vote_index]
        self._vote_index += 1
        return agent
