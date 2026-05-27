from __future__ import annotations
import threading
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

class TensionPayload(BaseModel):
    """Structured Pydantic Model for exact JSON schema outputs via AG2."""
    adjustments: Dict[str, float] = Field(..., description='Arbitrary domain key -> score [0.0, 1.0] (e.g. "Unit Economics", "Latency")')
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence from 0.0 to 1.0.")
    is_high_risk: bool = Field(..., description="Boolean flag for critical threat.")
    is_low_information: bool = Field(False, description="Flag True if 3 consecutive searches failed (Confidence Decay).")
    tool_call_hashes: List[str] = Field(default_factory=list, description='SHA256 prefixes from VoteReceiptLedger')

    @field_validator('adjustments')
    @classmethod
    def validate_scores(cls, v: Dict[str, float]):
        for dim, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f'{dim}: score {score} outside [0.0, 1.0]')
        return v

class ThreadSafeDict(dict):
    """Thread-safe dictionary implementation for multi-threaded multi-agent systems."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)

    def get(self, key, default=None):
        with self._lock:
            return super().get(key, default)

    def keys(self):
        with self._lock:
            return list(super().keys())

    def values(self):
        with self._lock:
            return list(super().values())

    def items(self):
        with self._lock:
            return list(super().items())

    def __len__(self):
        with self._lock:
            return super().__len__()

    def __contains__(self, key):
        with self._lock:
            return super().__contains__(key)

    def clear(self):
        with self._lock:
            super().clear()

    def setdefault(self, key, default=None):
        with self._lock:
            return super().setdefault(key, default)

    def pop(self, key, *args):
        with self._lock:
            return super().pop(key, *args)

class DebateStateCoordinator:
    """Thread-safe state coordinator for parallel multi-agent debate sessions."""
    
    def __init__(self, ledger, state_machine, receipt_ledger=None, reasoning_only: bool = False):
        self._lock = threading.Lock()
        self._ledger = ledger
        self._state_machine = state_machine
        self._receipt_ledger = receipt_ledger
        self.reasoning_only = reasoning_only
        self.live_tension_registry = ThreadSafeDict()
        self._voted_agents = set()
        self._high_risk_flags = {}
        self._confidence_ledger = {}

    def submit_tension_vector(self, agent_name: str, payload: TensionPayload) -> str:
        """Enforces thread-safety during parallel tool execution by boardroom agents."""
        with self._lock:
            if not hasattr(payload, "adjustments"):
                return "ER-400: TASK COMPLIANCE FAILURE. You must provide numerical adjustments."

            # Check research requirement via VoteReceiptLedger if present
            if self._receipt_ledger:
                ok, msg = self._receipt_ledger.can_vote(agent_name, min_tools=1)
                # U18: Bypassed research requirement if reasoning_only is active or explicit low info flagged
                if not ok and not (payload.is_low_information or self.reasoning_only):
                    return msg

            # Apply SOTA Quadratic Voting constraints to ensure budget compliance
            from tsc.layers.debate_ledger import apply_quadratic_voting_constraints
            
            payload.adjustments = apply_quadratic_voting_constraints(payload.adjustments)

            self.live_tension_registry[agent_name] = payload
            self._confidence_ledger[agent_name] = float(payload.confidence)
            self._voted_agents.add(agent_name)
            if payload.is_high_risk:
                self._high_risk_flags[agent_name] = True
            
            # Record in ledger
            self._ledger.record_confidence(agent_name, float(payload.confidence))
            self._ledger.mark_voted(agent_name)
            if payload.is_high_risk:
                self._ledger.mark_high_risk(agent_name)

            logger.info("Thread-Safe Vote registered for agent: %s (Confidence: %.2f)", agent_name, payload.confidence)
            return (
                f"\nCAST VOTE ALERT:\n"
                f"{agent_name} has officially registered a Confidence of {payload.confidence}.\n"
                f"High Risk Veto Triggered: {payload.is_high_risk}\n"
                f"Mathematical Alignments: {payload.adjustments}\n"
                f"[VOTE RECORDED — SUB-DEBATE WILL NOW TERMINATE]"
            )

    def get_consensus_metrics(self) -> Dict[str, Any]:
        """Calculates consensus metrics under lock."""
        with self._lock:
            if not self._confidence_ledger:
                return {"approval_confidence": 0.0, "high_risk_vetos": 0}
            
            avg_confidence = sum(self._confidence_ledger.values()) / len(self._confidence_ledger)
            high_risk_vetos = sum(1 for val in self._high_risk_flags.values() if val)
            
            return {
                "approval_confidence": avg_confidence,
                "high_risk_vetos": high_risk_vetos,
                "voter_count": len(self._voted_agents)
            }
