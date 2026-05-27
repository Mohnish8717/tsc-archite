from __future__ import annotations
import logging
import time
import threading
import hashlib
from typing import Dict, List, Any, Optional, Set
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ToolReceipt:
    def __init__(self, tool_name: str, agent_name: str, call_hash: str, timestamp: float = None, verified: bool = False):
        self.tool_name = tool_name
        self.agent_name = agent_name
        self.call_hash = call_hash
        self.timestamp = timestamp or time.time()
        self.verified = verified

class VoteReceiptLedger:
    def __init__(self):
        self._receipts: Dict[str, List[ToolReceipt]] = {}
        self._lock = threading.RLock()

    def record(self, agent: str, tool: str, result: str) -> str:
        call_hash = hashlib.sha256(
            f'{agent}:{tool}:{result}:{time.time()}'.encode()
        ).hexdigest()[:16]
        with self._lock:
            self._receipts.setdefault(agent, []).append(
                ToolReceipt(tool, agent, call_hash, verified=True)
            )
        return call_hash

    def can_vote(self, agent: str, min_tools: int = 1) -> tuple[bool, str]:
        with self._lock:
            receipts = list(self._receipts.get(agent, []))
            # Parent Lineage
            for parent_key in self._receipts.keys():
                if parent_key in agent and parent_key != agent:
                    receipts.extend(self._receipts[parent_key])
            
            verified = [r for r in receipts if r.verified]
            if len(verified) < min_tools:
                return False, (
                    f'ER-401: Insufficient research. {len(verified)}/{min_tools} '
                    f'verified tool calls on record for {agent} (including parent scope).'
                )
            return True, 'VOTE_AUTHORIZED'

class CognitiveLedger:
    """AGI-Grade Shared State Ledger — replaces text-only signals with structured programmatic state."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self.confidence_history: Dict[str, list] = {}
        self.tool_call_counts: Dict[str, int] = {}
        self.adjournment_reasons: Dict[str, str] = {}
        self.has_voted: Dict[str, bool] = {}
        self.high_risk_agents: Set[str] = set()
        self.blackboard_conflicts: Dict[str, str] = {}
        self.frustration_levels: Dict[str, float] = {}
        self.veto_used: Dict[str, bool] = {}
        
        # New State: Dynamic Hierarchical Task Ledger
        self.tasks: Dict[str, dict] = {
            "T1": {"title": "Technical Feasibility & Architecture", "status": "OPEN", "resolution": "", "subtasks": {}},
            "T2": {"title": "Financial Safety & Budget Runway", "status": "OPEN", "resolution": "", "subtasks": {}},
            "T3": {"title": "Market Fit & User Adoption", "status": "OPEN", "resolution": "", "subtasks": {}},
            "T4": {"title": "Security, Legal & Compliance", "status": "OPEN", "resolution": "", "subtasks": {}}
        }
        self.agenda_handled: bool = False

    def internal_add_micro_task(self, parent_id: str, micro_id: str, desc: str):
        with self._lock:
            if parent_id in self.tasks:
                self.tasks[parent_id]["subtasks"][micro_id] = {"description": desc, "status": "OPEN", "resolution": ""}

    def internal_update_task(self, task_id: str, status: str, resolution: str):
        with self._lock:
            for t_id, t_info in self.tasks.items():
                if t_id == task_id:
                    t_info["status"] = status
                    if resolution: t_info["resolution"] = resolution
                    return
                if task_id in t_info["subtasks"]:
                    t_info["subtasks"][task_id]["status"] = status
                    if resolution: t_info["subtasks"][task_id]["resolution"] = resolution
                    return

    def has_open_tasks(self) -> bool:
        """Returns True if there are any OPEN or IN_PROGRESS tasks/subtasks."""
        for t_info in self.tasks.values():
            if t_info["status"] in ["OPEN", "IN_PROGRESS"]:
                return True
            for st_info in t_info["subtasks"].values():
                if st_info["status"] in ["OPEN", "IN_PROGRESS"]:
                    return True
        return False

    def get_pending_task_summary(self) -> str:
        """Returns a summarized list of unresolved tasks."""
        pending = []
        for t_id, t_info in self.tasks.items():
            if t_info["status"] in ["OPEN", "IN_PROGRESS"]:
                pending.append(f"{t_id} ({t_info['title']})")
            for st_id, st_info in t_info["subtasks"].items():
                if st_info["status"] in ["OPEN", "IN_PROGRESS"]:
                    pending.append(f"{st_id} ({st_info['description']})")
        return ", ".join(pending) if pending else "NONE"

    def get_formatted_agenda(self) -> str:
        lines = ["# AUTONOMOUS TASK LEDGER (Memory of Progress)\n"]
        for t_id, t_info in self.tasks.items():
            checkbox = "[x]" if t_info["status"] == "RESOLVED" else ("[~]" if t_info["status"] == "IN_PROGRESS" else "[ ]")
            lines.append(f"- {checkbox} [{t_id}] {t_info['title']} ({t_info['status']})")
            if t_info.get("resolution"):
                lines.append(f"    └ Resolution: {t_info['resolution']}")
                
            for st_id, st_info in t_info["subtasks"].items():
                s_checkbox = "[x]" if st_info["status"] == "RESOLVED" else ("[~]" if st_info["status"] == "IN_PROGRESS" else "[ ]")
                lines.append(f"    - {s_checkbox} [{st_id}] {st_info['description']} ({st_info['status']})")
                if st_info.get("resolution"):
                    lines.append(f"        └ Resolution: {st_info['resolution']}")
        
        status = "BLOCKED" if self.has_open_tasks() else "READY"
        lines.append(f"\n--- VOTING STATUS: {status} ---")
        if status == "BLOCKED":
            lines.append(f"Pending: {self.get_pending_task_summary()}")
        return "\n".join(lines)
    
    def record_confidence(self, agent_name: str, confidence: float):
        with self._lock:
            if agent_name not in self.confidence_history:
                self.confidence_history[agent_name] = []
            self.confidence_history[agent_name].append(confidence)
    
    def record_tool_call(self, agent_name: str):
        with self._lock:
            self.tool_call_counts[agent_name] = self.tool_call_counts.get(agent_name, 0) + 1
    
    def get_evolution_delta(self, agent_name: str) -> str:
        """Returns a programmatic evolution report for the critic."""
        history = self.confidence_history.get(agent_name, [])
        tool_count = self.tool_call_counts.get(agent_name, 0)
        if len(history) < 2:
            return f"EVOLUTION STATUS: First round. Tools executed: {tool_count}. No delta available yet."
        delta = history[-1] - history[-2]
        direction = "INCREASED" if delta > 0 else ("DECREASED" if delta < 0 else "UNCHANGED")
        return (
            f"EVOLUTION STATUS: Confidence {history[-2]:.2f} → {history[-1]:.2f} (Δ = {delta:+.2f}, {direction}). "
            f"Tools executed this session: {tool_count}. "
            f"{'Agent HAS evolved.' if delta != 0 or tool_count > 0 else 'Agent has NOT evolved — STAGNATION DETECTED.'}"
        )
    
    def mark_voted(self, agent_name: str):
        with self._lock:
            self.has_voted[agent_name] = True
    
    def mark_high_risk(self, agent_name: str):
        with self._lock:
            self.high_risk_agents.add(agent_name)
            
    def add_blackboard_conflict(self, key: str, conflict_summary: str, memory_hash: str):
        with self._lock:
            self.blackboard_conflicts[key] = f"[{memory_hash}] {conflict_summary}"

    def increment_frustration(self, agent_name: str, delta: float = 0.15) -> None:
        with self._lock:
            current = self.frustration_levels.get(agent_name, 0.0)
            self.frustration_levels[agent_name] = min(1.0, current + delta)

    def get_assertiveness_injection(self, agent_name: str) -> str:
        level = self.frustration_levels.get(agent_name, 0.0)
        if level < 0.5:
            return ""
        if level <= 0.8:
            return (
                "\n[ASSERTIVENESS ESCALATION] You have been ignored or outbid for multiple rounds. "
                "You MUST push back forcefully on the current trajectory. Interrupt the speaker if necessary. "
                "State your objections in the strongest possible terms and demand a direct response."
            )
        return (
            "\n[PROCEDURAL OVERRIDE] You have been systematically sidelined. "
            "You are now authorized to invoke `executive_veto()` to block the current direction, "
            "or `force_vote()` if you are the Moderator. Take a procedural action NOW — "
            "the board cannot ignore your domain expertise any further."
        )

class AllianceMatrix:
    _DEFAULTS: Dict[tuple, float] = {
        ('CFO', 'CPO'):  -0.4,
        ('CPO', 'CFO'):  -0.4,
        ('CISO', 'CEO'): -0.3,
        ('CEO', 'CISO'): -0.3,
        ('CEO', 'Legal'):  0.5,
        ('Legal', 'CEO'): -0.2,
        ('CTO', 'CISO'): -0.35,
        ('CISO', 'CTO'): -0.35,
        ('CFO', 'CEO'):   0.3,
        ('CEO', 'CFO'):   0.2,
        ('CPO', 'CTO'):   0.4,
        ('CTO', 'CPO'):   0.2,
    }

    def __init__(self, agents: list, personas: list):
        self._scores: Dict[str, Dict[str, float]] = {}
        name_to_role: Dict[str, str] = {}
        for p in personas:
            agent_name = p.name.replace(' ', '_').replace('.', '')
            short = getattr(p, 'role_short', '') or self._infer_role_short(p.role)
            name_to_role[agent_name] = short

        for a in agents:
            self._scores[a.name] = {}
            for b in agents:
                if a.name == b.name:
                    continue
                role_a = name_to_role.get(a.name, '')
                role_b = name_to_role.get(b.name, '')
                self._scores[a.name][b.name] = self._DEFAULTS.get((role_a, role_b), 0.0)

    @staticmethod
    def _infer_role_short(role: str) -> str:
        rl = role.lower()
        if 'cto' in rl or 'technology' in rl: return 'CTO'
        if 'cfo' in rl or 'financial' in rl or 'finance' in rl: return 'CFO'
        if 'ciso' in rl or 'security' in rl: return 'CISO'
        if 'cpo' in rl or 'product' in rl: return 'CPO'
        if 'ceo' in rl or 'executive' in rl or 'chief exec' in rl: return 'CEO'
        if 'legal' in rl or 'counsel' in rl: return 'Legal'
        if 'marketing' in rl or 'cmo' in rl: return 'CMO'
        if 'data' in rl or 'cdo' in rl: return 'CDO'
        if 'sales' in rl: return 'Sales'
        if 'hr' in rl or 'people' in rl: return 'HR'
        return 'Other'

    def get(self, agent_a: str, agent_b: str) -> float:
        return self._scores.get(agent_a, {}).get(agent_b, 0.0)

    def set(self, agent_a: str, agent_b: str, score: float) -> None:
        if agent_a not in self._scores:
            self._scores[agent_a] = {}
        self._scores[agent_a][agent_b] = max(-1.0, min(1.0, score))


# ── Quadratic Voting Protocol ───────────────────────────────────────────

def apply_quadratic_voting_constraints(adjustments: Dict[str, float], credit_budget: float = 100.0) -> Dict[str, float]:
    """
    Applies Quadratic Voting budget constraints to an agent's adjustment scores.
    An agent has a budget of `credit_budget` credits (default 100).
    The cost of voting on a dimension is proportional to the square of its deviation from neutral (0.5).
    Formula: Credits_i = (d_i * 20)^2 where d_i = |adjustment_i - 0.5|.
    If sum(Credits_i) > budget, we scale the deviations quadratically to fit within the budget.
    """
    if not adjustments:
        return {}

    deviations: Dict[str, float] = {}
    total_credits = 0.0
    for dim, val in adjustments.items():
        dev = val - 0.5
        deviations[dim] = dev
        credits_spent = (abs(dev) * 20.0) ** 2
        total_credits += credits_spent

    if total_credits <= credit_budget or total_credits == 0.0:
        return adjustments

    # We scale down deviations quadratically: d_scaled = d * sqrt(budget / total_spent)
    scale_factor = (credit_budget / total_credits) ** 0.5
    scaled_adjustments: Dict[str, float] = {}
    for dim, dev in deviations.items():
        scaled_dev = dev * scale_factor
        scaled_adjustments[dim] = 0.5 + scaled_dev

    logger.info("Quadratic Voting: Scaled down adjustments from total cost %.1f to budget %.1f", total_credits, credit_budget)
    return scaled_adjustments
