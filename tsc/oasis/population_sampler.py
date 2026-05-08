"""
Population Sampler — Scale to 1M agents without 1M LLM calls.

Architecture:
  ┌─────────────────────────────────────────────┐
  │  Declared Population  (num_agents: up to 1M) │
  │  ┌──────────────────┐  ┌───────────────────┐ │
  │  │  ACTIVE COHORT   │  │  SHADOW AGENTS    │ │
  │  │  (llm_sample_size│  │  (remaining N-k)  │ │
  │  │   default 500)   │  │                   │ │
  │  │  Full LLM turns  │  │  Zero LLM cost    │ │
  │  │  Decision Journal│  │  Inherit state    │ │
  │  │  GM resolver     │  │  from nearest     │ │
  │  └──────────────────┘  │  active neighbor  │ │
  │                         └───────────────────┘ │
  │  Post-simulation: extrapolate metrics to full N│
  └─────────────────────────────────────────────┘

This is how Aaru/Deepsona legitimately claim "million agent" scale:
they simulate a representative sample and weight the output.
We do the same — with social dynamics Aaru doesn't have.
"""

import logging
import math
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("tsc.oasis.population_sampler")


@dataclass
class ShadowAgent:
    """A non-LLM agent that inherits state from its nearest active neighbor.
    
    Shadow agents contribute to population statistics and prediction metrics
    but do NOT consume LLM tokens. Their behavioral states are extrapolated
    from the most profile-similar active agent.
    """
    agent_id: str
    agent_name: str
    segment_source: str
    declared_index: int          # Position in full declared population
    nearest_active_id: str = "" # ID of active agent whose state we inherit
    similarity_score: float = 0.0
    weight: float = 1.0          # Statistical weight in population metrics
    
    # Inherited state (copied from nearest active agent post-simulation)
    satisfaction: float = 0.5
    frustration: float = 0.0
    trust: float = 0.5
    urgency: float = 0.0
    advocacy: float = 0.0
    decisions: list = field(default_factory=list)
    signals: list = field(default_factory=list)


class PopulationSampler:
    """Manages the split between active (LLM-powered) and shadow agents.
    
    Usage:
        sampler = PopulationSampler(all_profiles, llm_sample_size=500)
        active_profiles = sampler.active_profiles   # These get LLM turns
        # ... run simulation on active_profiles ...
        sampler.propagate_states(decision_journals)  # Shadow agents inherit
        report = sampler.build_extrapolated_report(decision_journals)
    """
    
    def __init__(self, all_profiles: list, llm_sample_size: int = 500):
        self.all_profiles = all_profiles
        self.declared_n = len(all_profiles)
        self.llm_sample_size = min(llm_sample_size, self.declared_n)
        
        # Split into active + shadow
        self.active_profiles = all_profiles[:self.llm_sample_size]
        shadow_profiles = all_profiles[self.llm_sample_size:]
        
        # Build shadow agent registry
        self.shadow_agents: List[ShadowAgent] = []
        for i, profile in enumerate(shadow_profiles):
            info = profile.user_info_dict if hasattr(profile, 'user_info_dict') else {}
            persona = info.get("profile", {})
            other = persona.get("other_info", {})
            self.shadow_agents.append(ShadowAgent(
                agent_id=str(getattr(profile, 'agent_id', f"shadow_{i}")),
                agent_name=info.get("name", f"Shadow_{i}"),
                segment_source=other.get("role", info.get("description", "")),
                declared_index=self.llm_sample_size + i,
            ))
        
        logger.info(
            f"📊 Population Sampler: {self.declared_n:,} declared agents → "
            f"{self.llm_sample_size:,} active (LLM) + {len(self.shadow_agents):,} shadow"
        )
    
    def match_shadows_to_active(self):
        """Assign each shadow agent to its nearest active neighbor.
        
        Uses segment_source string matching as a lightweight proxy for
        persona embedding similarity (no ML dependency needed).
        """
        active_sources = {}
        for p in self.active_profiles:
            info = p.user_info_dict if hasattr(p, 'user_info_dict') else {}
            persona = info.get("profile", {})
            other = persona.get("other_info", {})
            aid = str(getattr(p, 'agent_id', ''))
            active_sources[aid] = other.get("role", info.get("description", "")).lower()
        
        for shadow in self.shadow_agents:
            shadow_src = shadow.segment_source.lower()
            best_id, best_score = "", 0.0
            
            for aid, src in active_sources.items():
                # Simple overlap score: shared words / total words
                shadow_words = set(shadow_src.split())
                active_words = set(src.split())
                if shadow_words or active_words:
                    overlap = len(shadow_words & active_words)
                    union = len(shadow_words | active_words) or 1
                    score = overlap / union
                    if score > best_score:
                        best_score, best_id = score, aid
            
            # Fallback: assign to first active agent
            if not best_id and self.active_profiles:
                best_id = str(getattr(self.active_profiles[0], 'agent_id', ''))
                best_score = 0.0
            
            shadow.nearest_active_id = best_id
            shadow.similarity_score = best_score
    
    def propagate_states(self, decision_journals: Dict[str, Any]):
        """Copy behavioral states from active journals to shadow agents.
        
        Call this AFTER the simulation loop completes.
        """
        self.match_shadows_to_active()
        
        propagated = 0
        for shadow in self.shadow_agents:
            journal = decision_journals.get(shadow.nearest_active_id)
            if journal:
                shadow.satisfaction = journal.satisfaction
                shadow.frustration = journal.frustration
                shadow.trust = journal.trust
                shadow.urgency = journal.urgency
                shadow.advocacy = journal.advocacy
                shadow.decisions = list(journal.decisions)
                shadow.signals = list(journal.signals[-3:])  # Last 3 signals
                propagated += 1
        
        logger.info(f"🔁 State propagated to {propagated:,} shadow agents")
    
    def build_extrapolated_report(self, decision_journals: Dict[str, Any]) -> Dict[str, Any]:
        """Build population-scale metrics from active + shadow agents combined.
        
        Returns extrapolated values that represent the FULL declared population,
        not just the LLM-active cohort.
        """
        from .models import DecisionJournal
        
        # Collect all states (active journals + shadow agents)
        all_satisfaction, all_frustration, all_trust, all_advocacy = [], [], [], []
        all_decisions = []
        
        # Active agents
        for journal in decision_journals.values():
            all_satisfaction.append(journal.satisfaction)
            all_frustration.append(journal.frustration)
            all_trust.append(journal.trust)
            all_advocacy.append(journal.advocacy)
            all_decisions.extend(journal.decisions)
        
        # Shadow agents (post propagation)
        for shadow in self.shadow_agents:
            all_satisfaction.append(shadow.satisfaction)
            all_frustration.append(shadow.frustration)
            all_trust.append(shadow.trust)
            all_advocacy.append(shadow.advocacy)
            all_decisions.extend(shadow.decisions)
        
        total = len(all_satisfaction) or 1
        
        # Population-scale metrics
        pop_high_risk = sum(1 for f in all_frustration if f > 0.6) / total
        pop_low_risk = sum(1 for s in all_satisfaction if s > 0.6) / total
        pop_promoters = sum(1 for a in all_advocacy if a > 0.6) / total
        pop_detractors = sum(1 for f in all_frustration if f > 0.6) / total
        pop_nps = round((pop_promoters - pop_detractors) * 100, 1)
        
        # Confidence interval (95%) via normal approximation
        import math
        n = self.declared_n
        se = math.sqrt((pop_high_risk * (1 - pop_high_risk)) / max(self.llm_sample_size, 1))
        ci_margin = 1.96 * se  # 95% CI
        
        return {
            "declared_population": self.declared_n,
            "llm_active_cohort": self.llm_sample_size,
            "shadow_agents": len(self.shadow_agents),
            "extrapolated_high_risk_pct": round(pop_high_risk * 100, 1),
            "extrapolated_high_risk_ci": f"±{ci_margin*100:.1f}%",
            "extrapolated_nps": pop_nps,
            "extrapolated_churn_count": int(pop_high_risk * n),
            "extrapolated_champion_count": int(pop_promoters * n),
            "statistical_confidence": "95%",
            "margin_of_error": f"±{ci_margin*100:.1f}%",
        }


def compute_sample_size_for_confidence(
    population: int,
    confidence: float = 0.95,
    margin: float = 0.05,
    p: float = 0.5,
) -> int:
    """Cochran formula: minimum sample for desired confidence + margin.
    
    Examples:
      population=1_000_000, confidence=95%, margin=±5% → 384 agents
      population=1_000_000, confidence=99%, margin=±3% → 1844 agents
    """
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    n0 = (z ** 2 * p * (1 - p)) / (margin ** 2)
    # Finite population correction
    n = n0 / (1 + (n0 - 1) / population)
    return math.ceil(n)


def recommend_sample_size(declared_n: int) -> Tuple[int, str]:
    """Return optimal LLM sample size + explanation for given population."""
    if declared_n <= 1000:
        return declared_n, "Small population: run all agents with full LLM cognition"
    elif declared_n <= 50_000:
        n = compute_sample_size_for_confidence(declared_n, 0.95, 0.05)
        return max(n, 300), f"95% confidence, ±5% margin: {n} agents needed"
    elif declared_n <= 1_000_000:
        n = compute_sample_size_for_confidence(declared_n, 0.95, 0.03)
        return max(n, 500), f"95% confidence, ±3% margin: {n} agents needed"
    else:
        return 1000, "Very large population: 1000 agents gives <3% margin at 99% confidence"
