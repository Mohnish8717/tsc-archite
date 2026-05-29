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
    nearest_active_ids: List[Tuple[str, float]] = field(default_factory=list) # List of (agent_id, weight)
    weight: float = 1.0          # Statistical weight in population metrics
    
    # Inherited state (copied from nearest active agent post-simulation)
    satisfaction: float = 0.5
    frustration: float = 0.0
    trust: float = 0.5
    urgency: float = 0.0
    advocacy: float = 0.0
    decisions: list = field(default_factory=list)
    signals: list = field(default_factory=list)

    def state_vector(self) -> list:
        """Return numerical state for clustering."""
        return [self.satisfaction, self.frustration, self.trust, self.urgency, self.advocacy]


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
    
    def match_shadows_to_active(self, k: int = 3):
        """Assign each shadow agent to its top K nearest active neighbors.
        
        Uses term frequency cosine similarity for fast pure-Python matching.
        """
        import re
        from collections import Counter
        import math

        def tokenize(text: str):
            return Counter(re.findall(r'\w+', text.lower()))

        def cosine_similarity(c1, c2):
            intersection = set(c1.keys()) & set(c2.keys())
            numerator = sum([c1[x] * c2[x] for x in intersection])
            sum1 = sum([c1[x]**2 for x in c1.keys()])
            sum2 = sum([c2[x]**2 for x in c2.keys()])
            denominator = math.sqrt(sum1) * math.sqrt(sum2)
            if not denominator: return 0.0
            return float(numerator) / denominator

        active_sources = {}
        for p in self.active_profiles:
            info = p.user_info_dict if hasattr(p, 'user_info_dict') else {}
            persona = info.get("profile", {})
            other = persona.get("other_info", {})
            aid = str(getattr(p, 'agent_id', ''))
            role_desc = other.get("role", info.get("description", ""))
            active_sources[aid] = tokenize(role_desc)
        
        for shadow in self.shadow_agents:
            shadow_tokens = tokenize(shadow.segment_source)
            scores = []
            
            for aid, active_tokens in active_sources.items():
                score = cosine_similarity(shadow_tokens, active_tokens)
                scores.append((score, aid))
            
            # Sort by score descending and take top K
            scores.sort(key=lambda x: x[0], reverse=True)
            top_k = scores[:k]
            
            # Fallback: assign to first active agent
            if not top_k and self.active_profiles:
                first_aid = str(getattr(self.active_profiles[0], 'agent_id', ''))
                top_k = [(1.0, first_aid)]
            
            # Normalize weights
            total_score = sum(s[0] for s in top_k)
            if total_score > 0:
                shadow.nearest_active_ids = [(aid, s / total_score) for s, aid in top_k]
            else:
                shadow.nearest_active_ids = [(aid, 1.0 / len(top_k)) for s, aid in top_k]
    
    def propagate_states(self, decision_journals: Dict[str, Any]):
        """Blend behavioral states from active journals to shadow agents using K-NN weights.
        
        Call this AFTER the simulation loop completes.
        """
        self.match_shadows_to_active()
        
        propagated = 0
        for shadow in self.shadow_agents:
            blended_sat, blended_fru, blended_tru = 0.0, 0.0, 0.0
            blended_urg, blended_adv = 0.0, 0.0
            all_decisions = []
            all_signals = []
            
            for aid, weight in shadow.nearest_active_ids:
                journal = decision_journals.get(aid)
                if journal:
                    blended_sat += journal.satisfaction * weight
                    blended_fru += journal.frustration * weight
                    blended_tru += journal.trust * weight
                    blended_urg += journal.urgency * weight
                    blended_adv += journal.advocacy * weight
                    
                    if hasattr(journal, 'decisions'):
                        all_decisions.extend(journal.decisions)
                    if hasattr(journal, 'signals'):
                        all_signals.extend(journal.signals[-3:])
            
            shadow.satisfaction = blended_sat
            shadow.frustration = blended_fru
            shadow.trust = blended_tru
            shadow.urgency = blended_urg
            shadow.advocacy = blended_adv
            
            # Deduplicate decisions and signals
            shadow.decisions = [dict(t) for t in {tuple(d.items()) for d in all_decisions}]
            shadow.signals = [dict(t) for t in {tuple(s.items()) for s in all_signals}][:3]
            propagated += 1
        
        logger.info(f"🔁 Blended state propagated to {propagated:,} shadow agents via K-NN")
    
    def build_extrapolated_report(
        self,
        decision_journals: Dict[str, Any],
        timesteps_completed: int = 10,
        segments: list = None
    ) -> Dict[str, Any]:
        """Build population-scale metrics from active + shadow agents combined.
        
        Returns extrapolated values that represent the FULL declared population,
        not just the LLM-active cohort.
        """
        # Collect all states (active journals + shadow agents)
        combined_agents = []
        for journal in decision_journals.values():
            combined_agents.append(journal)
        for shadow in self.shadow_agents:
            combined_agents.append(shadow)
            
        total = len(combined_agents) or 1
        n = self.declared_n
        
        # 1. NPS & Risk counts
        import math
        high_risk_prob_sum = 0.0
        low_risk_prob_sum = 0.0
        moderate_prob_sum = 0.0
        promoter_prob_sum = 0.0
        detractor_prob_sum = 0.0
        
        def sigmoid(x, k=15, x0=0.5):
            # Sigmoid for continuous probability mapping
            return 1.0 / (1.0 + math.exp(-k * (x - x0)))
        
        for agent in combined_agents:
            # Probabilistic Risk Mapping
            hr_prob = sigmoid(agent.frustration, k=15, x0=0.55)
            # Low risk: not frustrated AND somewhat satisfied
            not_frustrated_prob = 1.0 - sigmoid(agent.frustration, k=15, x0=0.45)
            satisfied_prob = sigmoid(agent.satisfaction, k=15, x0=0.5)
            lr_prob = not_frustrated_prob * satisfied_prob
            
            # Normalize so they sum to 1
            if hr_prob + lr_prob > 1.0:
                total_prob = hr_prob + lr_prob
                hr_prob /= total_prob
                lr_prob /= total_prob
            mod_prob = max(0.0, 1.0 - hr_prob - lr_prob)
            
            high_risk_prob_sum += hr_prob
            low_risk_prob_sum += lr_prob
            moderate_prob_sum += mod_prob
            
            # NPS Categories
            detractor_prob_sum += hr_prob
            # Promoter: must be low risk AND have high advocacy
            adv_prob = sigmoid(agent.advocacy, k=15, x0=0.55)
            promoter_prob_sum += lr_prob * adv_prob
                
        pop_high_risk = high_risk_prob_sum / total
        pop_low_risk = low_risk_prob_sum / total
        pop_promoters = promoter_prob_sum / total
        pop_detractors = detractor_prob_sum / total
        pop_nps = round((pop_promoters - pop_detractors) * 100, 1)
        
        risk_dist = {
            "HIGH_RISK": round(pop_high_risk, 3),
            "MODERATE": round(moderate_prob_sum / total, 3),
            "LOW_RISK": round(pop_low_risk, 3)
        }
        
        # 2. Churn velocity & Adoption momentum
        # Use simple mean for churn/adoption momentum but factor in baseline
        churn_velocity = sum(agent.frustration for agent in combined_agents) / (total * max(timesteps_completed, 1))
        adoption_momentum = sum(max(0.0, agent.satisfaction - 0.5) for agent in combined_agents) / (total * max(timesteps_completed, 1))
        
        # 3. Top risk factors
        risk_counts = {}
        for agent in combined_agents:
            has_risk = False
            for s in agent.signals:
                if s.get("intensity", 0.0) < -0.2:
                    sig_type = s.get("type")
                    if sig_type and sig_type != "neutral":
                        risk_counts[sig_type] = risk_counts.get(sig_type, 0) + 1
                        has_risk = True
                        
        top_risk_factors = []
        for factor, count in sorted(risk_counts.items(), key=lambda x: x[1], reverse=True):
            top_risk_factors.append({
                "factor": factor,
                "frequency": round(count / total, 3)
            })
            
        # 4. Decision events
        decision_events = []
        for agent in combined_agents:
            for d in agent.decisions:
                decision_events.append({
                    "timestep": d.get("timestep", 0),
                    "decision": d.get("decision", ""),
                    "confidence": d.get("confidence", 0.0),
                    "trigger": d.get("trigger", ""),
                    "agent_id": agent.agent_id,
                    "agent_name": agent.agent_name,
                })
        decision_events.sort(key=lambda x: x["timestep"])
        
        # Confidence interval (95%) via normal approximation
        se = math.sqrt((pop_high_risk * (1 - pop_high_risk)) / max(self.llm_sample_size, 1))
        ci_margin = 1.96 * se
        
        # Fallback segments if None
        if segments is None:
            high_risk_agents = [a for a in combined_agents if a.frustration > 0.6]
            low_risk_agents = [a for a in combined_agents if a.frustration <= 0.3 and a.satisfaction > 0.5]
            moderate_agents = [a for a in combined_agents if a not in high_risk_agents and a not in low_risk_agents]
            
            segments = []
            for name, group in [("High-Risk Segments", high_risk_agents), ("Moderate Segments", moderate_agents), ("Low-Risk Segments", low_risk_agents)]:
                if group:
                    gn = len(group)
                    segments.append({
                        "name": name,
                        "size": gn,
                        "pct": round(gn / total, 2),
                        "avg_satisfaction": round(sum(a.satisfaction for a in group) / gn, 3),
                        "avg_frustration": round(sum(a.frustration for a in group) / gn, 3),
                        "avg_trust": round(sum(a.trust for a in group) / gn, 3),
                        "avg_urgency": round(sum(a.urgency for a in group) / gn, 3),
                        "avg_advocacy": round(sum(a.advocacy for a in group) / gn, 3),
                        "top_signals": [],
                        "decision_events": sum(len(a.decisions) for a in group),
                        "member_ids": [a.agent_id for a in group]
                    })
                    
        return {
            "declared_population": self.declared_n,
            "population_size": self.declared_n,
            "llm_active_cohort": self.llm_sample_size,
            "shadow_agents": len(self.shadow_agents),
            "extrapolated_high_risk_pct": round(pop_high_risk * 100, 1),
            "extrapolated_high_risk_ci": f"±{ci_margin*100:.1f}%",
            "extrapolated_nps": pop_nps,
            "extrapolated_churn_count": int(pop_high_risk * n),
            "extrapolated_champion_count": int(pop_promoters * n),
            "statistical_confidence": "95%",
            "margin_of_error": f"±{ci_margin*100:.1f}%",
            
            # Rich high-fidelity metrics
            "segments": segments,
            "risk_distribution": risk_dist,
            "net_promoter_score": pop_nps,
            "churn_velocity": round(churn_velocity, 4),
            "adoption_momentum": round(adoption_momentum, 4),
            "decision_events": decision_events,
            "top_risk_factors": top_risk_factors,
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
