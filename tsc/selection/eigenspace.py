"""
Phase 2: Eigenspace Projector.

Projects each FinalPersona onto the TensionVector's principal axis and
assigns an ATTRACTOR / REPELLER / SWING pole without any LLM calls.

Algorithm:
1. For every tension dimension, score each persona's profile text by
   counting keyword signals associated with +agreement or -resistance.
2. Weight the scores by the tension dimension's absolute value.
3. Sum → pole_score ∈ [-1, +1]
4. Classify:
     pole_score > +THRESHOLD → ATTRACTOR
     pole_score < -THRESHOLD → REPELLER
     otherwise              → SWING
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from tsc.models.personas import FinalPersona
from tsc.selection.models import PersonaCategory, PersonaPole, PoleType, TensionVector

# ─── Domain → Stakeholder Category mapping ────────────────────────────────────
_STAKEHOLDER_DOMAINS = {
    "legal", "compliance", "regulatory", "law", "policy",
    "hr", "human resources", "retention", "talent",
    "management", "executive", "cto", "vp", "director",
    "privacy", "gdpr", "security", "audit",
}
_ANTI_DOMAINS = {
    "competitor", "skeptic", "critic", "anti", "opponent",
    "reluctant", "resistant", "opposed",
}
# Domains whose expertise deserves an influence boost (rare but high-impact)
_HIGH_IMPACT_DOMAINS = {
    "legal": 3.0,
    "privacy": 2.5,
    "gdpr": 2.5,
    "compliance": 2.5,
    "hr": 2.0,
    "security": 2.0,
}

logger = logging.getLogger(__name__)

_POLE_THRESHOLD = 0.05   # |score| > this → ATTRACTOR or REPELLER (Lowered for real DB agents)

# ─── Dimension → keyword signals ─────────────────────────────────────
# Each entry: (positive_keywords, negative_keywords)
# positive → persona is aligned with the feature on this dimension
# negative → persona resists the feature on this dimension
_DIMENSION_SIGNALS: Dict[str, Tuple[List[str], List[str]]] = {
    # Time / Autonomy
    "time autonomy": (
        ["flexible", "async", "autonomy", "self-directed", "independent", "freedom"],
        ["meeting", "sync", "mandatory", "schedule", "interrupt", "block", "micromanagement"],
    ),
    # Privacy
    "privacy surface": (
        ["gdpr", "privacy", "consent", "data rights", "compliance", "anonymized"],
        ["share", "visible", "open", "transparent", "surveillance", "monitoring"],
    ),
    # Information flow / transparency
    "information flow": (
        ["transparency", "visibility", "report", "dashboard", "share", "insight", "clarity"],
        ["silo", "isolated", "opaque", "hidden"],
    ),
    # Team coherence
    "team coherence": (
        ["collaboration", "team player", "alignment", "coordination", "synergy", "unity"],
        ["solo", "individual", "remote-only", "distributed", "friction"],
    ),
    # Budget / Value
    "budget alignment": (
        ["cost-conscious", "roi", "budget", "efficiency", "savings", "profit", "value"],
        ["expensive", "overhead", "waste", "burn"],
    ),
    # Regulatory
    "regulatory risk": (
        ["legal", "compliance", "regulation", "policy", "law", "governance"],
        ["risk", "liability", "exposure"],
    ),
    # Technical Efficiency (formerly Complexity)
    "technical efficiency": (
        ["velocity", "automation", "scale", "performance", "optimization", "delivery"],
        ["debt", "manual", "bottleneck", "legacy"],
    ),
    "technical complexity": (
        ["devops", "infrastructure", "sre", "platform", "engineering", "innovative"],
        ["non-technical", "business", "operations", "simple"],
    ),
}


class EigenspaceProjector:
    """
    Phase 2 of PersonaSelectionEngine.
    Assigns ATTRACTOR / REPELLER / SWING poles to each persona.
    """

    def project(
        self,
        personas: List[FinalPersona],
        tension_vector: TensionVector,
    ) -> List[PersonaPole]:
        """
        Project all personas and return their pole assignments.
        """
        poles: List[PersonaPole] = []

        if not tension_vector.dimensions:
            logger.warning("Empty tension vector — all personas defaulting to SWING")
            for p in personas:
                poles.append(PersonaPole(
                    persona_name=p.name,
                    pole=PoleType.SWING,
                    pole_score=0.0,
                ))
            return poles

        for persona in personas:
            pole = self._project_one(persona, tension_vector)
            poles.append(pole)
            logger.debug(
                "Persona '%s' → %s (score=%.3f)",
                persona.name, pole.pole.value, pole.pole_score
            )

        attractor_n = sum(1 for p in poles if p.pole == PoleType.ATTRACTOR)
        repeller_n  = sum(1 for p in poles if p.pole == PoleType.REPELLER)
        swing_n     = sum(1 for p in poles if p.pole == PoleType.SWING)
        logger.info(
            "Phase 2: %d attractors, %d critics, %d swing voters",
            attractor_n, repeller_n, swing_n
        )

        # Guarantee at least one swing voter for persuadable simulation dynamics
        if swing_n == 0 and len(poles) > 0:
            # Convert the lowest |pole_score| persona to SWING
            sorted_by_abs = sorted(poles, key=lambda p: abs(p.pole_score))
            sorted_by_abs[0].pole = PoleType.SWING
            logger.info(
                "No swing voters found — forcing '%s' to SWING",
                sorted_by_abs[0].persona_name
            )

        return poles

    # ──────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────

    def _project_one(
        self,
        persona: FinalPersona,
        tension_vector: TensionVector,
    ) -> PersonaPole:
        profile_text = self._get_profile_text(persona).lower()
        dim_scores: Dict[str, float] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for dim_name, tension_value in tension_vector.dimensions.items():
            raw_score = self._score_dimension(profile_text, dim_name)
            if raw_score == 0.0:
                raw_score = self._score_dimension_dynamic(persona, dim_name)
                
            # Polarity check
            weighted_score = raw_score * tension_value
            
            abs_tension = abs(tension_value)
            weighted_sum += weighted_score * abs_tension
            total_weight += abs_tension
            print(f"  [DEBUG] Dim: {dim_name:25} | Raw: {raw_score:+.2f} | Tension: {tension_value:+.2f} | Weighted: {weighted_score:+.2f}")

        pole_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
        # Clamp
        pole_score = max(-1.0, min(1.0, pole_score))

        if pole_score > 0.03:
            pole = PoleType.ATTRACTOR
        elif pole_score < -0.03:
            pole = PoleType.REPELLER
        else:
            pole = PoleType.SWING

        category, domain_expertise = self._classify_category(persona)
        influence_weight = self._compute_influence_weight(profile_text, domain_expertise)

        return PersonaPole(
            persona_name=persona.name,
            pole=pole,
            pole_score=float(round(float(pole_score), 4)),
            category=category,
            influence_weight=influence_weight,
            domain_expertise=domain_expertise,
            dimension_scores=dim_scores,
        )

    def _score_dimension(self, profile_text: str, dim_name: str) -> float:
        """
        Score how strongly a persona aligns (+) or resists (-) one dimension.
        Uses lookup table; falls back to ±0 if dimension unknown.
        """
        key = dim_name.lower()
        # Try exact match then partial match
        signals = _DIMENSION_SIGNALS.get(key)
        if signals is None:
            for k, v in _DIMENSION_SIGNALS.items():
                if k in key or key in k:
                    signals = v
                    break

        if signals is None:
            return 0.0

        pos_keywords, neg_keywords = signals
        score = 0.0
        for kw in pos_keywords:
            if kw in profile_text:
                score += 0.25
        for kw in neg_keywords:
            if kw in profile_text:
                score -= 0.25
        return max(-1.0, min(1.0, score))

    def _score_dimension_dynamic(self, persona: FinalPersona, dim_name: str) -> float:
        """Dynamic keyword matching using persona traits for unknown dimensions."""
        import re
        # Split PascalCase/camelCase into spaces first
        spaced_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', dim_name)
        key = spaced_name.lower()
        
        # Collect all words and the full key
        dim_kws = [key] + [w for w in key.split() if len(w) > 3]
        
        score: float = 0.0
        full_text = persona.psychological_profile.full_profile_text.lower()
        
        # Check positive alignment (excited_by + traits + bio)
        pos_parts = (
            persona.psychological_profile.emotional_triggers.excited_by +
            persona.psychological_profile.key_traits +
            [full_text]
        )
        pos_text = " ".join(pos_parts).lower()
        
        for kw in dim_kws:
            if kw in pos_text:
                score += 0.25
                
        # Check negative alignment (frustrated_by + scared_of + bio)
        neg_parts = (
            persona.psychological_profile.emotional_triggers.frustrated_by +
            persona.psychological_profile.emotional_triggers.scared_of +
            [full_text]
        )
        neg_text = " ".join(neg_parts).lower()
        
        for kw in dim_kws:
            if kw in neg_text:
                score -= 0.25
        
        print(f"  [DEBUG] Dynamic Dim: {dim_name} | KWs: {dim_kws} | Score: {score} | PosLen: {len(pos_text)}")
        return max(-1.0, min(1.0, score))

    def _classify_category(
        self,
        persona: FinalPersona,
    ) -> tuple[PersonaCategory, list[str]]:
        """
        Classify persona into PRIMARY / STAKEHOLDER / ANTI using role + profile text.
        Returns (category, domain_expertise_list).
        """
        text = self._get_profile_text(persona).lower()

        matched_domains: list[str] = []
        is_stakeholder = any(d in text for d in _STAKEHOLDER_DOMAINS)
        is_anti = any(d in text for d in _ANTI_DOMAINS)

        # Collect matched high-impact domains for influence weighting
        for domain in _HIGH_IMPACT_DOMAINS:
            if domain in text:
                matched_domains.append(domain)

        if is_anti:
            return PersonaCategory.ANTI, matched_domains
        if is_stakeholder:
            return PersonaCategory.STAKEHOLDER, matched_domains
        return PersonaCategory.PRIMARY, matched_domains

    def _compute_influence_weight(self, profile_text: str, domain_expertise: list[str]) -> float:
        """
        Compute influence_weight multiplier.
        Rare domain experts (Legal, Privacy, HR) get boosted weight
        so their 1 vote counts as equivalent to N regular votes in OASIS.
        """
        weight = 1.0
        for domain in domain_expertise:
            boost = _HIGH_IMPACT_DOMAINS.get(domain, 1.0)
            if boost > weight:
                weight = boost  # take the highest applicable boost
        return weight

    def _get_profile_text(self, persona: FinalPersona) -> str:
        """Concatenate all text available on the persona for scoring."""
        parts = [
            persona.role,
            persona.psychological_profile.full_profile_text,
            " ".join(persona.psychological_profile.key_traits),
            " ".join(persona.psychological_profile.emotional_triggers.frustrated_by),
            " ".join(persona.psychological_profile.emotional_triggers.excited_by),
            " ".join(persona.psychological_profile.emotional_triggers.scared_of),
        ]
        return " ".join(p for p in parts if p)
