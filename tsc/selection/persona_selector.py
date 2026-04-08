"""
Tension-Aware Persona Selector.

Pre-selects the best "seeds" from a large pool of DB personas before
handing them to the GMM engine. Replaces naive random picking with a
Friction Score + MMR (Maximal Marginal Relevance) algorithm.

Friction Score formula:
    friction_score =
        0.4 * positive_relevance   (role/skill matches feature's target users)
      + 0.4 * conflict_score       (values conflict with the feature's threat axes)
      + 0.2 * influence_weight     (domain expert boost: legal=3x, privacy=2.5x)

MMR pass (after scoring):
    Iteratively select candidates that maximise:
        λ * friction_score - (1-λ) * max_similarity_to_already_selected
    This ensures seeds span different roles/poles rather than picking
    5 lawyers from the same department.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from tsc.models.personas import FinalPersona
from tsc.selection.models import TensionVector

logger = logging.getLogger(__name__)

# ─── Influence boost constants (mirrors eigenspace.py) ───────────────────────
_HIGH_IMPACT_DOMAINS: Dict[str, float] = {
    "legal":      3.0,
    "privacy":    2.5,
    "gdpr":       2.5,
    "compliance": 2.5,
    "hr":         2.0,
    "security":   2.0,
}

# Keywords that signal a persona is a *target user* of a feature (primary audience)
_PRIMARY_ROLE_SIGNALS = [
    "developer", "engineer", "programmer", "dev",
    "product", "designer", "analyst", "data scientist",
    "manager", "team lead", "tech lead",
    "enterprise", "startup", "founder", "executive", "ceo", "cio",
]

# Keywords that signal value conflict with typical "observability/control" features
# These help identify potential REPELLER seeds
_CONFLICT_SIGNALS = [
    "autonomy", "privacy", "freedom", "independent", "right to",
    "skeptic", "critic", "reluctant", "resistant", "oversight",
    "compliance", "legal", "gdpr", "regulation", "policy",
    "technical issue", "complex", "unclear", "delay", "friction",
]

# MMR trade-off: 1.0 = pure friction score, 0.0 = pure diversity
_MMR_LAMBDA = 0.65


class TensionAwarePersonaSelector:
    """
    Pre-selects the highest-friction, most-diverse persona seeds from a
    large DB candidate pool.

    Usage:
        selector = TensionAwarePersonaSelector()
        seeds = selector.select_seeds(candidates, tension_vector, target_seeds=8)
    """

    def select_seeds(
        self,
        candidates: List[FinalPersona],
        tension_vector: TensionVector,
        target_seeds: int = 8,
    ) -> List[FinalPersona]:
        """
        Main API.

        Args:
            candidates:     All personas pulled from the DB (can be 5–500)
            tension_vector: Phase 1 output — feature's named friction axes
            target_seeds:   How many seeds to hand to the GMM engine

        Returns:
            Subset of candidates, ordered by MMR rank.
        """
        if not candidates:
            return []
        if len(candidates) <= target_seeds:
            logger.info(
                "TensionAwareSelector: only %d candidates (≤ %d target) — returning all",
                len(candidates), target_seeds
            )
            return list(candidates)

        # Step 1: Compute Friction Scores
        scored: List[Tuple[FinalPersona, float]] = []
        for persona in candidates:
            score = self._friction_score(persona, tension_vector)
            scored.append((persona, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        log_top = scored[:10]
        logger.info(
            "TensionAwareSelector: %d candidates scored. Top 3 friction:",
            len(candidates)
        )
        for p, s in log_top[:3]:
            logger.info("  %-40s  friction=%.3f", p.name, s)

        # Step 2: MMR pass — greedy diversity selection
        selected = self._mmr_select(scored, target_seeds)

        logger.info(
            "TensionAwareSelector: %d candidates → %d diverse seeds",
            len(candidates), len(selected)
        )
        return selected

    # ──────────────────────────────────────────────────────────────────
    # Friction Score
    # ──────────────────────────────────────────────────────────────────

    def _friction_score(
        self,
        persona: FinalPersona,
        tension_vector: TensionVector,
    ) -> float:
        """
        Compute a [0, 1] score representing how much friction/value
        this persona would add to the simulation.
        """
        text = self._get_text(persona)

        positive_relevance = self._score_positive_relevance(text)
        conflict_score     = self._score_conflict(text, tension_vector)
        influence_weight   = self._compute_influence_weight(text)

        # Normalise influence_weight to [0,1] range (max boost is 3.0)
        influence_norm = min(1.0, (influence_weight - 1.0) / 2.0)

        raw = (
            0.40 * positive_relevance
            + 0.40 * conflict_score
            + 0.20 * influence_norm
        )
        return round(min(1.0, max(0.0, raw)), 4)

    def _score_positive_relevance(self, text: str) -> float:
        """How much this persona looks like a natural target/user of the feature."""
        matches = sum(1 for kw in _PRIMARY_ROLE_SIGNALS if kw in text)
        return min(1.0, matches * 0.2)

    def _score_conflict(
        self, text: str, tension_vector: TensionVector
    ) -> float:
        """
        How much this persona's values conflict with the feature's threat axes.
        A "Privacy Advocate" against a "Screen Recording" feature → high conflict.
        """
        # Base conflict from persona keywords
        kw_score = min(1.0, sum(1 for kw in _CONFLICT_SIGNALS if kw in text) * 0.15)

        # Amplify if the tension vector has strong threat axes (-ve dims)
        # Check for specific overlap between dimension keywords and persona text
        specific_conflict = 0.0
        for dim, val in tension_vector.dimensions.items():
            if val < -0.4:
                dim_kws = [dim.lower()] + dim.lower().split()
                if any(kw in text for kw in dim_kws):
                    specific_conflict += abs(val) * 0.5

        return min(1.0, kw_score + specific_conflict)

    def _compute_influence_weight(self, text: str) -> float:
        """Domain expert boost (mirrors eigenspace.py logic)."""
        weight = 1.0
        for domain, boost in _HIGH_IMPACT_DOMAINS.items():
            if domain in text and boost > weight:
                weight = boost
        return weight

    # ──────────────────────────────────────────────────────────────────
    # Maximal Marginal Relevance (MMR)
    # ──────────────────────────────────────────────────────────────────

    def _mmr_select(
        self,
        scored: List[Tuple[FinalPersona, float]],
        k: int,
    ) -> List[FinalPersona]:
        """
        Greedy MMR selection.
        Picks seeds that balance high friction_score with low role-similarity
        to already-selected seeds, ensuring diverse poles in the final set.
        """
        selected: List[FinalPersona] = []
        remaining: List[Tuple[FinalPersona, float]] = list(scored)

        while len(selected) < k and remaining:
            best_persona: Optional[FinalPersona] = None
            best_mmr = -math.inf

            for persona, f_score in remaining:
                # Diversity penalty: how similar to any already-selected?
                max_sim = max(
                    (self._role_similarity(persona, s) for s in selected),
                    default=0.0,
                )
                mmr = _MMR_LAMBDA * f_score - (1 - _MMR_LAMBDA) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_persona = persona

            if best_persona is None:
                break

            selected.append(best_persona)
            remaining = [(p, s) for (p, s) in remaining if p is not best_persona]

        return selected

    def _role_similarity(self, a: FinalPersona, b: FinalPersona) -> float:
        """
        Simple role-based similarity score [0, 1].
        Same role → 1.0; partially matching tokens → 0.5; different → 0.0.
        """
        role_a = {t.lower() for t in a.role.split()}
        role_b = {t.lower() for t in b.role.split()}
        overlap = len(role_a & role_b)
        union   = len(role_a | role_b) or 1
        return overlap / union

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_text(persona: FinalPersona) -> str:
        """Concatenate all scoreable text from a persona."""
        parts = [
            persona.role,
            getattr(persona, "title", ""),
            persona.psychological_profile.full_profile_text,
            " ".join(persona.psychological_profile.key_traits),
            " ".join(persona.psychological_profile.emotional_triggers.frustrated_by),
            " ".join(persona.psychological_profile.emotional_triggers.excited_by),
        ]
        return " ".join(p for p in parts if p).lower()
