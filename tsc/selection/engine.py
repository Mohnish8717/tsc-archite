"""
PersonaSelectionEngine — Main Orchestrator.

Coordinates all 6 phases of the Attractor-Repeller Persona Selection pipeline:

    Phase 1: Feature Tension Vector      (FeatureTensionAnalyzer)
    Phase 1.5: Tension-Aware Pre-Select  (Layer 3: TensionAwarePersonaSelector)
    Phase 2: Eigenspace Projection       (EigenspaceProjector)
    Phase 3: Priority Matrix Allocation  (70 / 20 / 10 split)
    Phase 4: GMM Stratified Expansion    (GMMSyntheticExpander)
    Phase 5: Epistemic Coverage Check    (EpistemicCoverageChecker)

Usage:
    engine = PersonaSelectionEngine(llm_client)
    expanded_personas, result = await engine.select(
        archetypes=layer3_personas,
        feature=feature_proposal,
        target_n=150,
    )
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from tsc.models.inputs import FeatureProposal
from tsc.models.personas import FinalPersona
from tsc.selection.coverage import EpistemicCoverageChecker
from tsc.selection.eigenspace import EigenspaceProjector
from tsc.selection.models import (
    PersonaCategory,
    PersonaPole,
    PoleType,
    PriorityMatrix,
    SelectionResult,
    TensionVector,
)
from tsc.selection.synthetic_expander import GMMSyntheticExpander
from tsc.selection.tension_vector import FeatureTensionAnalyzer

logger = logging.getLogger(__name__)

# ─── Strategy thresholds ─────────────────────────────────────────────
_EXPERT_JURY_MAX  = 20
_PROPORTIONAL_MAX = 100
# > 100 → full GMM with stratified sampling


class PersonaSelectionEngine:
    """
    Intelligent persona selection + synthetic expansion engine.

    Replaces naive random resampling in `profile_builder._resample_agents`
    with a 5-phase Attractor-Repeller + Priority Matrix pipeline.
    """

    def __init__(
        self,
        llm_client: Any,
        rng_seed: Optional[int] = None,
        priority_matrix: Optional[PriorityMatrix] = None,
    ) -> None:
        self._tension_analyzer = FeatureTensionAnalyzer(llm_client)
        self._projector        = EigenspaceProjector()
        self._expander         = GMMSyntheticExpander(llm_client, rng_seed=rng_seed)
        self._coverage         = EpistemicCoverageChecker()
        self._matrix           = priority_matrix or PriorityMatrix()
        logger.info(
            "PersonaSelectionEngine initialized (matrix: PRIMARY=%.0f%% / "
            "STAKEHOLDER=%.0f%% / ANTI=%.0f%%)",
            self._matrix.primary_weight * 100,
            self._matrix.stakeholder_weight * 100,
            self._matrix.anti_weight * 100,
        )

    async def select(
        self,
        archetypes: List[FinalPersona],
        feature: FeatureProposal,
        target_n: int,
    ) -> Tuple[List[FinalPersona], SelectionResult]:
        """
        Full 5-phase pipeline.

        Args:
            archetypes: List of FinalPersona from Layer 3 (3–8 expert archetypes)
            feature:    The FeatureProposal being evaluated
            target_n:   Number of simulation agents to produce

        Returns:
            (expanded_personas, SelectionResult)
        """
        t0 = time.time()
        logger.info(
            "PersonaSelectionEngine: %d archetypes → target %d agents",
            len(archetypes), target_n
        )

        # ── Phase 1: Feature Tension Vector ──────────────────────────
        logger.info("Phase 1: Extracting feature tension vector")
        tension_vector: TensionVector = await self._tension_analyzer.analyze(feature)

        # ── Phase 1.7: Epistemic Enrichment (formerly Phase 5) ────────
        # We enrich the seeds (archetypes) with missing domain fragments 
        # BEFORE projection so their pole assignment reflects the new context.
        logger.info("Phase 1.7: Epistemic enrichment of seeds")
        enriched_seeds, gaps = self._coverage.check(archetypes, [], tension_vector)
        archetypes = enriched_seeds

        # ── Phase 2: Eigenspace Projection + Category Assignment ─────
        logger.info("Phase 2: Projecting personas onto tension axis & classifying categories")
        poles: List[PersonaPole] = self._projector.project(archetypes, tension_vector)

        # ── Phase 3: Priority Matrix Allocation ──────────────────────
        strategy = self._choose_strategy(target_n)
        logger.info("Phase 3: Priority matrix allocation via '%s' strategy", strategy)
        per_category_quotas = self._compute_priority_quotas(poles, target_n, strategy)
        self._log_allocation(per_category_quotas)

        # ── Phase 4: Stratified GMM Expansion ────────────────────────
        logger.info("Phase 4: Stratified GMM expansion")
        expanded, expansion_metadata = await self._expander.expand_stratified(
            archetypes=archetypes,
            poles=poles,
            per_category_quotas=per_category_quotas,
        )

        # ── Build SelectionResult ─────────────────────────────────────

        # ── Build SelectionResult ─────────────────────────────────────
        pole_dist     = expansion_metadata.get("pole_distribution", {})
        cat_dist      = self._compute_category_distribution(expanded, poles)

        result = SelectionResult(
            tension_vector=tension_vector,
            poles=poles,
            epistemic_gaps=gaps,
            pole_distribution=pole_dist,
            category_distribution=cat_dist,
            priority_matrix=self._matrix,
            target_n=target_n,
            actual_n=len(expanded),
            strategy_used=strategy,
            expansion_metadata=expansion_metadata,
        )

        elapsed = time.time() - t0
        covered   = sum(1 for g in gaps if g.covered)
        uncovered = sum(1 for g in gaps if not g.covered)
        logger.info(
            "PersonaSelectionEngine complete: %d agents in %.1fs | "
            "poles: %s | categories: %s | epistemic gaps: %d covered, %d → Minority Voice",
            len(expanded), elapsed,
            ", ".join(f"{k}:{v:.0%}" for k, v in pole_dist.items()),
            ", ".join(f"{k}:{v:.0%}" for k, v in cat_dist.items()),
            covered, uncovered,
        )

        return expanded, result

    # ──────────────────────────────────────────────────────────────────
    # Phase 3: Priority Matrix Allocation
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _choose_strategy(target_n: int) -> str:
        if target_n <= _EXPERT_JURY_MAX:
            return "expert_jury"
        elif target_n <= _PROPORTIONAL_MAX:
            return "proportional"
        return "gmm"

    def _compute_priority_quotas(
        self,
        poles: List[PersonaPole],
        target_n: int,
        strategy: str,
    ) -> Dict[PersonaCategory, int]:
        """
        Compute per-category agent quotas honouring the 70/20/10 priority matrix.

        If the archetype pool doesn't include a STAKEHOLDER or ANTI persona,
        some PRIMARY slots absorb the difference — the real market distribution
        is respected as closely as possible.
        """
        categories_present = {p.category for p in poles}

        raw: Dict[PersonaCategory, float] = {
            PersonaCategory.PRIMARY:     self._matrix.primary_weight,
            PersonaCategory.STAKEHOLDER: self._matrix.stakeholder_weight,
            PersonaCategory.ANTI:        self._matrix.anti_weight,
        }

        # Redistribute weight for missing categories → PRIMARY absorbs
        missing_weight = sum(
            w for cat, w in raw.items() if cat not in categories_present
        )
        raw[PersonaCategory.PRIMARY] = min(
            1.0,
            raw[PersonaCategory.PRIMARY] + missing_weight,
        )

        # Convert weights to quota-int (floor, with remainder given to PRIMARY)
        quotas: Dict[PersonaCategory, int] = {}
        allocated = 0
        for cat in [PersonaCategory.STAKEHOLDER, PersonaCategory.ANTI]:
            if cat in categories_present:
                q = round(raw[cat] * target_n)
                quotas[cat] = q
                allocated += q
            else:
                quotas[cat] = 0

        quotas[PersonaCategory.PRIMARY] = target_n - allocated
        return quotas

    @staticmethod
    def _log_allocation(quotas: Dict[PersonaCategory, int]) -> None:
        total = sum(quotas.values()) or 1
        for cat, n in quotas.items():
            logger.info(
                "  %-12s  %3d agents  (%.0f%%)",
                cat.value, n, 100 * n / total
            )

    # ──────────────────────────────────────────────────────────────────
    # Diagnostics helper
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_category_distribution(
        personas: List[FinalPersona],
        poles: List[PersonaPole],
    ) -> Dict[str, float]:
        """Compute % distribution by PersonaCategory in the finally expanded list."""
        pole_map = {p.persona_name: p for p in poles}
        counts: Dict[str, int] = {cat.value: 0 for cat in PersonaCategory}
        for persona in personas:
            base_name = persona.name.split(" [")[0]
            pole_obj = pole_map.get(base_name) or pole_map.get(persona.name)
            cat = pole_obj.category.value if pole_obj else PersonaCategory.PRIMARY.value
            counts[cat] += 1
        total = len(personas) or 1
        return {k: round(v / total, 3) for k, v in counts.items()}

