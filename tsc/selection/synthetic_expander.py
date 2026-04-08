"""
Phase 4: GMM Synthetic Expander.

Scales a small set of FinalPersona "archetypes" (3–8) to any target N
by sampling from Gaussian distributions centred on each archetype cluster.

Key design decisions:
- Critics (REPELLER) → low noise sigma (predictable, consistent)
- Swing voters       → high noise sigma (unpredictable, high variance)
- Attractors         → medium noise sigma
- Core values are NEVER jittered — only surface demographics vary
- Each synthetic clone gets a unique name + modified bio fragment
"""
from __future__ import annotations

import copy
import logging
import random
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from tsc.models.personas import FinalPersona
from tsc.selection.models import PersonaPole, PoleType, SelectionResult

logger = logging.getLogger(__name__)


# ─── Variance by pole ────────────────────────────────────────────────
_SIGMA: Dict[PoleType, float] = {
    PoleType.ATTRACTOR: 0.35,
    PoleType.REPELLER:  0.15,   # Critics are predictable
    PoleType.SWING:     0.55,   # Swing voters are unpredictable
}

# ─── Internal frames (for INTERNAL personas — domain-specific to org dynamics) ──
_INTERNAL_FRAMES: List[Dict[str, Any]] = [
    {"label": "Automation-Skeptic",      "trait_add": "contrarian",         "trait_remove": "collaborative",      "stance_nudge": -0.25, "bio_prefix": "After watching initiatives fail by trusting automation blindly, {name} now demands explainability and human checkpoints before endorsing any autonomous decision-making in their domain."},
    {"label": "Velocity-Pragmatist",     "trait_add": "pragmatic",          "trait_remove": "detail-oriented",   "stance_nudge": +0.30, "bio_prefix": "{name} has delivered 40+ initiatives in the last 18 months. Their primary filter for any new capability: does it remove friction from the critical path? If it does, they will champion it internally."},
    {"label": "Privacy-Hawk",            "trait_add": "risk-conscious",     "trait_remove": "analytical",        "stance_nudge": -0.35, "bio_prefix": "{name} tracks data legislation closely and flags any system that creates new observability of individual or organisational behaviour. They require a clear data governance policy before endorsing anything."},
    {"label": "Alert-Fatigue-Sufferer",  "trait_add": "skeptical",          "trait_remove": "decisive",          "stance_nudge": -0.20, "bio_prefix": "{name} vividly recalls a period when a poorly-tuned system generated so many false signals it paralysed operations. They resist any new capability without hard precision evidence."},
    {"label": "Knowledge-Transfer-Guardian", "trait_add": "mentorship-driven", "trait_remove": "efficiency-focused", "stance_nudge": -0.15, "bio_prefix": "{name} invests heavily in growing the team's craft. They worry that automating key workflows will eliminate the learning opportunities that happen in human-to-human collaboration."},
    {"label": "Compliance-First",        "trait_add": "regulatory-minded",  "trait_remove": "innovative",        "stance_nudge": -0.30, "bio_prefix": "{name} sits at the intersection of operations and legal. Any capability that makes autonomous decisions in regulated contexts triggers an immediate liability review in their mind."},
    {"label": "Early-Adopter-Optimist",  "trait_add": "innovative",         "trait_remove": "risk-conscious",    "stance_nudge": +0.40, "bio_prefix": "{name} has already run internal pilots of emerging capabilities and presented results at leadership reviews. They see the proposed feature as the next obvious step and are ready to scale it immediately."},
    {"label": "Cost-Reduction-Advocate", "trait_add": "budget-conscious",   "trait_remove": "process-oriented",  "stance_nudge": +0.35, "bio_prefix": "{name} reports to leadership with a mandate to reduce operational overhead. The proposed capability maps directly to measurable cost savings, and they are motivated to make the numbers work."},
]

# ─── Market frames (for EXTERNAL personas — domain-agnostic buyer archetypes) ──
# Each frame represents a universal buyer psychology, applicable to any product,
# service, policy, or strategic initiative across any industry.
_MARKET_FRAMES: List[Dict[str, Any]] = [
    {
        "label": "Enterprise-Risk-Gatekeeper",
        "trait_add": "governance-focused",
        "trait_remove": "innovative",
        "stance_nudge": -0.30,
        "bio_prefix": "{name} operates in a large organisation where every new vendor or capability must pass a multi-stage approval process involving legal, security, and finance. Their default posture is thorough due diligence before any commitment.",
        "market_fields": {"company_size_band": "enterprise", "buyer_role": "influencer", "pricing_sensitivity": "low", "sales_cycle_weeks": 20, "regulatory_burden": "moderate"},
        "journey_fields": {"awareness_channel": "analyst-report", "roi_threshold_months": 18, "willingness_to_pay_band": "high"},
    },
    {
        "label": "Startup-Speed-Champion",
        "trait_add": "pragmatic",
        "trait_remove": "process-oriented",
        "stance_nudge": +0.35,
        "bio_prefix": "{name} runs a lean operation where speed-to-value is the only filter that matters. They evaluate new capabilities in days, not months, and will adopt anything that demonstrably moves the needle before competitors do.",
        "market_fields": {"company_size_band": "small", "buyer_role": "decision-maker", "pricing_sensitivity": "high", "sales_cycle_weeks": 2, "regulatory_burden": "none"},
        "journey_fields": {"awareness_channel": "peer-recommendation", "roi_threshold_months": 3, "willingness_to_pay_band": "low"},
    },
    {
        "label": "Cost-ROI-Optimizer",
        "trait_add": "budget-conscious",
        "trait_remove": "risk-conscious",
        "stance_nudge": +0.25,
        "bio_prefix": "{name} treats every solution decision as an investment analysis. They champion adoption only when the payback period is clear, quantified, and under 12 months — and they know how to build a business case that will pass CFO scrutiny.",
        "market_fields": {"company_size_band": "mid-market", "buyer_role": "champion", "pricing_sensitivity": "medium", "sales_cycle_weeks": 8, "regulatory_burden": "light"},
        "journey_fields": {"awareness_channel": "vendor-outreach", "roi_threshold_months": 9, "willingness_to_pay_band": "moderate"},
    },
    {
        "label": "Compliance-First-Buyer",
        "trait_add": "regulatory-minded",
        "trait_remove": "decisive",
        "stance_nudge": -0.25,
        "bio_prefix": "{name} operates in a sector where every new capability must satisfy audit, legal, and regulatory requirements before deployment. Their willingness to pay is high — but only for solutions that come with the compliance guarantees they need.",
        "market_fields": {"company_size_band": "enterprise", "buyer_role": "influencer", "pricing_sensitivity": "low", "sales_cycle_weeks": 24, "regulatory_burden": "heavy"},
        "journey_fields": {"awareness_channel": "internal-mandate", "roi_threshold_months": 24, "willingness_to_pay_band": "very-high"},
    },
    {
        "label": "Late-Majority-Skeptic",
        "trait_add": "skeptical",
        "trait_remove": "innovative",
        "stance_nudge": -0.20,
        "bio_prefix": "{name} does not move until a solution is proven by at least three reference organisations they respect. They are not opposed to the capability in principle — they simply need the market to validate it before they are willing to stake their reputation on it.",
        "market_fields": {"company_size_band": "mid-market", "buyer_role": "decision-maker", "pricing_sensitivity": "medium", "sales_cycle_weeks": 16, "regulatory_burden": "light"},
        "journey_fields": {"awareness_channel": "social-proof", "roi_threshold_months": 12, "willingness_to_pay_band": "moderate"},
    },
    {
        "label": "Early-Adopter-Champion",
        "trait_add": "innovative",
        "trait_remove": "risk-conscious",
        "stance_nudge": +0.40,
        "bio_prefix": "{name} actively scouts for capabilities that can give their organisation a competitive edge before the rest of the market catches on. They build internal business cases, run pilots proactively, and are willing to work through early rough edges.",
        "market_fields": {"company_size_band": "mid-market", "buyer_role": "champion", "pricing_sensitivity": "low", "sales_cycle_weeks": 4, "regulatory_burden": "light"},
        "journey_fields": {"awareness_channel": "conference", "roi_threshold_months": 6, "willingness_to_pay_band": "high"},
    },
    {
        "label": "SMB-Budget-Constrained",
        "trait_add": "cost-sensitive",
        "trait_remove": "strategic",
        "stance_nudge": -0.15,
        "bio_prefix": "{name} leads a small team where every discretionary spend requires personal sign-off. The capability appeals to them in theory, but they need a clear, affordable entry point and a short path to value before they can justify it to ownership.",
        "market_fields": {"company_size_band": "small", "buyer_role": "decision-maker", "pricing_sensitivity": "high", "sales_cycle_weeks": 3, "regulatory_burden": "none"},
        "journey_fields": {"awareness_channel": "organic-search", "roi_threshold_months": 4, "willingness_to_pay_band": "low"},
    },
    {
        "label": "Influencer-Without-Authority",
        "trait_add": "collaborative",
        "trait_remove": "decisive",
        "stance_nudge": 0.00,
        "bio_prefix": "{name} is the internal champion who sees the value most clearly — but purchasing decisions sit one or two levels above them. Their effectiveness depends on their ability to translate operational benefits into language that resonates with budget-holders they don't always have direct access to.",
        "market_fields": {"company_size_band": "enterprise", "buyer_role": "influencer", "pricing_sensitivity": "medium", "sales_cycle_weeks": 12, "regulatory_burden": "light"},
        "journey_fields": {"awareness_channel": "peer-recommendation", "roi_threshold_months": 12, "willingness_to_pay_band": "moderate"},
    },
]

# ─── Seniority / location pools (used for name suffix only) ──────────
_SENIORITY_LEVELS = ["Intern", "Junior", "Mid-Level", "Senior", "Principal", "Staff"]
_LOCATIONS = ["India", "Germany", "USA", "Brazil", "Japan", "Canada", "UK", "Australia"]


class GMMSyntheticExpander:
    """
    Phase 4 of PersonaSelectionEngine.

    API:
        expand(archetypes, poles, target_n) → List[FinalPersona]

    The returned list is always exactly target_n long (or as close as
    rounding allows).  Original archetypes are always included.
    """

    CLONE_NARRATIVE_PROMPT = """You are an expert character writer. You are creating a unique variation of a base persona for a simulation.
    Base Persona Name: {base_name}
    New Clone Name: {clone_name}
    New Trait Added: {trait_add}
    Divergence Frame: {frame_label}

    Rewrite the following core psychological dimensions for this new clone to aggressively reflect their new trait and divergence frame. This must feel radically distinct from the base persona:
    1. VIVID SCENE (100 words): A specific moment in their daily work dealing with a problem.
    2. EMOTIONAL TRIGGERS & VALUES (150 words): What excites, frustrates, and scares them; their deepest professional biases.
    3. DECISION PATTERNS (100 words): How they evaluate new tools, their threshold for proof, and risk tolerance.
    4. SIGNATURE QUOTE (60 words): A verbatim quote revealing their core bias.

    Return ONLY the specific text. Format strictly as:
    VIVID SCENE:
    ...
    EMOTIONAL TRIGGERS & VALUES:
    ...
    DECISION PATTERNS:
    ...
    SIGNATURE QUOTE:
    ...
    """

    def __init__(self, llm_client: Any, rng_seed: Optional[int] = None) -> None:
        self._llm = llm_client
        self._rng = random.Random(rng_seed)

    async def expand(
        self,
        archetypes: List[FinalPersona],
        poles: List[PersonaPole],
        target_n: int,
    ) -> Tuple[List[FinalPersona], Dict[str, Any]]:
        """
        Returns (expanded_personas, metadata_dict).
        metadata_dict contains diagnostic info for SelectionResult.
        """
        if not archetypes:
            return [], {}

        # Build pole index for quick lookup
        pole_map: Dict[str, PersonaPole] = {p.persona_name: p for p in poles}

        # Decide per-pole quota based on target_n
        quotas = self._compute_quotas(archetypes, pole_map, target_n)

        result: List[FinalPersona] = list(archetypes)   # always keep originals
        synthetic_count = 0
        tasks = []

        for archetype in archetypes:
            pole_obj = pole_map.get(archetype.name)
            pole_type = pole_obj.pole if pole_obj else PoleType.SWING
            sigma = _SIGMA[pole_type]
            quota = quotas.get(archetype.name, 0)

            for i in range(quota):
                tasks.append(self._create_and_jitter_clone(archetype, i, sigma, pole_obj))
                
        if tasks:
            clones = await asyncio.gather(*tasks)
            result.extend(clones)
            synthetic_count += len(clones)

        # Trim/pad to exact target_n
        result = await self._trim_or_pad(result, target_n, archetypes, pole_map)

        pole_dist = self._compute_pole_distribution(result, pole_map)
        metadata = {
            "synthetic_count": synthetic_count,
            "archetype_count": len(archetypes),
            "total": len(result),
            "pole_distribution": pole_dist,
        }
        logger.info(
            "Phase 4: Expanded %d → %d agents. Poles: %s",
            len(archetypes), len(result),
            ", ".join(f"{k}:{v:.0%}" for k, v in pole_dist.items())
        )
        return result, metadata

    async def expand_stratified(
        self,
        archetypes: List[FinalPersona],
        poles: List[PersonaPole],
        per_category_quotas: Dict[Any, int],
    ) -> Tuple[List[FinalPersona], Dict[str, Any]]:
        """
        Stratified expansion honouring a per-category quota dict.

        Each category (PRIMARY / STAKEHOLDER / ANTI) is expanded independently
        from its own archetype pool, preserving the 70/20/10 priority split. Creates clones asynchronously.

        Args:
            archetypes: All persona archetypes from Layer 3
            poles: PersonaPole list (includes .category from eigenspace projector)
            per_category_quotas: {PersonaCategory → target agent count}

        Returns:
            (expanded_personas, metadata_dict)
        """
        from tsc.selection.models import PersonaCategory

        if not archetypes:
            return [], {}

        pole_map: Dict[str, PersonaPole] = {p.persona_name: p for p in poles}

        # Group archetypes by category
        category_archetypes: Dict[str, List[FinalPersona]] = {
            cat.value: [] for cat in PersonaCategory
        }
        for arch in archetypes:
            pole_obj = pole_map.get(arch.name)
            cat = pole_obj.category.value if pole_obj else PersonaCategory.PRIMARY.value
            category_archetypes[cat].append(arch)

        # Fallback: if a category has no archetypes, pull from PRIMARY
        primary_pool = category_archetypes[PersonaCategory.PRIMARY.value] or archetypes

        result: List[FinalPersona] = list(archetypes)   # always keep originals
        synthetic_count = 0

        import asyncio
        tasks = []

        for cat, quota in per_category_quotas.items():
            cat_key = cat.value if hasattr(cat, "value") else str(cat)
            pool = category_archetypes.get(cat_key) or primary_pool
            need = max(0, quota - len(pool))  # subtract the originals already counted

            for i in range(need):
                src = self._rng.choice(pool)
                pole_obj = pole_map.get(src.name)
                pole_type = pole_obj.pole if pole_obj else PoleType.SWING
                sigma = _SIGMA[pole_type]
                force_anti = (cat_key == "ANTI")
                tasks.append(self._create_and_jitter_clone(src, i, sigma, pole_obj, force_anti))
                
        if tasks:
            clones = await asyncio.gather(*tasks)
            result.extend(clones)
            synthetic_count += len(clones)

        # Final trim/pad to total
        total_target = sum(per_category_quotas.values()) if per_category_quotas else len(result)
        result = await self._trim_or_pad(result, total_target, archetypes, pole_map)

        pole_dist = self._compute_pole_distribution(result, pole_map)
        metadata = {
            "synthetic_count": synthetic_count,
            "archetype_count": len(archetypes),
            "total": len(result),
            "pole_distribution": pole_dist,
            "strategy": "stratified",
            "category_quotas": {
                (k.value if hasattr(k, "value") else str(k)): v
                for k, v in per_category_quotas.items()
            },
        }
        logger.info(
            "Phase 4 (stratified): Expanded %d → %d agents (%d synthetic). Poles: %s",
            len(archetypes), len(result), synthetic_count,
            ", ".join(f"{k}:{v:.0%}" for k, v in pole_dist.items())
        )
        return result, metadata

    # ──────────────────────────────────────────────────────────────────
    # Quota Computation
    # ──────────────────────────────────────────────────────────────────

    def _compute_quotas(
        self,
        archetypes: List[FinalPersona],
        pole_map: Dict[str, PersonaPole],
        target_n: int,
    ) -> Dict[str, int]:
        """
        Compute how many synthetic clones to generate per archetype.
        Swing voters get a higher share to maximise behavioral uncertainty.
        """
        weights: Dict[str, float] = {}
        for a in archetypes:
            pole_obj = pole_map.get(a.name)
            pole_type = pole_obj.pole if pole_obj else PoleType.SWING
            # Swing → weight 2x, Critics → 1x, Attractors → 1.5x
            weights[a.name] = {
                PoleType.SWING:     2.0,
                PoleType.ATTRACTOR: 1.5,
                PoleType.REPELLER:  1.0,
            }[pole_type]

        total_weight = sum(weights.values())
        need = max(0, target_n - len(archetypes))
        quotas: Dict[str, int] = {}
        allocated = 0
        arc_list = list(archetypes)

        for i, a in enumerate(arc_list):
            if i == len(arc_list) - 1:
                # Last archetype gets remainder
                quotas[a.name] = need - allocated
            else:
                q = round((weights[a.name] / total_weight) * need)
                quotas[a.name] = q
                allocated += q

        return quotas

    async def _create_and_jitter_clone(self, src, i, sigma, pole_obj, force_anti=False):
        clone = await self._jitter_clone(src, i, sigma, force_anti)
        if pole_obj and pole_obj.influence_weight > 1.0:
            clone.influence_strength = float(min(0.95, clone.influence_strength * pole_obj.influence_weight))
        return clone

    # ──────────────────────────────────────────────────────────────────
    # Jitter Clone
    # ──────────────────────────────────────────────────────────────────

    async def _jitter_clone(
        self,
        source: FinalPersona,
        clone_index: int,
        sigma: float,
        force_anti: bool = False
    ) -> FinalPersona:
        """
        Deep-copy source and produce a PSYCHOLOGICALLY DISTINCT clone.

        For INTERNAL personas:
          - Picks from _INTERNAL_FRAMES (org-dynamics worldview)
          - Mutates key_traits, prepends bio divergence sentence

        For EXTERNAL personas:
          - Picks from _MARKET_FRAMES (domain-agnostic buyer archetype)
          - Mutates key_traits, prepends buyer-journey bio
          - Populates clone.market_context + clone.buyer_journey with
            Gaussian-jittered values derived from the frame defaults
        """
        from tsc.models.personas import MarketContext, BuyerJourney

        clone = copy.deepcopy(source)
        is_external = getattr(source, "persona_type", "INTERNAL") == "EXTERNAL"

        # Select the appropriate frame pool
        frame_pool = _MARKET_FRAMES if is_external else _INTERNAL_FRAMES
        frame = frame_pool[clone_index % len(frame_pool)]

        # Pick surface attributes for the name suffix
        seniority = self._rng.choice(_SENIORITY_LEVELS)
        location  = self._rng.choice(_LOCATIONS)

        # Unique name
        clone.name = f"{source.name} [{seniority}, {location}]_{clone_index:03d}"

        # Vary influence_strength and receptiveness with Gaussian noise
        clone.influence_strength = float(max(0.1, min(0.9,
            source.influence_strength + self._rng.gauss(0, sigma * 0.3)
        )))
        clone.receptiveness = float(max(0.1, min(0.9,
            source.receptiveness + self._rng.gauss(0, sigma * 0.3)
        )))

        # ── Trait Mutation ────────────────────────────────────────────
        traits = list(clone.psychological_profile.key_traits)
        trait_to_remove = frame["trait_remove"]
        trait_to_add    = frame["trait_add"]
        traits = [t for t in traits if t != trait_to_remove]
        if trait_to_add not in traits:
            traits.append(trait_to_add)
        clone.psychological_profile.key_traits = traits

        # ── Stance Jitter (SWING clones only) & ANTI Hostile Mutation ───
        if force_anti or (sigma >= 0.5 and self._rng.random() < 0.30):
            clone.psychological_profile.predicted_stance.prediction = "BEARISH"

        # ── Bio Divergence (NARRATIVE MUTATION) ───────────────────────
        bio_prefix = frame["bio_prefix"].format(name=source.name.split()[0])
        frame_tag = (
            f"\n[Frame: {frame['label']} | {seniority} {source.role} | {location}]\n"
        )
        
        anti_addon = ""
        if force_anti:
            anti_addon = "\n\nCRITICAL INSTRUCTION: This clone is part of the 'ANTI' quota (a minority voice). You MUST portray them as a hostile critic or severe skeptic. Their emotional triggers and decision patterns should explicitly reflect extreme caution, opposition, or harsh scrutiny of the proposed feature."

        # We rewrite the first two sections (Scene and Quote) to be unique via fast LLM
        prompt = self.CLONE_NARRATIVE_PROMPT.format(
            base_name=source.name,
            clone_name=clone.name,
            trait_add=trait_to_add,
            frame_label=frame['label']
        ) + anti_addon
        original_bio = clone.psychological_profile.full_profile_text
        try:
            llm_response = await self._llm.generate(
                system_prompt="You are an expert persona generator.",
                user_prompt=prompt,
                temperature=0.4,
                max_tokens=600
            )
            clone.psychological_profile.full_profile_text = (
                f"{frame_tag}{bio_prefix}\n\n"
                f"### UNIQUE DIVERGENCE ###\n{llm_response}\n\n"
                f"### BASE PSYCHOLOGY (Inherited) ###\n{original_bio}"
            )
        except Exception as e:
            logger.warning("Narrative mutation failed for clone %s: %s", clone.name, e)
            clone.psychological_profile.full_profile_text = (
                f"{frame_tag}{bio_prefix}\n\n" + original_bio
            )

        # ── Market Context (EXTERNAL only) ────────────────────────────
        if is_external and "market_fields" in frame:
            mf = frame["market_fields"]
            jf = frame.get("journey_fields", {})

            # Jitter numeric fields so each clone is distinct
            def _jitter_int(base: int, lo: int, hi: int) -> int:
                return int(max(lo, min(hi, base + int(self._rng.gauss(0, sigma * base * 0.3)))))

            clone.market_context = MarketContext(
                company_size_band=mf.get("company_size_band", "mid-market"),
                buyer_role=mf.get("buyer_role", "influencer"),
                annual_solution_budget_usd=_jitter_int(
                    getattr(source.market_context, "annual_solution_budget_usd", 50_000)
                    if source.market_context else 50_000,
                    5_000, 5_000_000
                ),
                pricing_sensitivity=mf.get("pricing_sensitivity", "medium"),
                sales_cycle_weeks=_jitter_int(mf.get("sales_cycle_weeks", 8), 1, 52),
                deployment_preference=getattr(source.market_context, "deployment_preference", "cloud")
                    if source.market_context else "cloud",
                industry_vertical=getattr(source.market_context, "industry_vertical", "technology")
                    if source.market_context else "technology",
                regulatory_burden=mf.get("regulatory_burden", "light"),
            )

            clone.buyer_journey = BuyerJourney(
                awareness_channel=jf.get("awareness_channel", "peer-recommendation"),
                evaluation_trigger=getattr(source.buyer_journey, "evaluation_trigger", "")
                    if source.buyer_journey else "",
                key_proof_points=list(getattr(source.buyer_journey, "key_proof_points", []))
                    if source.buyer_journey else [],
                deal_breakers=list(getattr(source.buyer_journey, "deal_breakers", []))
                    if source.buyer_journey else [],
                success_metric=getattr(source.buyer_journey, "success_metric", "")
                    if source.buyer_journey else "",
                roi_threshold_months=_jitter_int(jf.get("roi_threshold_months", 12), 1, 36),
                willingness_to_pay_band=jf.get("willingness_to_pay_band", "moderate"),
            )

        return clone

    # ──────────────────────────────────────────────────────────────────
    # Trim / Pad
    # ──────────────────────────────────────────────────────────────────

    async def _trim_or_pad(
        self,
        result: List[FinalPersona],
        target_n: int,
        archetypes: List[FinalPersona],
        pole_map: Dict[str, PersonaPole],
    ) -> List[FinalPersona]:
        if len(result) > target_n:
            # Keep all archetypes; trim synthetic clones
            originals = list(archetypes)
            synthetics = [p for p in result if p not in archetypes]
            self._rng.shuffle(synthetics)
            return originals + synthetics[:target_n - len(originals)]
        elif len(result) < target_n:
            # Pad with random clones from any archetype
            pad_tasks = []
            pad_count = target_n - len(result)
            for i in range(pad_count):
                src = self._rng.choice(archetypes)
                pole_obj = pole_map.get(src.name)
                sigma = _SIGMA[pole_obj.pole if pole_obj else PoleType.SWING]
                pad_tasks.append(self._jitter_clone(src, len(result) + i, sigma))
            if pad_tasks:
                clones = await asyncio.gather(*pad_tasks)
                result.extend(clones)
        return result

    # ──────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────

    def _compute_pole_distribution(
        self,
        personas: List[FinalPersona],
        pole_map: Dict[str, PersonaPole],
    ) -> Dict[str, float]:
        """Compute % of each pole type in the final expanded agent list."""
        counts = {pt.value: 0 for pt in PoleType}
        for p in personas:
            # Synthetic clones inherit the source archetype's pole
            base_name = p.name.split(" [")[0]   # strip synthetic suffix
            pole_obj = pole_map.get(base_name) or pole_map.get(p.name)
            pole_type = pole_obj.pole if pole_obj else PoleType.SWING
            counts[pole_type.value] += 1
        total = len(personas) or 1
        return {k: round(v / total, 3) for k, v in counts.items()}
