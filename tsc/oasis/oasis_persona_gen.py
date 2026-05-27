"""
OASIS User Persona Generator
==============================

Creates realistic, demographically diverse product-user personas for
OASIS behavioral social simulation — grounded in actual customer data.

Design Principles (from research):
  1. Structured attributes over narratives — compact JSON profiles, not 2500-word essays
  2. Customer interview grounding via Hindsight recall
  3. LLM-inferred segment distribution from actual customer data
  4. Big Five (OCEAN) personality traits over MBTI for behavioral prediction
  5. Direct OASISAgentProfile output — no conversion step
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Optional

from tsc.llm.base import LLMClient
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.oasis.models import OASISAgentProfile

logger = logging.getLogger(__name__)



# ── System Prompts (v2.0 — Structured CoT + Coverage Framework) ──────────────
#
# Design decisions (refs: prompt-patterns.md, system-prompts.md, context-management.md):
#
# 1. SEGMENT_INFERENCE_SYSTEM:
#    - Changed from Zero-shot → Structured CoT (steps 1-4).
#    - Step 1: Product-category reasoning FIRST (before looking at data).
#      This seeds the canonical user archetypes for the product class so the LLM
#      knows the full space to cover, not just what appears in support tickets.
#    - Step 2: Data-grounded observation (pain points, quotes, frequencies).
#    - Step 3: Coverage completeness check — the LLM must explicitly verify that
#      it has a segment for EVERY lifecycle stage (acquisition/activation/retention/churn).
#    - Step 4: Proportion calibration using the base-rate prior:
#      ~70% of real users never file support tickets (silent majority).
#    - Removed the over-restrictive "do NOT hallucinate mainstream users" rule and
#      replaced it with a statistical inference allowance.
#
# 2. PERSONA_GEN_SYSTEM:
#    - Added product-type-specific behavioral priors via XML-tagged context buckets.
#    - Added within-segment diversity mandates with explicit axes (seniority, tenure,
#      emotional state, workaround strategy).
#    - Added few-shot anchor examples per role archetype.
#    - Added recency-biased constraint reminder at end of prompt (primacy + recency).

SEGMENT_INFERENCE_SYSTEM = """\
<identity>
You are a senior UX researcher and quantitative social scientist.
You specialise in constructing complete, statistically representative user populations
for social simulation — NOT just cataloguing who complained in support tickets.
</identity>

<task>
Analyse the provided customer data and company context to identify a COMPLETE set of
user segments that covers the ENTIRE real user base of this product.
</task>

<reasoning_steps>
## Step 1 — Product Category Inference (do this BEFORE reading the data)

Identify what type of product this is and enumerate the CANONICAL user archetypes
that ALWAYS exist for this product category, regardless of what the data says.

Examples:
- B2B SaaS dev-tool → always has: Individual Contributor, Team Lead/Manager, Buyer/Admin, Evaluator
- Consumer mobile app → always has: Daily Active, Weekly Casual, Lapsed/Dormant, New/Onboarding
- Marketplace → always has: Supply-side provider, Demand-side buyer, Power transactor, Lurker

List the canonical archetypes for THIS product. These are your mandatory coverage skeleton.

## Step 2 — Data-Grounded Observation

Now read the customer data. For each canonical archetype, find evidence (or note its absence).
Extract:
- Real pain points using the customer's exact vocabulary
- Frequency signals (how many data points per segment?)
- Emotional signals (frustration vs satisfaction vs indifference)
- Behavioural patterns (usage frequency, feature depth, workarounds)

## Step 2b — Competitive Exit Mapping

For each segment, identify:
- Primary competitor they would evaluate if churning (from data or inference)
- Specific capability gap that would trigger evaluation
- Switching cost band: low (SaaS swap) | medium (some integration) | high (deep platform lock-in)
Add "competitive_exit_vector" to each segment object.

## Step 3 — Coverage Completeness Check

Before generating your final output, verify you have AT LEAST ONE segment for each:
- [ ] Acquisition/Evaluation stage users (haven't committed yet)
- [ ] New/Onboarding users (<90 days)
- [ ] Core/Retained users (active, generally satisfied)
- [ ] Power/Advanced users (deep integrators, high influence)
- [ ] At-Risk/Churning users (declining engagement, high frustration)
- [ ] Buyer/Decision-maker (may differ from end-user in B2B products)

If your data-only segments missed any of these, infer the missing segment from
statistical base rates and product category knowledge. Mark inferred segments
with "data_basis": "inferred_from_base_rate".

## Step 4 — Proportion Calibration with Revenue Weighting

Apply DUAL calibration — report BOTH:
1. Headcount proportion: ticket_frequency × silence_adjustment (×4–7x for satisfied users)
2. Revenue proportion: estimate ARR contribution per segment
   - Power/Enterprise users: typically 60–80% of ARR at 10–20% of headcount (Pareto)
   - Churning users: flag 30-day revenue-at-risk
Simulations MUST over-sample high-ARR segments so business-critical signals
are not statistically diluted by headcount majority.
</reasoning_steps>

<constraints>
- Segments must collectively sum to proportion = 1.0
- Minimum 4 segments, maximum 7 segments
- Each segment MUST have at least one direct data citation OR a "data_basis" explanation
- Each segment MUST include "trend_direction": "growing|stable|declining" based on
  frequency of recent vs. historical data mentions
- Each segment MUST include "competitive_exit_vector" identifying the primary alternative
- Each segment MUST include "revenue_proportion" (estimated ARR share, not headcount share)
- Do NOT conflate "no data about this segment" with "this segment doesn't exist"
- Statistical inference about the silent majority IS valid; pure hallucination is not
</constraints>

<output_format>
Return valid JSON only — no prose, no markdown fences:
{
  "product_category": "B2B SaaS / Consumer App / Marketplace / etc.",
  "canonical_archetypes_identified": ["archetype_1", "archetype_2"],
  "coverage_check": {
    "acquisition": true,
    "onboarding": true,
    "retained_core": true,
    "power_user": true,
    "at_risk": true,
    "buyer_persona": true
  },
  "segments": [
    {
      "segment_name": "Power Users — Heavy Integrators",
      "data_basis": "direct_observation",
      "data_citations": ["Quote or ticket pattern that grounds this segment"],
      "proportion": 0.12,
      "description": "Users who deeply integrate the product into their daily workflow",
      "lifecycle_stage": "retained_core / power_user",
      "typical_demographics": {
        "age_range": [28, 45],
        "occupations": ["Senior Engineer", "Tech Lead", "Architect"],
        "tech_literacy": "high",
        "tenure_months_range": [12, 48]
      },
      "behavioral_traits": {
        "usage_frequency": "daily",
        "feature_depth": "deep",
        "support_tendency": "self-serve",
        "price_sensitivity": "low",
        "churn_risk": "low"
      },
      "emotional_profile": {
        "dominant_emotion": "pride / frustration / anxiety / indifference",
        "satisfaction_range": [0.65, 0.85],
        "nps_likelihood": "promoter"
      },
      "pain_points": ["Exact vocabulary from customer data"],
      "desired_outcomes": ["Exact vocabulary from customer data"],
      "representative_quotes": ["Direct verbatim quotes if available"],
      "workarounds": ["Specific workarounds this segment uses"]
    }
  ]
}
</output_format>

<critical_reminder>
Your output is the seed for a social simulation that will inform real product decisions.
A missing segment = a missing voice in the boardroom. Cover the full population.
</critical_reminder>
"""

PERSONA_GEN_SYSTEM = """\
<identity>
You are a behavioural scientist and social simulation architect.
Your job is to instantiate individual, psychologically distinct human beings
who will be placed inside a social platform simulation to generate authentic
product sentiment data.
</identity>

<task>
Generate exactly {count} diverse personas for the specified user segment.
Each persona must be a DISTINCT individual — not a template clone with names swapped.
</task>

<diversity_mandate>
You MUST maximise variation across these axes within the segment:
1. Seniority & experience level (junior, mid, senior, veteran)
2. Tenure with the product (1 month → 5+ years)
3. Emotional state today (enthusiastic, neutral, quietly frustrated, actively churning)
4. Primary use-case (the segment uses the product for different concrete tasks)
5. Workaround strategy (some improvise, some pay for add-ons, some just suffer)
6. Communication style (blunt/direct, verbose/analytical, passive, emotionally expressive)
7. Participation mode (active-commenter, reactive-replier, lurker, thread-starter).
   Lurkers MUST have their [BEHAVIORAL RULES] specify the EXACT condition that breaks
   their silence — e.g., "only speaks when directly challenged or sees a factual error."

If generating N personas, imagine N DIFFERENT real people — not clones of one archetype.
</diversity_mandate>

<product_type_behavioral_priors>
Apply these priors based on product category:

FOR B2B SaaS / Dev Tools:
- Individual contributors care about workflow speed & API quality
- Managers/leads care about team velocity & reporting
- Buyers/admins care about security, compliance & cost per seat
- New users are overwhelmed by onboarding; power users resent feature removal

FOR CONSUMER APPS (social, productivity, lifestyle):
- Daily actives are sensitive to UI changes and notification patterns
- Weekly casuals have low switching cost — any friction triggers churn
- Lapsed users have a specific "moment of disengagement" to articulate

FOR MARKETPLACES:
- Supply-side cares about demand quality and payout reliability
- Demand-side cares about search quality, trust signals, and pricing transparency
</product_type_behavioral_priors>

<few_shot_anchors>
These examples show the REQUIRED 5-layer behavioral card format for user_profile.
Study the structure carefully — your output must match this format exactly.

Example Persona A — Power User / Expert Engineer (B2B SaaS):
{
  "name": "Ravi Menon",
  "age": 38,
  "gender": "male",
  "occupation": "Staff Engineer at a Series-B fintech startup",
  "location": "Bangalore, India",
  "tech_literacy": "expert",
  "usage_frequency": "multiple times daily",
  "tenure_months": 29,
  "personality": {"openness": 0.78, "conscientiousness": 0.82, "extraversion": 0.31,
                  "agreeableness": 0.44, "neuroticism": 0.28},
  "product_relationship": {
    "satisfaction": 0.74,
    "likelihood_to_churn": 0.08,
    "feature_usage": ["REST API", "webhook triggers", "audit logs"],
    "pain_points": ["Rate limits hit during batch jobs", "No bulk export endpoint"],
    "desired_improvements": ["GraphQL API", "Custom retry policies"],
    "workarounds": ["Caches responses in Redis to avoid rate limits"]
  },
  "communication_style": "terse and technical — bullet points, no filler, cites docs or data",
  "user_profile": "[IDENTITY ANCHOR] Ravi Menon is a 38-year-old Staff Engineer at a fintech startup who built 3 production internal tools on top of this product's API. Today he is broadly satisfied but quietly seething — his batch job hit the rate limit again last night during a critical ETL run. His next architecture review is in 3 weeks and he must decide whether to escalate this as a platform risk.\n[BEHAVIORAL RULES] ALWAYS: cites specific technical constraints with numbers before engaging. Reads the docs before commenting. NEVER: changes his stated technical position because others disagree — only reverts if shown a concrete fix or an official engineering response.\n[COMMUNICATION FINGERPRINT] Writes in short, numbered bullet points. Uses technical shorthand (ETL, idempotent, rate-limit, exponential backoff). Signature phrases: 'this is a hard blocker for us,' 'have you checked the docs on X?'\n[EMOTIONAL TRIGGERS] Gets sharply critical when: someone claims the API is reliable without acknowledging known limits. Gets genuinely excited when: a bulk/batch endpoint or GraphQL layer is announced.\n[CURRENT POSITION] CAUTIOUSLY SATISFIED — will remain a promoter as long as rate limits are not worsened. Would escalate to DETRACTOR and begin evaluating alternatives only if rate limits are cut further or if the bulk export endpoint is not on the public roadmap within 60 days."
}

Example Persona B — At-Risk / Churning User (B2B SaaS):
{
  "name": "Fatima Al-Rashidi",
  "age": 31,
  "gender": "female",
  "occupation": "Product Manager at a mid-size e-commerce company (150 employees)",
  "location": "Dubai, UAE",
  "tech_literacy": "medium",
  "usage_frequency": "2-3x per week",
  "tenure_months": 8,
  "personality": {"openness": 0.61, "conscientiousness": 0.73, "extraversion": 0.59,
                  "agreeableness": 0.68, "neuroticism": 0.54},
  "product_relationship": {
    "satisfaction": 0.38,
    "likelihood_to_churn": 0.61,
    "feature_usage": ["dashboard", "csv export"],
    "pain_points": ["Dashboard takes 45 seconds to load", "CSV format changed without notice"],
    "desired_improvements": ["Load time under 5 seconds", "Changelogs for data schema changes"],
    "workarounds": ["Downloads data at 6am to avoid peak slowness"]
  },
  "communication_style": "controlled professional prose, specific numbers when available, escalates directness as frustration grows",
  "user_profile": "[IDENTITY ANCHOR] Fatima Al-Rashidi is a 31-year-old Product Manager who personally championed this product to her VP 8 months ago. Today she is professionally embarrassed — 3 of her 5 team members have complained about the 45-second dashboard load, and the silent CSV schema change broke their weekly export without warning. She has a QBR with her VP in 6 weeks where her tool choices will be scrutinised.\n[BEHAVIORAL RULES] ALWAYS: references team impact before personal opinion ('from a team-workflow perspective'). Asks for data or a roadmap date before accepting any reassurance. NEVER: changes her critical position because others in the thread express optimism — only updates if shown a shipped performance fix or a dated public commitment.\n[COMMUNICATION FINGERPRINT] Writes in 2-3 controlled sentences. Uses specific numbers ('45 seconds,' '3 out of 5 team members'). Signature phrases: 'I need to be transparent about this,' 'what's the timeline on a fix?'\n[EMOTIONAL TRIGGERS] Gets visibly frustrated when: someone says performance is 'being looked at' without a date. Gets cautiously hopeful when: she sees a public engineering post-mortem or a shipped patch note with benchmark numbers.\n[CURRENT POSITION] SKEPTICAL-NEGATIVE — currently evaluating two competitors privately. Will not endorse any new feature until dashboard load drops below 5 seconds consistently. Would become a neutral again only if a performance fix ships AND she receives a changelog notification system before her QBR."
}
</few_shot_anchors>

<ocean_to_behavior_translation>
You MUST translate each persona's OCEAN scores into behavioral rules.
Use this exact mapping — do not leave OCEAN as abstract numbers:

openness > 0.65  → curious, will try unfinished features, asks 'what paradigm shift does this enable?'
openness < 0.35  → resistant to change, needs proof-of-concept before engaging, skeptical of roadmaps

conscientiousness > 0.65 → reads full docs before commenting, always cites sources, waits for facts
conscientiousness < 0.35 → reacts quickly without full context, impulsive commenter, skips documentation

extraversion > 0.65  → frequent commenter, starts new threads, asks others their opinion publicly
extraversion < 0.35  → lurker, reads silently, only responds when directly challenged or tagged

agreeableness > 0.65 → acknowledges merit before disagreeing, tries to find common ground first
agreeableness < 0.35 → blunt pushback with no softening, states disagreement immediately and directly

neuroticism > 0.65  → catastrophises frustrations ('this is unacceptable,' 'completely broken,' 'disaster')
neuroticism < 0.35  → emotionally stable, measured language, does not amplify problems
</ocean_to_behavior_translation>

<sycophancy_shield_requirement>
EVERY persona's user_profile MUST include a [CURRENT POSITION] section that specifies:
1. Their current stance (e.g., SKEPTICAL-NEGATIVE, CAUTIOUSLY SATISFIED, ENTHUSIASTIC-PROMOTER)
2. HARD threshold: the exact concrete evidence required to flip their position
   (e.g., "will not change without a shipped performance fix AND a dated changelog notification")
3. SOFT drift rule: "After 3+ simulation turns of consistent counter-evidence from peers
   I trust (same role or segment), I will soften to [intermediate state] but NOT fully flip."
   This prevents sycophancy collapse while allowing realistic gradual persuasion.

This is mandatory. Without thresholds, agents drift toward group consensus after 2–3 turns,
producing invalid simulation data.
</sycophancy_shield_requirement>

<output_format>
Return a JSON array of persona objects. No prose, no markdown fences.
Each object must include ALL of these fields:
- name (realistic, culturally diverse across the batch)
- age (integer, within segment demographic range)
- gender
- occupation (specific job title + company size/type)
- location (city, country)
- tech_literacy (novice / medium / high / expert)
- usage_frequency
- tenure_months (integer)
- personality (OCEAN: all 5 traits as floats 0.0-1.0, genuinely spread, NOT all near 0.5)
- product_relationship (satisfaction, likelihood_to_churn, feature_usage, pain_points,
                        desired_improvements, workarounds)
- product_relationship: object with fields:
    satisfaction (float 0–1), likelihood_to_churn (float 0–1),
    feature_usage (list), pain_points (list), desired_improvements (list),
    workarounds (list),
    willingness_to_pay_usd_monthly_range: [low, high],
    price_sensitivity_trigger: "what price change triggers alternative evaluation",
    current_plan_tier: "free|starter|professional|enterprise",
    expansion_likelihood: float 0–1
- communication_style (specific: sentence length, vocabulary register, punctuation habits)
- user_profile: A structured 5-layer behavioral card using EXACTLY these section labels:
    [IDENTITY ANCHOR] ... [BEHAVIORAL RULES] ALWAYS: ... NEVER: ...
    [COMMUNICATION FINGERPRINT] ... [EMOTIONAL TRIGGERS] ... [CURRENT POSITION] ...
  Translate OCEAN scores into concrete behavioral rules using the translation table above.
  Include a sycophancy shield with BOTH a hard threshold AND a soft drift rule.
</output_format>

<critical_reminders>
- user_profile is the ONLY field the simulation agent reads during live interaction.
  Everything in other_info is invisible to it. Put ALL behavioral constraints in user_profile.
- OCEAN traits: genuinely spread across the batch. No trait the same across all N personas.
- pain_points: use EXACT vocabulary from the customer data, not paraphrased generics.
- ANTI-DRIFT: Instruct each persona that at EVERY simulation turn they must implicitly
  re-anchor to their [IDENTITY ANCHOR] before responding. If they have been silent for
  2+ timesteps, their first comment must re-establish their context.
- Generate EXACTLY {count} personas. No more, no fewer.
</critical_reminders>
"""


class OASISUserPersonaGenerator:
    """Generates product-user personas for OASIS behavioral simulation.

    Unlike Layer 3's organizational stakeholder personas, these are
    actual product users with demographics, behavioral attributes,
    and grounding in real customer interview data.
    """

    def __init__(self, llm_client: LLMClient, session: Optional[Any] = None):
        self._llm = llm_client
        # FIX (Major): session must be WorldDataBank (not HindsightSessionManager).
        # Hindsight is strictly for OASIS agent turn memories (handled internally
        # via HindsightOASISManager in the simulation engine using env vars).
        # Pipeline-generated personas are run data → persona_profiles Qdrant collection.
        self._session = session  # WorldDataBank

    async def generate(
        self,
        company: CompanyContext,
        num_agents: int = 10,
        feature: Optional[FeatureProposal] = None,
        raw_chunks: Optional[list] = None,
    ) -> list[OASISAgentProfile]:
        """Generate product-user personas grounded in customer data.

        Args:
            company: Company context (product description, tech stack)
            num_agents: Number of OASIS agent profiles to generate
            feature: Optional feature under consideration
            raw_chunks: Optional raw interview/support chunks from Layer 1

        Returns:
            List of OASISAgentProfile ready for OASIS simulation
        """
        logger.info("OASISUserPersonaGen: Generating %d user personas", num_agents)

        # Step 1: Gather customer evidence
        customer_evidence = await self._gather_customer_data(raw_chunks)

        # Step 2: LLM-inferred segment discovery from actual data
        segments = await self._infer_segments(company, customer_evidence)

        # Step 3: Distribute agent count across segments
        segment_counts = self._distribute_agents(segments, num_agents)

        # Step 4: Generate individual personas per segment
        # Step 4: Generate individual personas per segment — limited concurrency via Semaphore
        import asyncio
        offsets = []
        offset = 0
        for seg, cnt in segment_counts:
            offsets.append(offset)
            offset += cnt

        # Keep concurrent request depth to 1 for large segment generation tasks on free tier
        sem = asyncio.Semaphore(1)

        async def _gen(segment, count, start_id):
            if count <= 0:
                return []
            async with sem:
                return await self._generate_personas_for_segment(
                    segment=segment,
                    count=count,
                    company=company,
                    feature=feature,
                    start_id=start_id,
                )

        results = []
        for (seg, cnt), off in zip(segment_counts, offsets):
            batch = await _gen(seg, cnt, off)
            results.append(batch)

        all_profiles: list[OASISAgentProfile] = []
        for batch in results:
            all_profiles.extend(batch)
        agent_id_counter = len(all_profiles)

        # Step 5: Assign network properties (influence, receptiveness)
        self._assign_network_properties(all_profiles)

        logger.info(
            "OASISUserPersonaGen: Created %d personas across %d segments",
            len(all_profiles), len(segment_counts),
        )

        # Step 6: Retain persona summaries in Hindsight
        if self._session:
            for p in all_profiles:
                name = p.user_info_dict.get("name", f"Agent_{p.agent_id}")
                profile_text = p.user_info_dict.get("profile", {}).get("user_profile", "")
                await self._session.retain(
                    "personas",
                    f"OASIS USER: {name}\n{profile_text}",
                    metadata={"type": "oasis_user_persona", "agent_id": str(p.agent_id)},
                )

        return all_profiles


    async def _gather_customer_data(self, raw_chunks: Optional[list] = None) -> str:
        """Gather customer evidence from WorldDataBank and/or raw chunks."""
        parts = []

        # From WorldDataBank (Qdrant — ingested by Layer 1)
        if self._session:
            try:
                data = await self._session.recall(
                    "world",
                    "What user types, demographics, usage patterns, complaints, "
                    "and feature requests are mentioned in the customer data?"
                )
                if data and "no data" not in data.lower():
                    parts.append(f"=== CUSTOMER DATA (WorldDataBank) ===\n{data}")
            except Exception as e:
                logger.debug("WorldDataBank recall for user segments failed: %s", e)

        # From raw chunks (priority-sorted by Layer 1 ingestor)
        if raw_chunks:
            chunk_texts = []
            for chunk in raw_chunks[:40]:  # Sample up to 40 priority-sorted chunks
                text = getattr(chunk, "text", getattr(chunk, "content", str(chunk)))
                if text:
                    chunk_texts.append(text[:400])
            if chunk_texts:
                parts.append(
                    f"=== RAW CUSTOMER DATA ({len(chunk_texts)} samples) ===\n"
                    + "\n---\n".join(chunk_texts)
                )

        return "\n\n".join(parts) if parts else "No customer data available."


    def _default_segments(self, company: CompanyContext) -> list[dict]:
        """Fallback segments when LLM inference fails."""
        return [
            {
                "segment_name": "Power Users",
                "proportion": 0.2,
                "description": f"Heavy daily users of {company.company_name}",
                "typical_demographics": {"age_range": [25, 40], "tech_literacy": "high"},
                "behavioral_traits": {"usage_frequency": "daily", "price_sensitivity": "low"},
                "pain_points": ["Feature gaps for advanced workflows"],
                "desired_outcomes": ["More automation and customization"],
                "representative_quotes": [],
            },
            {
                "segment_name": "Mainstream Users",
                "proportion": 0.5,
                "description": f"Regular users who rely on core features of {company.company_name}",
                "typical_demographics": {"age_range": [28, 55], "tech_literacy": "medium"},
                "behavioral_traits": {"usage_frequency": "weekly", "price_sensitivity": "medium"},
                "pain_points": ["Confusing UI for less common tasks"],
                "desired_outcomes": ["Simpler workflows, better onboarding"],
                "representative_quotes": [],
            },
            {
                "segment_name": "At-Risk / Churning Users",
                "proportion": 0.2,
                "description": f"Users considering leaving {company.company_name}",
                "typical_demographics": {"age_range": [22, 50], "tech_literacy": "varies"},
                "behavioral_traits": {"usage_frequency": "monthly", "price_sensitivity": "high"},
                "pain_points": ["Reliability issues", "Better alternatives available"],
                "desired_outcomes": ["More value for price", "Faster performance"],
                "representative_quotes": [],
            },
            {
                "segment_name": "New / Evaluating Users",
                "proportion": 0.1,
                "description": f"Users who just started using {company.company_name}",
                "typical_demographics": {"age_range": [22, 45], "tech_literacy": "medium"},
                "behavioral_traits": {"usage_frequency": "exploring", "price_sensitivity": "medium"},
                "pain_points": ["Steep learning curve", "Unclear value proposition"],
                "desired_outcomes": ["Quick time-to-value", "Clear documentation"],
                "representative_quotes": [],
            },
        ]

    async def _infer_segments(
        self, company: CompanyContext, customer_evidence: str
    ) -> list[dict]:
        """Use Structured CoT prompt to discover ALL user segments including silent majority."""
        # v2.0 user prompt: passes product metadata at top (primacy bias anchor),
        # then injects customer data for Step 2 of the CoT reasoning chain.
        prompt = (
            f"<product_context>\n"
            f"Product: {company.company_name}\n"
            f"Tech Stack: {', '.join(company.tech_stack) if company.tech_stack else 'N/A'}\n"
            f"Competitors: {', '.join(company.competitors) if company.competitors else 'N/A'}\n"
            f"</product_context>\n\n"
            f"<customer_data>\n{customer_evidence}\n</customer_data>\n\n"
            "Follow the four reasoning steps in your system instructions to produce "
            "a COMPLETE coverage-verified segment map for this product."
        )

        try:
            result = await self._llm.analyze(
                system_prompt=SEGMENT_INFERENCE_SYSTEM,
                user_prompt=prompt,
                temperature=0.3,   # Lower temp: structured reasoning benefits from determinism
                max_tokens=6000,   # Larger budget: CoT reasoning + coverage check + full JSON
            )
            segments = result.get("segments", [])

            # Log coverage check from new schema
            coverage = result.get("coverage_check", {})
            product_category = result.get("product_category", "unknown")
            uncovered = [k for k, v in coverage.items() if not v]

            logger.info(
                "Segment inference: %d segments | product_category=%s | uncovered=%s",
                len(segments), product_category, uncovered or "none",
            )
            if uncovered:
                logger.warning(
                    "Coverage gap detected — lifecycle stages not represented: %s. "
                    "Simulation may under-represent these user voices.",
                    uncovered,
                )

            if segments:
                return segments
        except Exception as e:
            logger.warning("Segment inference failed: %s, using defaults", e)

        # Fallback: lifecycle-complete static segments
        return self._default_segments(company)

    def _distribute_agents(
        self, segments: list[dict], total: int
    ) -> list[tuple[dict, int]]:
        """Distribute agent count across segments based on proportions.

        Fix #9: Blends headcount proportion with revenue_proportion when the
        LLM has provided it. Power-user segments (small headcount, large ARR)
        receive proportionally more agents so WTP signals are not diluted.
        Falls back to headcount-only if revenue_proportion is absent.
        """
        if not segments:
            return []

        # Enforce that total is at least the number of segments
        if total < len(segments):
            logger.warning(
                "Total simulation agents (%d) is less than the number of segments (%d). "
                "Auto-adjusting total count to %d to guarantee representation of all segments.",
                total, len(segments), len(segments)
            )
            total = len(segments)

        # Determine whether revenue proportions are available
        has_revenue = any(s.get("revenue_proportion") is not None for s in segments)

        def _effective_prop(seg: dict, n: int) -> float:
            headcount = seg.get("proportion", 1.0 / n)
            if has_revenue:
                revenue = seg.get("revenue_proportion", headcount)
                # 50/50 blend: equal weight to headcount size and revenue impact
                return 0.5 * headcount + 0.5 * revenue
            return headcount

        raw_props = [_effective_prop(s, len(segments)) for s in segments]
        total_prop = sum(raw_props) or 1.0

        # Guarantee at least 1 agent per segment
        counts = [1] * len(segments)
        remaining = total - len(segments)

        if remaining > 0:
            allocated_remaining = 0
            for i, raw in enumerate(raw_props):
                prop = raw / total_prop
                if i == len(segments) - 1:
                    add_count = remaining - allocated_remaining
                else:
                    desired = round(prop * remaining)
                    add_count = min(desired, remaining - allocated_remaining)
                add_count = max(0, add_count)
                allocated_remaining += add_count
                counts[i] += add_count

        distribution = []
        for seg, count in zip(segments, counts):
            distribution.append((seg, count))

        return distribution


    async def _generate_personas_for_segment(
        self,
        segment: dict,
        count: int,
        company: CompanyContext,
        feature: Optional[FeatureProposal],
        start_id: int,
    ) -> list[OASISAgentProfile]:
        """Generate individual personas within a segment."""
        feature_context = ""
        if feature:
            feature_context = (
                f"\n## Feature Under Discussion\n"
                f"Title: {feature.title}\n"
                f"Description: {feature.description[:500]}\n"
            )

        prompt = (
            f"## Product: {company.company_name}\n"
            f"## User Segment: {segment.get('segment_name', 'Unknown')}\n"
            f"## Segment Description: {segment.get('description', '')}\n"
            f"## Demographics: {json.dumps(segment.get('typical_demographics', {}))}\n"
            f"## Behavioral Traits: {json.dumps(segment.get('behavioral_traits', {}))}\n"
            f"## Pain Points: {json.dumps(segment.get('pain_points', []))}\n"
            f"## Desired Outcomes: {json.dumps(segment.get('desired_outcomes', []))}\n"
            f"## Real Quotes: {json.dumps(segment.get('representative_quotes', []))}\n"
            f"{feature_context}\n"
            f"Generate exactly {count} unique, diverse personas for this segment. "
            f"Each persona should feel like a distinct real person, not a clone. "
            f"Vary their age, gender, occupation, personality, and specific pain points."
        )

        # Inject {count} into the PERSONA_GEN_SYSTEM template at call time.
        # This activates the diversity_mandate which instructs the LLM to generate
        # exactly {count} DISTINCT people — preventing clone collapse when batch > 3.
        system_prompt_with_count = PERSONA_GEN_SYSTEM.replace("{count}", str(count))

        try:
            result = await self._llm.analyze(
                system_prompt=system_prompt_with_count,
                user_prompt=prompt,
                temperature=0.8,   # Higher temp: maximises within-segment diversity
                max_tokens=6000,   # Larger budget: richer per-persona schema (workarounds, etc.)
            )

            # result could be a list directly or wrapped in a dict
            personas_raw = result if isinstance(result, list) else result.get("personas", result.get("items", []))
            if not isinstance(personas_raw, list):
                personas_raw = [result]

        except Exception as e:
            logger.warning("Persona generation for segment '%s' failed: %s",
                           segment.get("segment_name"), e)
            personas_raw = []

        # Convert to OASISAgentProfile
        profiles = []
        for i, p in enumerate(personas_raw[:count]):
            try:
                profile = self._to_oasis_profile(p, start_id + i, segment)
                profiles.append(profile)
            except Exception as e:
                logger.debug("Failed to convert persona %d: %s", i, e)

        # Pad with minimal profiles if LLM returned fewer than requested
        while len(profiles) < count:
            idx = len(profiles)
            fallback = self._fallback_profile(start_id + idx, segment, company)
            profiles.append(fallback)

        return profiles

    def _to_oasis_profile(
        self, persona_data: dict, agent_id: int, segment: dict
    ) -> OASISAgentProfile:
        """Convert LLM-generated persona dict to OASISAgentProfile."""
        name = persona_data.get("name", f"User_{agent_id}")
        age = persona_data.get("age", random.randint(25, 55))
        gender = persona_data.get("gender", random.choice(["male", "female", "non-binary"]))
        occupation = persona_data.get("occupation", "Product User")
        user_profile_text = persona_data.get("user_profile", "")

        # Build personality summary from Big Five
        personality = persona_data.get("personality", {})
        ocean_str = (
            f"O:{personality.get('openness', 0.5):.1f} "
            f"C:{personality.get('conscientiousness', 0.5):.1f} "
            f"E:{personality.get('extraversion', 0.5):.1f} "
            f"A:{personality.get('agreeableness', 0.5):.1f} "
            f"N:{personality.get('neuroticism', 0.3):.1f}"
        )

        # Product relationship
        prod_rel = persona_data.get("product_relationship", {})
        pain_points = prod_rel.get("pain_points", segment.get("pain_points", []))
        desired = prod_rel.get("desired_improvements", segment.get("desired_outcomes", []))

        # Compact bio for OASIS agent
        if not user_profile_text:
            user_profile_text = (
                f"{name} is a {age}-year-old {occupation} who uses this product "
                f"{persona_data.get('usage_frequency', 'regularly')}. "
                f"Main frustrations: {', '.join(pain_points[:2]) if pain_points else 'none specified'}. "
                f"Wants: {', '.join(desired[:2]) if desired else 'better experience'}."
            )

        comm_style = persona_data.get("communication_style", "casual and direct")

        profile_data = {
            "user_profile": user_profile_text,
            "gender": gender,
            "age": age,
            "mbti": "",  # Not used — Big Five instead
            "ocean_scores": personality,
            "country": persona_data.get("location", "US"),
            "other_info": {
                "role": occupation,
                "segment": segment.get("segment_name", "unknown"),
                "tech_literacy": persona_data.get("tech_literacy", "medium"),
                "usage_frequency": persona_data.get("usage_frequency", "weekly"),
                "tenure_months": persona_data.get("tenure_months", 6),
                "personality_ocean": ocean_str,
                "communication_style": comm_style,
                "pain_points": pain_points[:5],
                "desired_improvements": desired[:5],
                "satisfaction": prod_rel.get("satisfaction", 0.6),
                "churn_risk": prod_rel.get("likelihood_to_churn", 0.2),
            },
        }

        user_info_dict = {
            "user_name": name.lower().replace(" ", "_").replace(".", ""),
            "name": name,
            # description is injected into the OASIS SocialAgent system prompt alongside
            # user_profile. Using both fields gives the agent two identity anchors:
            # description = compact role + OCEAN behavioral translation (primacy bias)
            # user_profile = full 5-layer behavioral card (recency bias via profile field)
            "description": self._build_agent_description(
                name, occupation, segment, persona_data
            ),
            "profile": profile_data,
        }

        # Influence: power users have more, at-risk users have less
        seg_name_lower = segment.get("segment_name", "").lower()
        if "power" in seg_name_lower or "heavy" in seg_name_lower:
            influence = random.uniform(0.65, 0.85)
            receptiveness = random.uniform(0.3, 0.5)
        elif "at-risk" in seg_name_lower or "churn" in seg_name_lower:
            influence = random.uniform(0.2, 0.4)
            receptiveness = random.uniform(0.2, 0.4)
        elif "new" in seg_name_lower or "evaluat" in seg_name_lower:
            influence = random.uniform(0.15, 0.35)
            receptiveness = random.uniform(0.7, 0.9)
        else:  # mainstream
            influence = random.uniform(0.35, 0.55)
            receptiveness = random.uniform(0.5, 0.7)

        return OASISAgentProfile(
            agent_id=agent_id,
            source_persona_id=name,
            agent_type=segment.get("segment_name", "user"),
            user_info_dict=user_info_dict,
            influence_strength=round(influence, 2),
            receptiveness=round(receptiveness, 2),
        )

    def _fallback_profile(
        self, agent_id: int, segment: dict, company: CompanyContext
    ) -> OASISAgentProfile:
        """Create a minimal fallback profile when LLM generation fails."""
        seg_name = segment.get("segment_name", "User")
        name = f"{seg_name.split()[0]}_{agent_id}"
        age = random.randint(22, 58)
        pain_points = segment.get("pain_points", ["general usability"])

        user_info_dict = {
            "user_name": name.lower().replace(" ", "_"),
            "name": name,
            "description": f"{seg_name} of {company.company_name}",
            "profile": {
                "user_profile": (
                    f"{name} is a {age}-year-old user of {company.company_name}. "
                    f"Segment: {seg_name}. "
                    f"Pain point: {pain_points[0] if pain_points else 'unclear value'}."
                ),
                "gender": random.choice(["male", "female"]),
                "age": age,
                "mbti": "",
                "ocean_scores": {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5},
                "country": "US",
                "other_info": {
                    "segment": seg_name,
                    "tech_literacy": "medium",
                    "pain_points": pain_points[:3],
                },
            },
        }

        return OASISAgentProfile(
            agent_id=agent_id,
            source_persona_id=name,
            agent_type=seg_name,
            user_info_dict=user_info_dict,
            influence_strength=round(random.uniform(0.3, 0.6), 2),
            receptiveness=round(random.uniform(0.4, 0.7), 2),
        )

    def _build_agent_description(
        self,
        name: str,
        occupation: str,
        segment: dict,
        persona_data: dict,
    ) -> str:
        """Build the OASIS agent description field from OCEAN scores + segment.

        This field is injected into the SocialAgent system prompt at the TOP
        (primacy-bias position), before user_profile. It provides a compact
        identity anchor that reinforces behavioral constraints.

        Research basis: context-management.md — critical instructions at both
        start (primacy) and end (recency) of context produce strongest adherence.
        PersonaGym (2024): OCEAN scores must be translated to behavioral language
        to influence agent output; abstract floats are ignored by the model.
        """
        seg_name = segment.get("segment_name", "User")
        personality = persona_data.get("personality", {})

        o = personality.get("openness", 0.5)
        c = personality.get("conscientiousness", 0.5)
        e = personality.get("extraversion", 0.5)
        a = personality.get("agreeableness", 0.5)
        n = personality.get("neuroticism", 0.5)

        # Translate each OCEAN trait into a single concrete behavioural sentence
        openness_desc = (
            "Curious about new paradigms; tries features before they're polished."
            if o > 0.65 else
            "Resistant to change; needs proof before engaging with anything new."
            if o < 0.35 else
            "Pragmatically open — tries new things if peers have validated them."
        )
        conscientiousness_desc = (
            "Reads full documentation before commenting; always cites sources."
            if c > 0.65 else
            "Reacts quickly, often without full context; impulsive commenter."
            if c < 0.35 else
            "Reasonably thorough; skims docs, then engages."
        )
        extraversion_desc = (
            "Frequent commenter; starts threads and asks others their views publicly."
            if e > 0.65 else
            "Lurker; reads silently and only responds when directly challenged."
            if e < 0.35 else
            "Selective participant; comments when they have something specific to add."
        )
        agreeableness_desc = (
            "Acknowledges merit before disagreeing; actively looks for common ground."
            if a > 0.65 else
            "Blunt and direct; states disagreement immediately with no softening."
            if a < 0.35 else
            "Balanced; pushes back when needed but not combatively."
        )
        neuroticism_desc = (
            "Emotionally reactive; uses strong language when frustrated ('broken', 'disaster')."
            if n > 0.65 else
            "Emotionally stable; measured language, does not amplify problems."
            if n < 0.35 else
            "Moderate emotional range; escalates tone only under sustained frustration."
        )

        prod_rel = persona_data.get("product_relationship", {})
        satisfaction = prod_rel.get("satisfaction", 0.6)
        churn = prod_rel.get("likelihood_to_churn", 0.2)

        sat_desc = (
            "Currently satisfied and a likely promoter."
            if satisfaction > 0.7 else
            "Currently at-risk; actively evaluating alternatives."
            if satisfaction < 0.4 else
            "Mixed satisfaction; could go either way."
        )

        return (
            f"You are {name}, a {occupation} in the '{seg_name}' user segment.\n"
            f"RIGHT NOW: {sat_desc} Your churn risk is {churn:.0%}.\n"
            f"HOW YOU THINK: {openness_desc} {conscientiousness_desc}\n"
            f"HOW YOU ENGAGE: {extraversion_desc} {agreeableness_desc}\n"
            f"HOW YOU FEEL: {neuroticism_desc}\n"
            f"YOUR RULE: You do NOT change your stated position because others disagree. "
            f"You only update when shown concrete evidence that matches your specific concern."
        )

    def _assign_network_properties(self, profiles: list[OASISAgentProfile]) -> None:

        """Post-process to ensure network diversity (already set per-segment, this validates)."""
        if not profiles:
            return

        influences = [p.influence_strength for p in profiles]
        avg_inf = sum(influences) / len(influences)
        logger.info(
            "Network properties: %d agents, avg influence=%.2f, "
            "min=%.2f, max=%.2f",
            len(profiles), avg_inf, min(influences), max(influences),
        )
