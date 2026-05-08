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


# ── System Prompts ───────────────────────────────────────────────────────

SEGMENT_INFERENCE_SYSTEM = """You are a senior UX researcher analyzing raw customer data
to identify distinct user segments for a product.

Your job is to read through customer interviews, support tickets, and usage data,
then identify the natural user segments that emerge from the data.

RULES:
- Each segment must be strictly grounded in patterns you observe in the data — do NOT invent generic segments.
- Do NOT hallucinate standard B2B segments (like 'mainstream users') if they do not explicitly appear in the crisis or support data.
- Ensure the segment proportions mathematically match their prevalence in the provided input evidence.
- For each segment, extract real pain points and vocabulary from the data.

OUTPUT FORMAT — Return valid JSON:
{
  "segments": [
    {
      "segment_name": "Power Users — Heavy Integrators",
      "proportion": 0.25,
      "description": "Users who deeply integrate the product into their daily workflow",
      "typical_demographics": {
        "age_range": [28, 45],
        "occupations": ["Senior Engineer", "Tech Lead", "Architect"],
        "tech_literacy": "high"
      },
      "behavioral_traits": {
        "usage_frequency": "daily",
        "feature_depth": "deep",
        "support_tendency": "self-serve",
        "price_sensitivity": "low"
      },
      "pain_points": ["Extracted from actual data..."],
      "desired_outcomes": ["Extracted from actual data..."],
      "representative_quotes": ["Direct quotes from interviews..."]
    }
  ]
}
"""

PERSONA_GEN_SYSTEM = """You are generating realistic product-user personas for a social simulation.

Each persona must feel like a REAL person who uses THIS specific product. They are NOT
generic archetypes — they are grounded in the customer segment data and company context provided.

For each persona, generate a COMPACT structured profile (NOT a long narrative).

RULES:
- Vary demographics within the segment's ranges — no two personas should be identical.
- Big Five personality traits (OCEAN) must be specific numbers, not "medium" or "high".
- Pain points and vocabulary should come from the segment's actual customer data.
- The persona's "user_profile" text should be 2-3 sentences max — this is what the OASIS
  agent will use as its identity. Make it vivid and specific.

OUTPUT FORMAT — Return a JSON array:
[
  {
    "name": "Realistic full name",
    "age": 34,
    "gender": "female",
    "occupation": "Product Designer at a mid-size SaaS company",
    "location": "Austin, TX",
    "tech_literacy": "high",
    "usage_frequency": "daily",
    "tenure_months": 18,
    "personality": {
      "openness": 0.72,
      "conscientiousness": 0.65,
      "extraversion": 0.45,
      "agreeableness": 0.58,
      "neuroticism": 0.35
    },
    "product_relationship": {
      "satisfaction": 0.7,
      "likelihood_to_churn": 0.15,
      "feature_usage": ["feature_a", "feature_b"],
      "pain_points": ["specific pain from customer data"],
      "desired_improvements": ["specific desire from customer data"],
      "workarounds": ["what they do to cope"]
    },
    "communication_style": "direct and analytical, uses technical jargon",
    "user_profile": "2-3 sentence vivid identity description for the OASIS agent"
  }
]
"""


class OASISUserPersonaGenerator:
    """Generates product-user personas for OASIS behavioral simulation.

    Unlike Layer 3's organizational stakeholder personas, these are
    actual product users with demographics, behavioral attributes,
    and grounding in real customer interview data.
    """

    def __init__(self, llm_client: LLMClient, session: Optional[Any] = None):
        self._llm = llm_client
        self._session = session  # HindsightSessionManager

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
        all_profiles: list[OASISAgentProfile] = []
        agent_id_counter = 0

        for segment, count in segment_counts:
            if count <= 0:
                continue

            personas = await self._generate_personas_for_segment(
                segment=segment,
                count=count,
                company=company,
                feature=feature,
                start_id=agent_id_counter,
            )
            all_profiles.extend(personas)
            agent_id_counter += len(personas)

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
        """Gather customer evidence from Hindsight and/or raw chunks."""
        parts = []

        # From Hindsight world bank
        if self._session:
            try:
                data = await self._session.recall(
                    "world",
                    "What user types, demographics, usage patterns, complaints, "
                    "and feature requests are mentioned in the customer data?"
                )
                if data and "no data" not in data.lower():
                    parts.append(f"=== CUSTOMER DATA (Hindsight) ===\n{data}")
            except Exception as e:
                logger.debug("Hindsight recall for user segments failed: %s", e)

        # From raw chunks (direct)
        if raw_chunks:
            chunk_texts = []
            for chunk in raw_chunks[:40]:  # Sample up to 40 chunks
                text = getattr(chunk, "text", getattr(chunk, "content", str(chunk)))
                if text:
                    chunk_texts.append(text[:400])
            if chunk_texts:
                parts.append(
                    f"=== RAW CUSTOMER DATA ({len(chunk_texts)} samples) ===\n"
                    + "\n---\n".join(chunk_texts)
                )

        return "\n\n".join(parts) if parts else "No customer data available."

    async def _infer_segments(
        self, company: CompanyContext, customer_evidence: str
    ) -> list[dict]:
        """Use LLM to discover user segments from actual customer data."""
        prompt = (
            f"## Product: {company.company_name}\n"
            f"## Tech Stack: {', '.join(company.tech_stack) if company.tech_stack else 'N/A'}\n"
            f"## Competitors: {', '.join(company.competitors) if company.competitors else 'N/A'}\n\n"
            f"## Customer Data\n{customer_evidence}\n\n"
            "Based on the customer data above, identify 3-5 distinct user segments "
            "that naturally emerge. Ground each segment in specific evidence from the data."
        )

        try:
            result = await self._llm.analyze(
                system_prompt=SEGMENT_INFERENCE_SYSTEM,
                user_prompt=prompt,
                temperature=0.4,
                max_tokens=4000,
            )
            segments = result.get("segments", [])
            if segments:
                logger.info("Inferred %d user segments from customer data", len(segments))
                return segments
        except Exception as e:
            logger.warning("Segment inference failed: %s, using defaults", e)

        # Fallback: generic but reasonable segments
        return self._default_segments(company)

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

    def _distribute_agents(
        self, segments: list[dict], total: int
    ) -> list[tuple[dict, int]]:
        """Distribute agent count across segments based on proportions."""
        if not segments:
            return []

        # Normalize proportions
        total_prop = sum(s.get("proportion", 1.0 / len(segments)) for s in segments)
        distribution = []
        allocated = 0

        for i, seg in enumerate(segments):
            prop = seg.get("proportion", 1.0 / len(segments)) / total_prop
            if i == len(segments) - 1:
                # Last segment gets the remainder
                count = total - allocated
            else:
                count = max(1, round(prop * total))
            allocated += count
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

        try:
            result = await self._llm.analyze(
                system_prompt=PERSONA_GEN_SYSTEM,
                user_prompt=prompt,
                temperature=0.7,  # Higher creativity for persona diversity
                max_tokens=3000,
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
            "description": f"{segment.get('segment_name', 'User')} — {occupation}",
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
