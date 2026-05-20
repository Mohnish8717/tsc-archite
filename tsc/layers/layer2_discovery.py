"""
Layer 2: Feature Discovery Engine
==================================

Analyzes behavioral simulation output + raw customer data from Hindsight
to propose features the product should build next.

This is the core differentiator — instead of requiring a pre-written
feature_proposal.json, the system discovers what to build from:
  1. Raw customer interviews and support tickets (Hindsight "world" bank)
  2. OASIS behavioral simulation insights (Hindsight "simulation" bank)
  3. Company context and priorities

Output: List of FeatureProposal candidates ranked by evidence strength.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from tsc.llm.base import LLMClient
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.oasis.models import MarketSentimentSeries

logger = logging.getLogger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────────

FEATURE_DISCOVERY_SYSTEM = """You are a world-class Product Manager with deep expertise in
customer research synthesis. Your job is to analyze raw customer data and behavioral
simulation results to identify the most impactful features to build next.

You think like a PM at a top-tier product company: you look for patterns across
multiple data sources, quantify pain points by frequency and severity, identify
unmet needs that competitors haven't addressed, and propose features that are
both high-impact and technically feasible.

RULES:
- Every proposed feature MUST cite specific customer quotes or behavioral data as evidence.
- Rank features by: (1) frequency of pain point mentions, (2) severity of impact,
  (3) alignment with company priorities, (4) competitive gap.
- Be specific about WHAT to build, WHO it's for, and WHY it matters.
- If an existing feature proposal is provided, validate it against the evidence
  and either strengthen it with data or flag gaps.

OUTPUT FORMAT: Return valid JSON with this structure:
{
  "analysis_summary": "Brief synthesis of what the data reveals",
  "top_pain_points": [
    {"pain_point": "...", "frequency": "high/medium/low", "severity": "critical/major/minor",
     "customer_quotes": ["quote1", "quote2"], "affected_segments": ["segment1"]}
  ],
  "proposed_features": [
    {
      "title": "Feature Name",
      "description": "What this feature does and how it solves the pain point",
      "target_users": "Who benefits from this feature",
      "justification": "Why this is worth building, citing specific evidence",
      "customer_evidence": ["Direct quotes or behavioral data supporting this"],
      "affected_domains": ["domain1", "domain2"],
      "priority": "P0/P1/P2",
      "effort_estimate": "small/medium/large",
      "competitive_advantage": "How this differentiates from competitors"
    }
  ]
}
"""


class FeatureDiscoveryEngine:
    """Discovers features to build from customer data and behavioral simulation."""

    def __init__(self, llm_client: LLMClient, session: Optional[Any] = None):
        self._llm = llm_client
        self._session = session  # HindsightSessionManager

    async def process(
        self,
        company: CompanyContext,
        behavioral_results: Optional[MarketSentimentSeries] = None,
        existing_proposal: Optional[FeatureProposal] = None,
        raw_chunks: Optional[list] = None,
    ) -> list[FeatureProposal]:
        """Analyze customer data and simulation results to propose features.

        Args:
            company: Company context (product description, priorities, competitors)
            behavioral_results: OASIS simulation output (agent interactions)
            existing_proposal: Optional pre-existing feature proposal to validate
            raw_chunks: Optional enriched chunks from Layer 1

        Returns:
            Ranked list of FeatureProposal candidates with customer evidence
        """
        t0 = time.time()
        logger.info("Layer 2: Starting Feature Discovery")

        # 1. Gather evidence from all sources
        customer_data = await self._gather_customer_evidence(raw_chunks)
        simulation_data = await self._gather_simulation_evidence(behavioral_results)
        
        # 2. Build the discovery prompt
        prompt = self._build_discovery_prompt(
            company=company,
            customer_data=customer_data,
            simulation_data=simulation_data,
            existing_proposal=existing_proposal,
        )

        # 3. LLM synthesis
        logger.info("  Synthesizing feature proposals from %d chars of evidence...",
                     len(customer_data) + len(simulation_data))
        try:
            result = await self._llm.analyze(
                system_prompt=FEATURE_DISCOVERY_SYSTEM,
                user_prompt=prompt,
                temperature=0.4,
                max_tokens=6000,
            )
        except Exception as e:
            logger.error("Feature Discovery LLM call failed: %s", e)
            # If LLM fails and we have an existing proposal, return it as-is
            if existing_proposal:
                return [existing_proposal]
            return [FeatureProposal(
                title="Feature Discovery Failed",
                description=f"LLM synthesis failed: {e}. Please provide a feature_proposal.json.",
            )]

        # 4. Parse proposals
        proposals = self._parse_proposals(result, company)

        # 5. If user provided a proposal, enrich it and put it first
        if existing_proposal:
            enriched = self._enrich_existing_proposal(existing_proposal, result)
            # Remove any duplicate from discovered proposals
            proposals = [p for p in proposals if p.title.lower() != enriched.title.lower()]
            proposals.insert(0, enriched)

        # 6. Retain discoveries into Hindsight
        if self._session:
            for p in proposals:
                await self._session.retain("discovery", 
                    f"PROPOSED FEATURE: {p.title}\n"
                    f"Description: {p.description}\n"
                    f"Target Users: {p.target_users}\n"
                    f"Priority: {p.priority or 'unranked'}",
                    metadata={"type": "feature_proposal", "title": p.title}
                )

        logger.info(
            "Layer 2 complete: %d feature(s) proposed (%.1fs)",
            len(proposals), time.time() - t0,
        )
        return proposals

    async def _gather_customer_evidence(self, raw_chunks: Optional[list] = None) -> str:
        """Gather customer evidence from Hindsight world bank and/or raw chunks."""
        evidence_parts = []

        # From Hindsight (if connected)
        if self._session:
            try:
                pain_points = await self._session.recall("world",
                    "What are the top customer complaints, pain points, and feature requests?")
                if pain_points and "no data" not in pain_points.lower():
                    evidence_parts.append(f"=== CUSTOMER PAIN POINTS (from Hindsight) ===\n{pain_points}")
            except Exception as e:
                logger.debug("Hindsight world recall failed: %s", e)

            try:
                usage_data = await self._session.recall("world",
                    "What usage patterns, drop-off points, and engagement issues are mentioned?")
                if usage_data and "no data" not in usage_data.lower():
                    evidence_parts.append(f"=== USAGE PATTERNS ===\n{usage_data}")
            except Exception as e:
                logger.debug("Hindsight usage recall failed: %s", e)

        # From raw chunks (direct access)
        if raw_chunks:
            # FIX (Major): Sort chunks by urgency + sentiment before capping.
            # Rationale: The LLM bases the product vision on these samples.
            # Picking the FIRST 30 (file order) anchors proposals to irrelevant
            # boilerplate. Instead, surface the highest-signal chunks — those
            # marked CRITICAL/HIGH urgency or with negative sentiment (the pain points
            # that drive real feature decisions).
            def _chunk_priority(c) -> int:
                urgency = str(getattr(c, "urgency_level", "")).upper()
                sentiment = str(getattr(c, "sentiment_label", "")).upper()
                score = 0
                if urgency == "CRITICAL":
                    score += 3
                elif urgency == "HIGH":
                    score += 2
                elif urgency == "MEDIUM":
                    score += 1
                # Negative sentiment = pain point = high product signal
                if sentiment in ("NEGATIVE", "ANGER", "FRUSTRATION", "FEAR"):
                    score += 2
                elif sentiment == "NEUTRAL":
                    score += 0
                return score

            # Sort descending — highest priority pain points first
            sorted_chunks = sorted(raw_chunks, key=_chunk_priority, reverse=True)
            top_chunks = sorted_chunks[:30]  # cap after priority sort

            # Minimum evidence threshold (user-confirmed requirement):
            # if we have fewer than 3 chunks, warn but continue —
            # the LLM fallback in _build_discovery_prompt still runs.
            if len(top_chunks) < 3:
                logger.warning(
                    "Feature Discovery: only %d customer evidence chunks available "
                    "(minimum recommended: 3). Proposals may lack evidence grounding.",
                    len(top_chunks),
                )

            chunk_texts = []
            for chunk in top_chunks:
                text = getattr(chunk, 'text', getattr(chunk, 'content', str(chunk)))
                if text:
                    chunk_texts.append(text[:500])
            if chunk_texts:
                evidence_parts.append(
                    f"=== RAW CUSTOMER DATA ({len(chunk_texts)} chunks, priority-sorted) ===\n"
                    + "\n---\n".join(chunk_texts)
                )

        return "\n\n".join(evidence_parts) if evidence_parts else "No customer data available."

    async def _gather_simulation_evidence(
        self, behavioral_results: Optional[MarketSentimentSeries] = None
    ) -> str:
        """Gather behavioral simulation evidence."""
        evidence_parts = []

        # From Hindsight simulation bank
        if self._session:
            try:
                sim_data = await self._session.recall("simulation",
                    "What did simulated users say about their needs, frustrations, and desired features?")
                if sim_data and "no data" not in sim_data.lower():
                    evidence_parts.append(f"=== SIMULATION BEHAVIORAL INSIGHTS ===\n{sim_data}")
            except Exception as e:
                logger.debug("Hindsight simulation recall failed: %s", e)

        # From direct MarketSentimentSeries
        if behavioral_results and behavioral_results.agent_interactions:
            agent_quotes = []
            for agent_id, interactions in behavioral_results.agent_interactions.items():
                for interaction in interactions[:3]:  # Top 3 per agent
                    agent_quotes.append(f"  Agent {agent_id}: {interaction[:300]}")
            if agent_quotes:
                evidence_parts.append(
                    f"=== OASIS AGENT INTERACTIONS ({len(agent_quotes)} samples) ===\n"
                    + "\n".join(agent_quotes[:50])
                )

        return "\n\n".join(evidence_parts) if evidence_parts else "No simulation data available."

    def _build_discovery_prompt(
        self,
        company: CompanyContext,
        customer_data: str,
        simulation_data: str,
        existing_proposal: Optional[FeatureProposal] = None,
    ) -> str:
        """Build the feature discovery prompt with all evidence."""
        sections = [
            f"## Company Context",
            f"Company: {company.company_name}",
            f"Tech Stack: {', '.join(company.tech_stack) if company.tech_stack else 'Not specified'}",
            f"Current Priorities: {', '.join(company.current_priorities) if company.current_priorities else 'Not specified'}",
            f"Competitors: {', '.join(company.competitors) if company.competitors else 'Not specified'}",
            f"Team Size: {company.team_size or 'Unknown'}",
            f"Constraints: {', '.join(company.constraints) if company.constraints else 'None specified'}",
            "",
            f"## Customer Evidence",
            customer_data,
            "",
            f"## Behavioral Simulation Results",
            simulation_data,
        ]

        if existing_proposal:
            sections.extend([
                "",
                f"## Existing Feature Proposal (validate against evidence)",
                f"Title: {existing_proposal.title}",
                f"Description: {existing_proposal.description}",
                f"Target Users: {existing_proposal.target_users}",
                "",
                "TASK: Validate this proposal against the customer evidence above.",
                "If the evidence supports it, strengthen the justification with specific quotes.",
                "If the evidence contradicts it, flag the gaps and propose alternatives.",
                "Also propose 1-2 additional features from the evidence.",
            ])
        else:
            sections.extend([
                "",
                "## Task",
                "Based on ALL the evidence above, identify the top 3 features this product",
                "should build next. Rank them by evidence strength and business impact.",
                "Each feature must cite specific customer quotes or simulation data as justification.",
            ])

        return "\n".join(sections)

    def _parse_proposals(self, result: dict, company: CompanyContext) -> list[FeatureProposal]:
        """Parse LLM output into FeatureProposal objects."""
        proposals = []
        raw_features = result.get("proposed_features", [])

        for feat in raw_features[:5]:  # Cap at 5
            try:
                proposal = FeatureProposal(
                    title=feat.get("title", "Untitled Feature"),
                    description=feat.get("description", ""),
                    target_users=feat.get("target_users", ""),
                    affected_domains=feat.get("affected_domains", []),
                    priority=feat.get("priority"),
                    tech_stack=company.tech_stack,
                    customer_segments=feat.get("affected_segments", []),
                )
                proposals.append(proposal)
            except Exception as e:
                logger.warning("Failed to parse feature proposal: %s", e)

        if not proposals:
            logger.warning("No features parsed from LLM output")

        return proposals

    def _enrich_existing_proposal(
        self, proposal: FeatureProposal, analysis: dict
    ) -> FeatureProposal:
        """Enrich an existing proposal with evidence from the analysis."""
        # Find matching pain points
        pain_points = analysis.get("top_pain_points", [])
        evidence_quotes = []
        for pp in pain_points:
            evidence_quotes.extend(pp.get("customer_quotes", []))

        # Build enriched description
        enriched_desc = proposal.description
        if evidence_quotes:
            enriched_desc += "\n\n--- Customer Evidence ---\n"
            for quote in evidence_quotes[:5]:
                enriched_desc += f"• \"{quote}\"\n"

        summary = analysis.get("analysis_summary", "")
        if summary:
            enriched_desc += f"\n--- Analysis ---\n{summary}"

        return FeatureProposal(
            title=proposal.title,
            description=enriched_desc,
            target_users=proposal.target_users,
            target_user_count=proposal.target_user_count,
            effort_weeks_min=proposal.effort_weeks_min,
            effort_weeks_max=proposal.effort_weeks_max,
            affected_domains=proposal.affected_domains,
            existing_features=proposal.existing_features,
            tech_stack=proposal.tech_stack,
            priority=proposal.priority,
            revenue_model=proposal.revenue_model,
            pricing_strategy=proposal.pricing_strategy,
            customer_segments=proposal.customer_segments,
        )
