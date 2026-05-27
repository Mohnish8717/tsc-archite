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
from tsc.memory.world_rag import _get_qdrant, _embed

logger = logging.getLogger(__name__)


# ── Prompts and Schemas ──────────────────────────────────────────────────

FEATURE_MAP_SYSTEM = """You are a Customer Research Synthesis Map Agent.
Analyze a subset of raw customer data and behavioral simulation outputs to identify top pain points and feature suggestions.

OUTPUT FORMAT: Return valid JSON with this exact structure:
{
  "top_pain_points": [
    {
      "pain_point": "Friction description",
      "frequency": "high/medium/low",
      "severity": "critical/major/minor",
      "customer_quotes": ["specific quote or data point"],
      "affected_segments": ["segments"]
    }
  ],
  "proposed_features": [
    {
      "title": "Proposed Feature Title",
      "description": "What it does",
      "target_users": "Target audience",
      "justification": "Why it is needed, citing specific evidence",
      "customer_evidence": ["quote or data point"],
      "affected_domains": ["domain1"],
      "priority": "P0/P1/P2",
      "effort_estimate": "small/medium/large",
      "competitive_advantage": "Differentiator"
    }
  ]
}
"""

FEATURE_REDUCE_SYSTEM = """You are a Senior Product Manager.
You have been given a set of intermediate pain points and feature proposals synthesized from different segments of customer and simulation data.
Your task is to merge, filter, and prioritize these inputs into a final, coherent product recommendation list.
Consolidate similar pain points, deduplicate duplicate features, and select the top 3-5 most impactful, evidence-backed features.

OUTPUT FORMAT: Return valid JSON with this exact structure:
{
  "analysis_summary": "Brief overall synthesis of what the data reveals",
  "top_pain_points": [
    {
      "pain_point": "Consolidated friction description",
      "frequency": "high/medium/low",
      "severity": "critical/major/minor",
      "customer_quotes": ["quote1", "quote2"],
      "affected_segments": ["segment1"]
    }
  ],
  "proposed_features": [
    {
      "title": "Deduplicated Feature Name",
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

DISCOVERY_JSON_SCHEMA = {
  "type": "object",
  "properties": {
    "analysis_summary": {"type": "string"},
    "top_pain_points": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "pain_point": {"type": "string"},
          "frequency": {"type": "string", "enum": ["high", "medium", "low"]},
          "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
          "customer_quotes": {
            "type": "array",
            "items": {"type": "string"}
          },
          "affected_segments": {
            "type": "array",
            "items": {"type": "string"}
          }
        },
        "required": ["pain_point", "frequency", "severity", "customer_quotes", "affected_segments"]
      }
    },
    "proposed_features": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "description": {"type": "string"},
          "target_users": {"type": "string"},
          "justification": {"type": "string"},
          "customer_evidence": {
            "type": "array",
            "items": {"type": "string"}
          },
          "affected_domains": {
            "type": "array",
            "items": {"type": "string"}
          },
          "priority": {"type": "string", "enum": ["P0", "P1", "P2"]},
          "effort_estimate": {"type": "string", "enum": ["small", "medium", "large"]},
          "competitive_advantage": {"type": "string"}
        },
        "required": ["title", "description", "target_users", "justification", "customer_evidence", "affected_domains", "priority", "effort_estimate", "competitive_advantage"]
      }
    }
  },
  "required": ["analysis_summary", "top_pain_points", "proposed_features"]
}


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
        t0 = time.time()
        logger.info("Layer 2: Starting SOTA Feature Discovery with Map-Reduce & Deduplication")

        # 1. Gather raw customer data and priority sort
        top_chunks = []
        if raw_chunks:
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
                if sentiment in ("NEGATIVE", "ANGER", "FRUSTRATION", "FEAR"):
                    score += 2
                return score

            sorted_chunks = sorted(raw_chunks, key=_chunk_priority, reverse=True)
            top_chunks = sorted_chunks[:100]  # Take top 100 for Map-Reduce

        # 2. Map Phase: Segment chunks into batches of 10 and map customer pain points
        intermediate_pain_points = []
        intermediate_features = []

        if top_chunks:
            chunk_size = 10
            batches = [top_chunks[i:i + chunk_size] for i in range(0, len(top_chunks), chunk_size)]
            logger.info("  Map Phase: Processing %d batches of customer chunks...", len(batches))
            
            for idx, batch in enumerate(batches):
                chunk_texts = []
                for chunk in batch:
                    text = getattr(chunk, 'text', getattr(chunk, 'content', str(chunk)))
                    if text:
                        chunk_texts.append(text[:500])
                
                batch_evidence = f"=== BATCH {idx+1} CUSTOMER DATA ===\n" + "\n---\n".join(chunk_texts)
                try:
                    res = await self._llm.analyze(
                        system_prompt=FEATURE_MAP_SYSTEM,
                        user_prompt=batch_evidence,
                        temperature=0.3,
                        max_tokens=4000
                    )
                    intermediate_pain_points.extend(res.get("top_pain_points", []))
                    intermediate_features.extend(res.get("proposed_features", []))
                except Exception as e:
                    logger.warning("Batch map step failed: %s", e)
        
        # 3. Gather simulation and hindsight evidence
        customer_data = await self._gather_customer_evidence(raw_chunks)
        simulation_data = await self._gather_simulation_evidence(behavioral_results)

        # 4. Reduce Phase: Merge intermediate findings with global context and Hindsight data
        reduce_input = (
            f"## Company Context\n"
            f"Company: {company.company_name}\n"
            f"Tech Stack: {', '.join(company.tech_stack) if company.tech_stack else 'Not specified'}\n"
            f"Priorities: {', '.join(company.current_priorities) if company.current_priorities else 'Not specified'}\n\n"
            f"## Hindsight & Simulation Insights\n"
            f"{simulation_data}\n\n"
            f"## Intermediate Mapped Pain Points\n"
            f"{json.dumps(intermediate_pain_points[:20], indent=2)}\n\n"
            f"## Intermediate Mapped Feature Ideas\n"
            f"{json.dumps(intermediate_features[:15], indent=2)}\n"
        )

        logger.info("  Reduce Phase: Synthesizing final feature proposals...")
        try:
            result = await self._llm.analyze(
                system_prompt=FEATURE_REDUCE_SYSTEM,
                user_prompt=reduce_input,
                json_schema=DISCOVERY_JSON_SCHEMA,
                temperature=0.3,
                max_tokens=6000
            )
        except Exception as e:
            logger.error("Feature Discovery Reduce step failed: %s", e)
            if existing_proposal:
                return [existing_proposal]
            return [FeatureProposal(
                title="Feature Discovery Failed",
                description=f"Reduce step failed: {e}. Please provide a feature_proposal.json.",
            )]

        # 5. Parse proposals
        proposals = self._parse_proposals(result, company)

        # 6. Semantic Deduplication check in Qdrant (Plane-2 discovery_data)
        deduplicated_proposals = []
        try:
            client = _get_qdrant()
            from qdrant_client.models import PointStruct
            from datetime import datetime, timezone
            
            for p in proposals:
                p_text = f"{p.title}: {p.description}"
                p_vec = _embed([p_text])[0]
                
                try:
                    search_results = await client.query_points(
                        collection_name="discovery_data",
                        query=p_vec,
                        limit=3,
                        with_payload=True
                    )
                    
                    match_found = False
                    for hit in search_results.points:
                        if hit.score > 0.85:
                            logger.info("Deduplication: Feature '%s' matches existing '%s' (similarity: %.3f). Merging...", p.title, hit.payload.get('title'), hit.score)
                            merged_proposal = await self._merge_proposals_llm(p, hit.payload)
                            
                            ts = datetime.now(timezone.utc).isoformat()
                            await client.upsert(
                                collection_name="discovery_data",
                                points=[PointStruct(
                                    id=hit.id,
                                    vector=p_vec,
                                    payload={
                                        "title": merged_proposal.title,
                                        "text": merged_proposal.description,
                                        "target_users": merged_proposal.target_users,
                                        "priority": merged_proposal.priority or "P1",
                                        "timestamp": ts,
                                        "run_id": self._session.run_id if self._session and hasattr(self._session, "run_id") else "global"
                                    }
                                )]
                            )
                            p = merged_proposal
                            match_found = True
                            break
                            
                    if not match_found:
                        ts = datetime.now(timezone.utc).isoformat()
                        from uuid import uuid4
                        new_id = str(uuid4())
                        await client.upsert(
                            collection_name="discovery_data",
                            points=[PointStruct(
                                id=new_id,
                                vector=p_vec,
                                payload={
                                    "title": p.title,
                                    "text": p.description,
                                    "target_users": p.target_users,
                                    "priority": p.priority or "P1",
                                    "timestamp": ts,
                                    "run_id": self._session.run_id if self._session and hasattr(self._session, "run_id") else "global"
                                }
                            )]
                        )
                except Exception as ex:
                    logger.debug("Qdrant query/upsert for deduplication failed: %s", ex)
                
                deduplicated_proposals.append(p)
            proposals = deduplicated_proposals
        except Exception as e:
            logger.debug("Could not connect to Qdrant or initialize client for deduplication: %s", e)

        # 7. If user provided a proposal, enrich it and put it first
        if existing_proposal:
            enriched = self._enrich_existing_proposal(existing_proposal, result)
            proposals = [p for p in proposals if p.title.lower() != enriched.title.lower()]
            proposals.insert(0, enriched)

        # 8. Retain discoveries into Hindsight
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

    async def _merge_proposals_llm(self, new_p: FeatureProposal, existing_payload: dict) -> FeatureProposal:
        """Merge two semantically identical proposals using the LLM client."""
        merge_prompt = (
            f"Merge the following two features:\n\n"
            f"Feature 1 (New):\n"
            f"- Title: {new_p.title}\n"
            f"- Description: {new_p.description}\n"
            f"- Target Users: {new_p.target_users}\n"
            f"- Affected Domains: {new_p.affected_domains}\n\n"
            f"Feature 2 (Existing):\n"
            f"- Title: {existing_payload.get('title')}\n"
            f"- Description: {existing_payload.get('text')}\n"
            f"- Target Users: {existing_payload.get('target_users')}\n"
            f"- Affected Domains: {existing_payload.get('affected_domains', [])}\n"
        )
        
        system_prompt = """You are a Product Manager. Merge these two duplicate feature proposals into a single, cohesive, enriched feature proposal. Preserve any concrete customer quotes, priorities, or metrics from both.
        Return a valid JSON with this exact structure:
        {
          "title": "Final merged title",
          "description": "Comprehensive merged description combining both descriptions and evidence.",
          "target_users": "Merged target users definition",
          "affected_domains": ["domain1", "domain2"]
        }"""
        
        try:
            res = await self._llm.analyze(
                system_prompt=system_prompt,
                user_prompt=merge_prompt,
                temperature=0.2,
                max_tokens=2000
            )
            return FeatureProposal(
                title=res.get("title", new_p.title),
                description=res.get("description", new_p.description),
                target_users=res.get("target_users", new_p.target_users),
                affected_domains=res.get("affected_domains", new_p.affected_domains),
                tech_stack=new_p.tech_stack,
                priority=new_p.priority,
                customer_segments=new_p.customer_segments
            )
        except Exception as e:
            logger.warning("Feature merge LLM call failed: %s. Returning original.", e)
            return new_p

    async def _gather_customer_evidence(self, raw_chunks: Optional[list] = None) -> str:
        """Gather customer evidence from Hindsight world bank and/or raw chunks."""
        evidence_parts = []

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

        if raw_chunks:
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
                if sentiment in ("NEGATIVE", "ANGER", "FRUSTRATION", "FEAR"):
                    score += 2
                return score

            sorted_chunks = sorted(raw_chunks, key=_chunk_priority, reverse=True)
            top_chunks = sorted_chunks[:30]

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

        if self._session:
            try:
                sim_data = await self._session.recall("simulation",
                    "What did simulated users say about their needs, frustrations, and desired features?")
                if sim_data and "no data" not in sim_data.lower():
                    evidence_parts.append(f"=== SIMULATION BEHAVIORAL INSIGHTS ===\n{sim_data}")
            except Exception as e:
                logger.debug("Hindsight simulation recall failed: %s", e)

        if behavioral_results and behavioral_results.agent_interactions:
            agent_quotes = []
            for agent_id, interactions in behavioral_results.agent_interactions.items():
                for interaction in interactions[:3]:
                    agent_quotes.append(f"  Agent {agent_id}: {interaction[:300]}")
            if agent_quotes:
                evidence_parts.append(
                    f"=== OASIS AGENT INTERACTIONS ({len(agent_quotes)} samples) ===\n"
                    + "\n".join(agent_quotes[:50])
                )

        return "\n\n".join(evidence_parts) if evidence_parts else "No simulation data available."

    def _parse_proposals(self, result: dict, company: CompanyContext) -> list[FeatureProposal]:
        """Parse LLM output into FeatureProposal objects."""
        proposals = []
        raw_features = result.get("proposed_features", [])

        for feat in raw_features[:5]:
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
        pain_points = analysis.get("top_pain_points", [])
        evidence_quotes = []
        for pp in pain_points:
            evidence_quotes.extend(pp.get("customer_quotes", []))

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

