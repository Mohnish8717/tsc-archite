"""Knowledge graph storage and retrieval via Zep."""

from __future__ import annotations

import logging
from typing import Any, Optional

from tsc.memory.zep_client import ZepMemoryClient
from tsc.models.chunks import ProblemContextBundle
from tsc.models.graph import GraphEntity, GraphRelationship, KnowledgeGraph

logger = logging.getLogger(__name__)


class GraphStore:
    """Stores and retrieves knowledge graph data through Zep.

    Provides methods to:
    - Ingest entities and relationships as facts
    - Retrieve context for stakeholder profiling
    - Search for domain-specific facts
    """

    def __init__(self, zep_client: ZepMemoryClient):
        self._zep = zep_client
        self._local_facts: list[dict[str, Any]] = []

    async def store_graph(
        self, graph: KnowledgeGraph, bundle: ProblemContextBundle
    ) -> int:
        """Store graph entities with full document context in Zep."""
        facts = []

        # Store entities as rich profile facts
        for entity in graph.nodes.values():
            # Find all chunks this entity appears in
            # Note: entity.name is normalized, chunk mentions are raw
            entity_chunks = [
                chunk
                for chunk in bundle.chunks
                if any(
                    e.text.strip().lower() == entity.name.replace("_", " ")
                    or e.text.strip().lower() == entity.name
                    for e in chunk.entities
                )
            ]

            if not entity_chunks:
                # Fallback: if no chunks found by exact match, use the one stored in entity.chunk_ids
                chunk_map = {c.chunk_id: c for c in bundle.chunks}
                entity_chunks = [
                    chunk_map[cid] for cid in entity.chunk_ids if cid in chunk_map
                ]

            # Get sentiment and urgency aggregation
            sentiments = dict(entity.sentiment_distribution)
            avg_urgency = entity.average_urgency

            # Collect actual mentions with context (top 10)
            mentions_with_context = []
            for chunk in entity_chunks[:10]:
                mentions_with_context.append(
                    {
                        "text": chunk.text[:300],
                        "sentiment": chunk.sentiment.label.value,
                        "urgency": chunk.urgency,
                        "topic": chunk.topic_category.value,
                        "source": chunk.source_type,
                        "chunk_id": chunk.chunk_id,
                    }
                )

            fact_text = (
                f"{entity.full_name or entity.name} ({entity.type}). "
                f"Mentioned {entity.mentions} times. "
                f"Sentiment: {sentiments}. "
                f"Avg Urgency: {avg_urgency:.1f}/5. "
                f"Context: {'; '.join(m['text'][:100] for m in mentions_with_context[:3])}"
            )

            facts.append(
                {
                    "fact": fact_text,
                    "entities": [entity.name],
                    "metadata": {
                        "type": "entity_profile",
                        "entity_type": entity.type,
                        "entity_id": entity.id,
                        "mentions": entity.mentions,
                        "confidence": entity.confidence,
                        "average_urgency": avg_urgency,
                        "sentiment_distribution": sentiments,
                        "mention_contexts": mentions_with_context,
                        "source_types": list(
                            set(c.source_type for c in entity_chunks)
                        ),
                    },
                }
            )

        # Store relationships as facts
        for rel in graph.edges:
            src = graph.nodes.get(rel.source_entity)
            tgt = graph.nodes.get(rel.target_entity)
            if not src or not tgt:
                continue

            fact_text = (
                f"{src.name} ({rel.relationship_type.value}) {tgt.name}. "
                f"Evidence: {rel.evidence_count} chunks. "
                f"Confidence: {rel.confidence:.2f}"
            )

            facts.append(
                {
                    "fact": fact_text,
                    "entities": [src.name, tgt.name],
                    "metadata": {
                        "type": "relationship",
                        "relationship_type": rel.relationship_type.value,
                        "confidence": rel.confidence,
                        "strength": rel.strength.value,
                        "evidence_count": rel.evidence_count,
                        "evidence_chunks": rel.evidence_chunks,
                    },
                }
            )

        # Keep local copy for fallback
        self._local_facts.extend(facts)

        # Store in Zep
        stored = await self._zep.ingest_facts(facts)
        logger.info(
            "✓ Stored %d rich facts in Zep (%d entities, %d relationships)",
            stored,
            len(graph.nodes),
            len(graph.edges),
        )
        return stored

    async def retrieve_context(
        self, query: str, limit: int = 30
    ) -> list[dict[str, Any]]:
        """Retrieve relevant facts for a query as rich objects."""
        # Try Zep first
        zep_results = await self._zep.search_facts(query, limit=limit)
        if zep_results:
            # Return full Zep result (contains 'fact' and 'metadata')
            return zep_results

        # Fallback: local keyword search
        query_lower = query.lower()
        matched = []
        for fact in self._local_facts:
            # Check fact text or entities
            if query_lower in fact["fact"].lower() or any(
                query_lower in e.lower() for e in fact.get("entities", [])
            ):
                matched.append(fact)
        return matched[:limit]

    async def retrieve_stakeholder_context(
        self,
        stakeholder_name: str,
        stakeholder_role: str,
    ) -> dict[str, list[Any]]:
        """Retrieve rich stakeholder context including document-level evidence."""
        try:
            # Query Zep for facts about this person (expanded limit)
            facts = await self._zep.search_facts(
                f"{stakeholder_name} {stakeholder_role}", limit=100
            )

            personal_facts = []
            org_context = []
            constraint_context = []

            for fact_data in facts:
                metadata = fact_data.get("metadata", {})
                fact_text = fact_data.get("fact", "")

                # 1. Extract rich mention contexts if available
                mention_contexts = metadata.get("mention_contexts", [])
                if mention_contexts:
                    for mention in mention_contexts:
                        personal_facts.append(
                            {
                                "text": mention["text"],
                                "sentiment": mention.get("sentiment", "NEUTRAL"),
                                "urgency": mention.get("urgency", 3),
                                "topic": mention.get("topic", "feedback"),
                                "source": mention.get("source", "unknown"),
                                "confidence": metadata.get("confidence", 0.8),
                            }
                        )
                else:
                    # Fallback to standard fact summary
                    personal_facts.append(
                        {
                            "text": fact_text,
                            "sentiment": "NEUTRAL",
                            "urgency": metadata.get("urgency", 3),
                            "topic": "general",
                            "source": "summary",
                            "confidence": metadata.get("confidence", 0.7),
                        }
                    )

                # 2. Extract org context (from relationships)
                if metadata.get("type") == "relationship":
                    org_context.append(fact_text)

                # 3. Extract constraints
                if (
                    metadata.get("entity_type") == "CONSTRAINT"
                    or metadata.get("urgency", 0) >= 4
                ):
                    constraint_context.append(fact_text)

            # Deduplicate and return (limit to top 50 personal facts)
            return {
                "personal_facts": personal_facts[:50],
                "org_context": list(set(org_context)),
                "constraint_context": list(set(constraint_context)),
            }

        except Exception as e:
            logger.error("Failed to retrieve rich stakeholder context: %s", e)
            return {
                "personal_facts": [],
                "org_context": [],
                "constraint_context": [],
            }

    async def retrieve_customer_context(
        self,
        segment_type: str,
        use_case: str,
    ) -> dict[str, list[Any]]:
        """Retrieve rich external customer context (mirrors stakeholder pattern)."""
        # 1. Retrieve raw contexts
        personal_raw = await self.retrieve_context(
            f"Facts about {segment_type} customers", limit=30
        )
        org_raw = await self.retrieve_context(
            f"Use case: {use_case}", limit=20
        )
        constraint_raw = await self.retrieve_context(
            "Market constraints competitors pricing", limit=20
        )

        # 2. Extract rich mention contexts (HACK: using the same extractor logic)
        def _extract_rich(raw_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
            results = []
            for f in raw_facts:
                m_ctx = f.get("metadata", {}).get("mention_contexts", [])
                if m_ctx:
                    for m in m_ctx:
                        results.append({
                            "text": m["text"],
                            "sentiment": m.get("sentiment", "NEUTRAL"),
                            "urgency": m.get("urgency", 3),
                            "source": m.get("source", "external"),
                            "confidence": f.get("metadata", {}).get("confidence", 0.7)
                        })
                else:
                    results.append({
                        "text": f["fact"],
                        "sentiment": "NEUTRAL",
                        "urgency": f.get("metadata", {}).get("urgency", 3),
                        "source": "summary",
                        "confidence": 0.6
                    })
            return results

        return {
            "personal_facts": _extract_rich(personal_raw),
            "org_context": [f["fact"] for f in org_raw],
            "constraint_context": [f["fact"] for f in constraint_raw],
        }
