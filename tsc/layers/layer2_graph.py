"""Layer 2: Knowledge Graph Builder.

Transforms enriched chunks into a queryable knowledge graph.
Extracts entities, finds relationships, builds graph, stores in memory.

Critical fixes:
  1. Deterministic entity ID generation (content-hash, not enumerate)
  2. Relationships reference entity IDs, not names
  3. Entity extraction with type validation and bounds checking
  4. Relationship validation & filtering (self-loops, broken refs, low conf)
  5. Graph integrity validation (orphans, broken refs, self-loops)

Optimizations:
  1. Entity normalization caching
  2. Stratified chunk sampling for LLM
  3. Batched LLM relationship extraction
  4. Early pruning of weak relationships
  5. Circular dependency detection
  6. Relationship strength recalculation on dedup
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import time
from collections import Counter, defaultdict
from typing import Any

from tsc.llm.base import LLMClient
from tsc.llm.prompts import RELATIONSHIP_SYSTEM, RELATIONSHIP_USER
from tsc.memory.graph_store import GraphStore
from tsc.models.chunks import EnrichedChunk, ProblemContextBundle
from tsc.models.graph import (
    EvidenceQuality,
    GraphEntity,
    GraphMetadata,
    GraphRelationship,
    KnowledgeGraph,
    RelationshipStrength,
    RelationshipType,
)

logger = logging.getLogger(__name__)

# Valid entity types for validation
_VALID_ENTITY_TYPES: frozenset[str] = frozenset(
    {"PERSON", "ORG", "PRODUCT", "CONSTRAINT", "PAIN_POINT", "METRIC", "UNKNOWN"}
)


class KnowledgeGraphBuilder:
    """Layer 2: Build a knowledge graph from enriched chunks."""

    def __init__(self, llm_client: LLMClient, graph_store: GraphStore) -> None:
        self._llm = llm_client
        self._store = graph_store
        self._norm_cache: dict[str, str] = {}  # OPT-1: normalization cache

    # ── Public API ───────────────────────────────────────────────────

    async def process(self, bundle: ProblemContextBundle) -> KnowledgeGraph:
        """Execute full Layer 2 pipeline with all fixes and optimizations."""
        t0 = time.time()
        logger.info(
            "Layer 2: Building knowledge graph from %d chunks", len(bundle.chunks)
        )

        # Step 2.1: Extract entities with validation (CRITICAL FIX #1, #3)
        entities = self._extract_entities(bundle.chunks)
        if not entities:
            raise ValueError("No entities extracted from documents")
        logger.info("✓ Extracted %d entities", len(entities))

        # Step 2.2a: Co-occurrence relationships (CRITICAL FIX #2 — IDs, not names)
        co_occur_rels = self._extract_cooccurrence_relationships(
            bundle.chunks, entities
        )
        logger.info("✓ Found %d co-occurrence relationships", len(co_occur_rels))

        # Step 2.2b: LLM relationships (OPT-2 stratified, OPT-3 batched)
        llm_rels = await self._extract_relationships_batched(
            bundle.chunks, entities
        )
        logger.info("✓ Found %d LLM relationships", len(llm_rels))

        # Combine all relationships
        all_rels = co_occur_rels + llm_rels
        logger.info("✓ Total before filtering: %d relationships", len(all_rels))

        # Step 2.2c: Prune weak relationships (OPT-4)
        all_rels, _prune_stats = self._prune_weak_relationships(
            all_rels, min_confidence=0.5
        )

        # Step 2.2d: Deduplicate and recalculate strength (OPT-6)
        relationships = self._deduplicate_relationships(all_rels)
        logger.info("✓ After dedup: %d relationships", len(relationships))

        # Step 2.2e: Validate all relationships (CRITICAL FIX #4)
        relationships, _filter_stats = self._filter_relationships(
            relationships, entities
        )
        logger.info("✓ After validation: %d relationships", len(relationships))

        # Step 2.3: Build graph
        graph = self._build_graph(entities, relationships)
        logger.info(
            "✓ Graph built: %d nodes, %d edges (density: %.3f)",
            graph.metadata.total_nodes,
            graph.metadata.total_edges,
            graph.metadata.graph_density,
        )

        # Step 2.3a: Validate graph integrity (CRITICAL FIX #5)
        integrity = self._validate_graph_integrity(graph)
        if not integrity["valid"]:
            logger.error("Graph integrity validation FAILED — proceeding with caution")

        # Step 2.3b: Detect circular dependencies (OPT-5)
        self._detect_circular_dependencies(graph)

        # Step 2.4: Store in memory
        stored = await self._store.store_graph(graph, bundle)
        logger.info("✓ Stored %d facts in memory", stored)

        elapsed = time.time() - t0
        logger.info(
            "Layer 2 complete: %d entities, %d relationships, %.1fs",
            len(entities),
            len(relationships),
            elapsed,
        )

        return graph

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #1: Deterministic Entity ID
    # ═════════════════════════════════════════════════════════════════

    def _generate_entity_id(self, entity_type: str, name: str) -> str:
        """Generate a deterministic entity ID from content hash."""
        content = f"{entity_type.upper()}:{name}".encode("utf-8")
        hash_suffix = hashlib.md5(content).hexdigest()[:8]
        return f"{entity_type.lower()}_{name}_{hash_suffix}"

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #3: Entity Extraction with Validation
    # ═════════════════════════════════════════════════════════════════

    def _extract_entities(
        self, chunks: list[EnrichedChunk]
    ) -> dict[str, GraphEntity]:
        """Extract, validate, and deduplicate entities from chunks."""
        raw: dict[str, dict[str, Any]] = {}  # normalized_name → aggregated data
        skipped_count = 0

        for chunk in chunks:
            for ent in chunk.entities:
                # Validate entity text
                if not ent.text or not ent.text.strip():
                    skipped_count += 1
                    continue

                name = self._normalize_entity_name(ent.text)

                # Check name validity
                if not name or len(name) < 2:
                    logger.debug(
                        "Skipping invalid entity name: '%s' → '%s'",
                        ent.text,
                        name,
                    )
                    skipped_count += 1
                    continue

                # Check type validity
                entity_type = ent.type if ent.type in _VALID_ENTITY_TYPES else "UNKNOWN"
                if ent.type != entity_type:
                    logger.debug(
                        "Normalizing invalid entity type: %s → %s",
                        ent.type,
                        entity_type,
                    )

                # Prevent unbounded growth
                if name not in raw and len(raw) >= 5000:
                    logger.warning(
                        "Entity count exceeding 5000, possible extraction issue"
                    )
                    break

                # Aggregate entity data
                if name not in raw:
                    raw[name] = {
                        "name": name,
                        "type": entity_type,
                        "full_name": ent.text,
                        "mentions": 0,
                        "raw_mentions": set(),
                        "chunk_ids": set(),
                        "contexts": [],
                        "confidences": [],
                        "urgencies": [],
                        "sentiments": Counter(),
                    }

                data = raw[name]
                data["mentions"] += 1
                data["raw_mentions"].add(ent.text)
                data["chunk_ids"].add(chunk.chunk_id)
                data["confidences"].append(ent.confidence)
                data["urgencies"].append(chunk.urgency)
                data["sentiments"][chunk.sentiment.label.value] += 1

                # Add context sample (max 5)
                if len(data["contexts"]) < 5:
                    ctx = chunk.text[:200]
                    if ctx not in data["contexts"]:
                        data["contexts"].append(ctx)

        if skipped_count > 0:
            logger.info(
                "Skipped %d invalid entities during extraction", skipped_count
            )

        # Convert to GraphEntity objects with deterministic IDs (FIX #1)
        entities: dict[str, GraphEntity] = {}
        for name, data in raw.items():
            entity_id = self._generate_entity_id(data["type"], name)

            entities[entity_id] = GraphEntity(
                id=entity_id,
                name=name,
                type=data["type"],
                full_name=data["full_name"],
                mentions=data["mentions"],
                raw_mentions=list(data["raw_mentions"]),
                chunk_ids=list(data["chunk_ids"]),
                contexts=data["contexts"],
                confidence=round(
                    sum(data["confidences"]) / len(data["confidences"]), 3
                ),
                average_urgency=round(
                    sum(data["urgencies"]) / len(data["urgencies"]), 1
                ),
                sentiment_distribution=dict(data["sentiments"]),
            )

        logger.info(
            "Extracted %d entities (%d skipped, avg mentions: %.1f)",
            len(entities),
            skipped_count,
            sum(e.mentions for e in entities.values()) / max(len(entities), 1),
        )

        return entities

    # ═════════════════════════════════════════════════════════════════
    # OPT-1: Entity Normalization Caching
    # ═════════════════════════════════════════════════════════════════

    def _normalize_entity_name(self, text: str) -> str:
        """Normalize entity name with caching."""
        if text in self._norm_cache:
            return self._norm_cache[text]

        normalized = text.strip().lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = re.sub(r"\s+", "_", normalized)

        self._norm_cache[text] = normalized
        return normalized

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #2: Co-occurrence Relationships (IDs, not names)
    # ═════════════════════════════════════════════════════════════════

    def _extract_cooccurrence_relationships(
        self,
        chunks: list[EnrichedChunk],
        entities: dict[str, GraphEntity],
    ) -> list[GraphRelationship]:
        """Extract co-occurrence relationships using entity IDs."""
        relationships: list[GraphRelationship] = []
        rel_idx = 0

        # Build entity name → ID mapping
        name_to_id: dict[str, str] = {e.name: eid for eid, e in entities.items()}

        for chunk in chunks:
            chunk_entity_names = [
                self._normalize_entity_name(e.text) for e in chunk.entities
            ]
            # Filter to known entities only
            valid = [n for n in chunk_entity_names if n in name_to_id]

            for i, src_name in enumerate(valid):
                src_id = name_to_id[src_name]

                for tgt_name in valid[i + 1 :]:
                    if src_name == tgt_name:
                        continue

                    tgt_id = name_to_id[tgt_name]

                    if src_id == tgt_id:  # Avoid self-loops
                        continue

                    relationships.append(
                        GraphRelationship(
                            id=f"rel_{rel_idx:04d}",
                            source_entity=src_id,  # ← ID, not name
                            target_entity=tgt_id,  # ← ID, not name
                            relationship_type=RelationshipType.MENTIONED_WITH,
                            confidence=0.7,
                            evidence_chunks=[chunk.chunk_id],
                            evidence_quality=EvidenceQuality.INFERRED,
                            evidence_count=1,
                            strength=RelationshipStrength.WEAK,
                        )
                    )
                    rel_idx += 1

        logger.info(
            "Extracted %d co-occurrence relationships", len(relationships)
        )
        return relationships

    # ═════════════════════════════════════════════════════════════════
    # OPT-2: Stratified Sampling for LLM
    # ═════════════════════════════════════════════════════════════════

    def _get_stratified_sample(
        self,
        chunks: list[EnrichedChunk],
        max_chunks: int = 30,
    ) -> list[EnrichedChunk]:
        """Sample chunks stratified by source type for representative coverage."""
        by_source: dict[str, list[EnrichedChunk]] = defaultdict(list)

        for chunk in chunks:
            by_source[chunk.source_type].append(chunk)

        sampled: list[EnrichedChunk] = []
        for _source_type, source_chunks in by_source.items():
            # Sample ~20% from each source (min 1)
            sample_size = max(1, len(source_chunks) // 5)
            sampled.extend(
                random.sample(
                    source_chunks, min(sample_size, len(source_chunks))
                )
            )

        # Shuffle and limit
        random.shuffle(sampled)
        result = sampled[:max_chunks]

        logger.info(
            "Sampled %d chunks stratified from %d total (%d sources)",
            len(result),
            len(chunks),
            len(by_source),
        )
        return result

    # ═════════════════════════════════════════════════════════════════
    # OPT-3: Batched LLM Relationship Extraction (with FIX #2 — IDs)
    # ═════════════════════════════════════════════════════════════════

    async def _extract_relationships_batched(
        self,
        chunks: list[EnrichedChunk],
        entities: dict[str, GraphEntity],
        batch_size: int = 10,
    ) -> list[GraphRelationship]:
        """Extract relationships using batched LLM calls with stratified sample."""
        relationships: list[GraphRelationship] = []
        rel_idx = 0

        # Get top entities sorted by mentions
        top_entities = sorted(
            entities.values(), key=lambda e: e.mentions, reverse=True
        )[:50]

        if not top_entities or not chunks:
            return relationships

        # Stratified sample of chunks for context (OPT-2)
        sample_chunks = self._get_stratified_sample(chunks, max_chunks=20)
        combined_text = "\n---\n".join(c.text for c in sample_chunks)[:3000]

        # Batch entities for separate LLM calls
        entity_batches = [
            top_entities[i : i + batch_size]
            for i in range(0, len(top_entities), batch_size)
        ]

        # Build name → ID mapping for reference resolution
        name_to_id: dict[str, str] = {
            e.name: eid for eid, e in entities.items()
        }

        for batch_idx, entity_batch in enumerate(entity_batches):
            try:
                entity_list = [
                    {"name": e.name, "type": e.type} for e in entity_batch
                ]

                prompt = RELATIONSHIP_USER.render(
                    entities=entity_list,
                    text=combined_text,
                )

                result = await self._llm.analyze(
                    system_prompt=RELATIONSHIP_SYSTEM,
                    user_prompt=prompt,
                    temperature=0.3,
                    max_tokens=1500,
                )

                batch_count = 0
                for rel_data in result.get("relationships", []):
                    src_name = self._normalize_entity_name(
                        rel_data.get("source", "")
                    )
                    tgt_name = self._normalize_entity_name(
                        rel_data.get("target", "")
                    )

                    # Validate both exist in our entity set (FIX #2)
                    src_id = name_to_id.get(src_name)
                    if not src_id:
                        logger.debug(
                            "LLM referenced unknown entity: %s", src_name
                        )
                        continue

                    tgt_id = name_to_id.get(tgt_name)
                    if not tgt_id:
                        logger.debug(
                            "LLM referenced unknown entity: %s", tgt_name
                        )
                        continue

                    if src_id == tgt_id:  # Avoid self-loops
                        continue

                    try:
                        rel_type = RelationshipType(
                            rel_data.get("type", "MENTIONED_WITH")
                        )
                    except ValueError:
                        rel_type = RelationshipType.MENTIONED_WITH

                    relationships.append(
                        GraphRelationship(
                            id=f"rel_{rel_idx:04d}",
                            source_entity=src_id,  # ← ID, not name
                            target_entity=tgt_id,  # ← ID, not name
                            relationship_type=rel_type,
                            confidence=rel_data.get("confidence", 0.7),
                            weight=rel_data.get("weight", 0.0),
                            evidence_chunks=[],
                            evidence_quality=EvidenceQuality.DIRECT,
                            evidence_count=1,
                            strength=RelationshipStrength.MEDIUM,
                        )
                    )
                    rel_idx += 1
                    batch_count += 1

                logger.info(
                    "Batch %d/%d: extracted %d relationships",
                    batch_idx + 1,
                    len(entity_batches),
                    batch_count,
                )

            except Exception as e:
                logger.warning("LLM batch %d failed: %s", batch_idx + 1, e)
                continue

        return relationships

    # ═════════════════════════════════════════════════════════════════
    # OPT-4: Early Pruning of Weak Relationships
    # ═════════════════════════════════════════════════════════════════

    def _prune_weak_relationships(
        self,
        relationships: list[GraphRelationship],
        min_confidence: float = 0.5,
        min_evidence: int = 1,
    ) -> tuple[list[GraphRelationship], dict[str, int]]:
        """Prune weak relationships before dedup."""
        stats: dict[str, int] = {
            "total_input": len(relationships),
            "pruned_confidence": 0,
            "pruned_evidence": 0,
            "total_output": 0,
        }

        pruned: list[GraphRelationship] = []
        for rel in relationships:
            if rel.confidence < min_confidence:
                stats["pruned_confidence"] += 1
                continue

            if rel.evidence_count < min_evidence:
                stats["pruned_evidence"] += 1
                continue

            pruned.append(rel)

        stats["total_output"] = len(pruned)

        logger.info(
            "Relationship pruning: %d → %d (low_conf=%d, low_evidence=%d)",
            stats["total_input"],
            stats["total_output"],
            stats["pruned_confidence"],
            stats["pruned_evidence"],
        )

        return pruned, stats

    # ═════════════════════════════════════════════════════════════════
    # OPT-6: Deduplication with Strength Recalculation
    # ═════════════════════════════════════════════════════════════════

    def _calculate_relationship_strength(
        self, evidence_count: int, confidence: float
    ) -> RelationshipStrength:
        """Determine strength based on evidence and confidence."""
        strength_score = evidence_count * confidence

        if strength_score >= 3.0:
            return RelationshipStrength.STRONG
        elif strength_score >= 1.5:
            return RelationshipStrength.MEDIUM
        else:
            return RelationshipStrength.WEAK

    def _deduplicate_relationships(
        self, rels: list[GraphRelationship]
    ) -> list[GraphRelationship]:
        """Merge duplicate relationships with strength recalculation."""
        seen: dict[str, GraphRelationship] = {}

        for rel in rels:
            key = (
                f"{rel.source_entity}_{rel.target_entity}"
                f"_{rel.relationship_type.value}"
            )

            if key in seen:
                existing = seen[key]
                existing.evidence_count += rel.evidence_count
                existing.evidence_chunks.extend(rel.evidence_chunks)
                # Deduplicate chunk IDs
                existing.evidence_chunks = list(set(existing.evidence_chunks))
                existing.confidence = max(existing.confidence, rel.confidence)
                # Recalculate strength (OPT-6)
                existing.strength = self._calculate_relationship_strength(
                    existing.evidence_count, existing.confidence
                )
            else:
                seen[key] = rel

        logger.info(
            "Deduplication: %d → %d relationships", len(rels), len(seen)
        )

        return list(seen.values())

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #4: Relationship Validation & Filtering
    # ═════════════════════════════════════════════════════════════════

    def _filter_relationships(
        self,
        relationships: list[GraphRelationship],
        entities: dict[str, GraphEntity],
        min_confidence: float = 0.5,
    ) -> tuple[list[GraphRelationship], dict[str, int]]:
        """Filter relationships by reference integrity, self-loops, and confidence."""
        stats: dict[str, int] = {
            "total_input": len(relationships),
            "invalid_refs": 0,
            "self_loops": 0,
            "low_confidence": 0,
            "total_output": 0,
        }

        filtered: list[GraphRelationship] = []
        for rel in relationships:
            # Validate reference integrity
            if rel.source_entity not in entities:
                stats["invalid_refs"] += 1
                continue
            if rel.target_entity not in entities:
                stats["invalid_refs"] += 1
                continue

            # Skip self-loops
            if rel.source_entity == rel.target_entity:
                stats["self_loops"] += 1
                continue

            # Skip low confidence
            if rel.confidence < min_confidence:
                stats["low_confidence"] += 1
                continue

            filtered.append(rel)

        stats["total_output"] = len(filtered)

        logger.info(
            "Relationship filtering: %d → %d "
            "(invalid_refs=%d, self_loops=%d, low_conf=%d)",
            stats["total_input"],
            stats["total_output"],
            stats["invalid_refs"],
            stats["self_loops"],
            stats["low_confidence"],
        )

        return filtered, stats

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #5: Graph Integrity Validation
    # ═════════════════════════════════════════════════════════════════

    def _validate_graph_integrity(
        self, graph: KnowledgeGraph
    ) -> dict[str, Any]:
        """Comprehensive graph integrity check."""
        issues: dict[str, Any] = {
            "orphaned_nodes": [],
            "broken_relationships": [],
            "self_loops": [],
            "valid": True,
        }

        # Check all relationship endpoints exist
        for rel in graph.edges:
            if rel.source_entity not in graph.nodes:
                issues["broken_relationships"].append(
                    f"source:{rel.source_entity}"
                )
                issues["valid"] = False

            if rel.target_entity not in graph.nodes:
                issues["broken_relationships"].append(
                    f"target:{rel.target_entity}"
                )
                issues["valid"] = False

            if rel.source_entity == rel.target_entity:
                issues["self_loops"].append(rel.id)
                issues["valid"] = False

        # Check for orphaned nodes (no relationships)
        nodes_with_edges: set[str] = set()
        for rel in graph.edges:
            nodes_with_edges.add(rel.source_entity)
            nodes_with_edges.add(rel.target_entity)

        for nid in graph.nodes:
            if nid not in nodes_with_edges:
                issues["orphaned_nodes"].append(nid)

        if issues["broken_relationships"]:
            logger.error(
                "GRAPH CORRUPTION: %d broken relationships",
                len(issues["broken_relationships"]),
            )

        if issues["self_loops"]:
            logger.warning(
                "GRAPH ANOMALY: %d self-loops", len(issues["self_loops"])
            )

        if issues["valid"]:
            logger.info(
                "Graph integrity validation PASSED "
                "(orphaned=%d, broken=0, self_loops=0)",
                len(issues["orphaned_nodes"]),
            )
        else:
            logger.error("Graph integrity validation FAILED")

        return issues

    # ═════════════════════════════════════════════════════════════════
    # OPT-5: Circular Dependency Detection
    # ═════════════════════════════════════════════════════════════════

    def _detect_circular_dependencies(
        self, graph: KnowledgeGraph
    ) -> dict[str, Any]:
        """Detect circular dependency chains using DFS."""

        def _find_cycles(
            current: str,
            visited: set[str],
            rec_stack: set[str],
            adj: dict[str, list[str]],
        ) -> list[list[str]]:
            visited.add(current)
            rec_stack.add(current)
            cycles: list[list[str]] = []

            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    cycles.extend(
                        _find_cycles(neighbor, visited, rec_stack, adj)
                    )
                elif neighbor in rec_stack:
                    cycles.append([current, neighbor])

            rec_stack.remove(current)
            return cycles

        # Build directed adjacency list
        adj: dict[str, list[str]] = defaultdict(list)
        for rel in graph.edges:
            adj[rel.source_entity].append(rel.target_entity)

        # Find all cycles
        visited: set[str] = set()
        all_cycles: list[list[str]] = []

        for node in adj:
            if node not in visited:
                cycles = _find_cycles(node, visited, set(), adj)
                all_cycles.extend(cycles)

        result: dict[str, Any] = {
            "has_cycles": len(all_cycles) > 0,
            "cycle_count": len(all_cycles),
            "cycles": all_cycles[:10],  # Limit output
        }

        if result["has_cycles"]:
            logger.warning(
                "Circular dependencies detected: %d cycles",
                result["cycle_count"],
            )
        else:
            logger.info("No circular dependencies found")

        return result

    # ═════════════════════════════════════════════════════════════════
    # Step 2.3: Graph Construction
    # ═════════════════════════════════════════════════════════════════

    def _build_graph(
        self,
        entities: dict[str, GraphEntity],
        relationships: list[GraphRelationship],
    ) -> KnowledgeGraph:
        """Build queryable graph with indices."""
        # Build indices
        by_name: dict[str, list[str]] = defaultdict(list)
        by_type: dict[str, list[str]] = defaultdict(list)
        by_rel_type: dict[str, list[str]] = defaultdict(list)
        adjacency: dict[str, list[str]] = defaultdict(list)

        for eid, entity in entities.items():
            by_name[entity.name].append(eid)
            by_type[entity.type].append(eid)

        for rel in relationships:
            by_rel_type[rel.relationship_type.value].append(rel.id)
            # Add to adjacency (both directions for undirected traversal)
            adjacency[rel.source_entity].append(rel.id)
            adjacency[rel.target_entity].append(rel.id)

        # Calculate metrics
        total_nodes = len(entities)
        total_edges = len(relationships)
        node_types = dict(Counter(e.type for e in entities.values()))
        avg_edges = total_edges / max(total_nodes, 1)
        avg_confidence = sum(e.confidence for e in entities.values()) / max(
            total_nodes, 1
        )
        max_possible = (
            total_nodes * (total_nodes - 1) / 2 if total_nodes > 1 else 1
        )
        density = total_edges / max_possible

        return KnowledgeGraph(
            nodes=entities,
            edges=relationships,
            indices={
                "by_name": dict(by_name),
                "by_type": dict(by_type),
                "by_rel_type": dict(by_rel_type),
                "adjacency": dict(adjacency),
            },
            metadata=GraphMetadata(
                total_nodes=total_nodes,
                total_edges=total_edges,
                node_types=node_types,
                avg_edges_per_node=round(avg_edges, 1),
                avg_node_confidence=round(avg_confidence, 2),
                graph_density=round(density, 3),
            ),
        )

    # ═════════════════════════════════════════════════════════════════
    # Diagnostics
    # ═════════════════════════════════════════════════════════════════

    def get_layer2_diagnostics(
        self, graph: KnowledgeGraph
    ) -> dict[str, Any]:
        """Get comprehensive diagnostics for debugging."""
        # Entity diagnostics
        entity_types: dict[str, int] = defaultdict(int)
        high_confidence = 0
        low_confidence = 0
        total_mentions = 0

        for entity in graph.nodes.values():
            entity_types[entity.type] += 1
            if entity.confidence >= 0.8:
                high_confidence += 1
            elif entity.confidence < 0.5:
                low_confidence += 1
            total_mentions += entity.mentions

        avg_mentions = total_mentions / max(len(graph.nodes), 1)

        # Relationship diagnostics
        rel_types: dict[str, int] = defaultdict(int)
        strong_rels = 0
        weak_rels = 0
        conf_sum = 0.0

        for rel in graph.edges:
            rel_types[rel.relationship_type.value] += 1
            if rel.strength == RelationshipStrength.STRONG:
                strong_rels += 1
            elif rel.strength == RelationshipStrength.WEAK:
                weak_rels += 1
            conf_sum += rel.confidence

        avg_rel_confidence = conf_sum / max(len(graph.edges), 1)

        # Node connectivity
        in_degree: dict[str, int] = defaultdict(int)
        out_degree: dict[str, int] = defaultdict(int)

        for rel in graph.edges:
            out_degree[rel.source_entity] += 1
            in_degree[rel.target_entity] += 1

        isolated_nodes = sum(
            1
            for nid in graph.nodes
            if nid not in out_degree and nid not in in_degree
        )

        return {
            "entities": {
                "total": len(graph.nodes),
                "by_type": dict(entity_types),
                "high_confidence": high_confidence,
                "low_confidence": low_confidence,
                "avg_mentions": round(avg_mentions, 1),
            },
            "relationships": {
                "total": len(graph.edges),
                "by_type": dict(rel_types),
                "strong": strong_rels,
                "weak": weak_rels,
                "avg_confidence": round(avg_rel_confidence, 3),
            },
            "connectivity": {
                "isolated_nodes": isolated_nodes,
                "avg_out_degree": round(
                    len(graph.edges) / max(len(graph.nodes), 1), 2
                ),
                "max_out_degree": (
                    max(out_degree.values()) if out_degree else 0
                ),
            },
            "graph_properties": {
                "density": graph.metadata.graph_density,
                "nodes": graph.metadata.total_nodes,
                "edges": graph.metadata.total_edges,
            },
            "cache_stats": {
                "norm_cache_size": len(self._norm_cache),
            },
        }
