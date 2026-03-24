"""Zep Cloud SDK wrapper for TSC memory operations.

Uses GROUP GRAPHS for shared pipeline-level knowledge storage.
Falls back to in-memory keyword index when Zep is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

ZEP_CHAR_LIMIT = 9_500  # graph.add hard limit is 10,000; use 9,500 for safety


class ZepMemoryClient:
    """Wrapper around the Zep Cloud SDK for knowledge graph operations.

    Uses GROUP GRAPHS (not conversational memory sessions).
    Falls back to in-memory keyword search when Zep is unavailable.
    """

    def __init__(self, api_key: str):
        self._client: Any = None
        self._zep_available: bool = False
        self._group_id: Optional[str] = None
        self._user_id: Optional[str] = None
        self._facts_saved: int = 0

        # Local fallback storage
        self._local_facts: list[dict[str, Any]] = []
        self._local_index: dict[str, list[int]] = {}
        self._index_lock = threading.Lock()
        self._local_cap: int = 50_000

        try:
            from zep_cloud.client import AsyncZep
            self._client = AsyncZep(api_key=api_key)
            self._zep_available = True
            logger.info("Zep client initialized (zep-cloud v2, graph API)")
        except ImportError:
            logger.warning(
                "zep-cloud not installed — running in local-only mode. "
                "Install with: pip install zep-cloud"
            )
        except Exception as e:
            logger.warning("Zep client init failed: %s — running in local-only mode", e)

    async def initialize_session(self, session_name: str = "tsc_evaluation") -> str:
        """Create a new group for this pipeline run."""
        self._user_id = f"tsc_{session_name}_{uuid.uuid4().hex[:8]}"
        self._group_id = f"tsc_group_{uuid.uuid4().hex[:8]}"

        if not self._zep_available or not self._client:
            logger.debug("Zep unavailable — using group_id locally only: %s", self._group_id)
            return self._group_id

        try:
            await self._client.user.add(
                user_id=self._user_id,
                first_name="TSC",
                last_name=session_name[:50],
            )
            await self._client.group.add(
                group_id=self._group_id,
                description=f"TSC pipeline run: {session_name}",
            )
            logger.info(
                "Zep session initialized: user=%s, group=%s",
                self._user_id, self._group_id,
            )
        except Exception as e:
            logger.warning("Zep session init failed (non-fatal): %s", e)
            self._zep_available = False

        return self._group_id or session_name

    async def ingest_facts(
        self, facts: list[dict[str, Any]], batch_size: int = 5
    ) -> int:
        """Store facts in Zep graph AND local index.

        Always stores locally regardless of Zep availability.
        Returns count of facts stored.
        """
        if not facts:
            return 0

        # Always store locally
        self._store_locally(facts)

        # Optionally store in Zep
        if not self._zep_available or not self._client or not self._group_id:
            self._facts_saved += len(facts)
            return len(facts)

        # Serialize facts to text chunks within char limit
        chunks = self._serialize_to_chunks(facts)

        try:
            # Try graph.add for each chunk
            await self._ingest_sequential(chunks)
            logger.info(
                "Ingested %d facts (%d chunks) into Zep group %s",
                len(facts), len(chunks), self._group_id,
            )
        except Exception as e:
            logger.warning("Zep ingestion failed: %s — facts stored locally only", e)

        self._facts_saved += len(facts)
        return len(facts)

    async def _ingest_sequential(self, chunks: list[str]) -> None:
        """Add chunks one at a time with delay to avoid rate limits."""
        for i, chunk in enumerate(chunks):
            try:
                await self._client.graph.add(
                    group_id=self._group_id,
                    type="text",
                    data=chunk,
                )
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning("Sequential graph.add chunk %d failed: %s", i, e)

    async def search_facts(
        self,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search facts via Zep graph.search with local fallback."""
        if not query:
            return self._local_facts[:limit]

        if limit <= 0:
            return []

        # Try Zep first
        if self._zep_available and self._client and self._group_id:
            try:
                results = await self._client.graph.search(
                    group_id=self._group_id,
                    query=query[:256],
                    scope="edges",
                    limit=limit,
                )
                parsed = []
                edges = getattr(results, "edges", None) or []
                for edge in edges:
                    # MiroFish Optimization: Extract temporal metadata if present
                    metadata = getattr(edge, "metadata", {}) or {}
                    parsed.append({
                        "fact": getattr(edge, "fact", str(edge)),
                        "score": float(getattr(edge, "score", 0.0) or 0.0),
                        "uuid": str(getattr(edge, "uuid_", "") or ""),
                        "created_at": metadata.get("created_at"),
                        "expired_at": metadata.get("expired_at")
                    })
                if parsed:
                    logger.debug("Zep graph.search returned %d results", len(parsed))
                    return parsed
            except Exception as e:
                logger.debug("Zep graph.search failed (%s) — using local fallback", e)

        # Local keyword fallback
        return self._local_keyword_search(query, limit)

    async def get_opinion_evolution(self, entity_name: str, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve historical stances for a specific entity to analyze opinion shifts (Temporal RAG)."""
        if not self._zep_available or not self._client or not self._group_id:
            # Local fallback search for entity mentions
            return [f for f in self._local_facts if entity_name.lower() in str(f).lower()][:limit]

        try:
            # Query for all edges related to the entity
            results = await self._client.graph.search(
                group_id=self._group_id,
                query=f"Stance and opinion of {entity_name}",
                scope="edges",
                limit=limit,
            )
            parsed = []
            edges = getattr(results, "edges", None) or []
            for edge in edges:
                metadata = getattr(edge, "metadata", {}) or {}
                parsed.append({
                    "fact": getattr(edge, "fact", str(edge)),
                    "timestamp": metadata.get("created_at"),
                    "is_active": metadata.get("expired_at") is None
                })
            # Sort by timestamp if available
            return sorted(parsed, key=lambda x: x.get("timestamp") or "", reverse=True)
        except Exception as e:
            logger.warning("Opinion evolution search failed: %s", e)
            return []

    async def get_graph_entities(self) -> list[dict[str, Any]]:
        """Retrieve entities from the Zep knowledge graph."""
        if not self._client or not self._group_id or not self._zep_available:
            return []

        try:
            episodes = await self._client.graph.search(
                group_id=self._group_id,
                query="all entities",
                limit=100,
            )
            return [
                {"name": str(e), "source": "zep_graph"}
                for e in (
                    getattr(episodes, "results", None)
                    or getattr(episodes, "edges", None)
                    or episodes
                    or []
                )
            ]
        except Exception as e:
            logger.warning("Zep graph retrieval failed: %s", e)
            return []

    # ── Local storage ────────────────────────────────────────────────

    def _store_locally(self, facts: list[dict[str, Any]]) -> None:
        """Thread-safe local storage with eviction when cap is reached."""
        with self._index_lock:
            # Evict oldest 10% if at capacity
            if len(self._local_facts) + len(facts) > self._local_cap:
                evict_count = max(1, self._local_cap // 10)
                self._local_facts = self._local_facts[evict_count:]
                # Rebuild index after eviction (simpler than adjusting indices)
                self._local_index = {}
                for idx, fact in enumerate(self._local_facts):
                    fact_text = str(fact.get("fact", "") or fact.get("data", "") or str(fact))
                    for word in self._tokenize(fact_text):
                        self._local_index.setdefault(word, []).append(idx)
                logger.info("Evicted %d oldest facts from local store", evict_count)

            start_idx = len(self._local_facts)
            self._local_facts.extend(facts)

            # Build keyword index for new facts
            for i, fact in enumerate(facts):
                fact_text = str(fact.get("fact", "") or fact.get("data", "") or str(fact))
                for word in self._tokenize(fact_text):
                    self._local_index.setdefault(word, []).append(start_idx + i)

    def _local_keyword_search(
        self, query: str, limit: int
    ) -> list[dict[str, Any]]:
        """Score local facts by keyword overlap, with substring fallback."""
        query_words = self._tokenize(query)
        if not query_words:
            return self._local_facts[:limit]

        with self._index_lock:
            scores: dict[int, int] = {}
            for word in query_words:
                for idx in self._local_index.get(word, []):
                    scores[idx] = scores.get(idx, 0) + 1

            if not scores:
                query_lower = query.lower()
                for i, fact in enumerate(self._local_facts):
                    fact_text = str(fact.get("fact", "")).lower()
                    if query_lower in fact_text:
                        scores[i] = 1

            if not scores:
                return []

            ranked = sorted(scores.keys(), key=lambda i: (scores[i], i), reverse=True)
            return [
                {**self._local_facts[i], "score": float(scores[i])}
                for i in ranked[:limit]
                if i < len(self._local_facts)
            ]

    def _tokenize(self, text: str) -> list[str]:
        """Extract index-worthy words: lowercase, length > 3, alpha only."""
        return [w.lower() for w in str(text).split() if len(w) > 3 and w.isalpha()]

    def _serialize_to_chunks(self, facts: list[dict[str, Any]]) -> list[str]:
        """Serialize facts to text chunks within ZEP_CHAR_LIMIT."""
        chunks: list[str] = []
        current_lines: list[str] = []
        current_len = 0

        for fact in facts:
            # MiroFish Optimization: Include temporal metadata in the line for search grounding
            fact_prefix = ""
            if fact.get("created_at"):
                fact_prefix = f"[{fact['created_at']}] "
            
            base_fact = str(fact.get("fact", "") or fact.get("data", "") or json.dumps(fact))
            line = f"{fact_prefix}{base_fact}"
            
            # Truncate oversized individual facts
            if len(line) > ZEP_CHAR_LIMIT:
                logger.warning("Single fact exceeds %d chars — truncating", ZEP_CHAR_LIMIT)
                line = line[:ZEP_CHAR_LIMIT]
            line_len = len(line) + 1

            if current_len + line_len > ZEP_CHAR_LIMIT:
                if current_lines:
                    chunks.append("\n".join(current_lines))
                current_lines = [line]
                current_len = line_len
            else:
                current_lines.append(line)
                current_len += line_len

        if current_lines:
            chunks.append("\n".join(current_lines))

        return chunks if chunks else [""]

    @property
    def facts_saved_count(self) -> int:
        return self._facts_saved
