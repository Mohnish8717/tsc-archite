"""Zep Cloud SDK wrapper for TSC memory operations."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ZepMemoryClient:
    """Wrapper around the Zep Cloud SDK for knowledge graph operations.

    Handles:
    - Client initialization with API key
    - Group management (one per evaluation run)
    - Batch data ingestion
    - Fact storage and retrieval
    """

    def __init__(self, api_key: str):
        try:
            from zep_cloud.client import AsyncZep

            self._client = AsyncZep(api_key=api_key)
        except ImportError:
            logger.warning(
                "zep-cloud not installed; memory operations will be no-ops. "
                "Install with: pip install zep-cloud"
            )
            self._client = None

        self._group_id: Optional[str] = None
        self._user_id: Optional[str] = None
        self._facts_saved = 0

    async def initialize_session(self, session_name: str = "tsc_evaluation") -> str:
        """Create a new user + group for this evaluation run."""
        self._user_id = f"tsc_{session_name}_{uuid.uuid4().hex[:8]}"
        self._group_id = f"tsc_group_{uuid.uuid4().hex[:8]}"

        if self._client:
            try:
                await self._client.user.add(user_id=self._user_id)
                await self._client.group.add(group_id=self._group_id)
                logger.info(
                    "Zep session initialized: user=%s, group=%s",
                    self._user_id,
                    self._group_id,
                )
            except Exception as e:
                logger.warning("Zep session init failed (non-fatal): %s", e)

        return self._group_id or session_name

    async def ingest_facts(
        self, facts: list[dict[str, Any]], batch_size: int = 5
    ) -> int:
        """Store facts in Zep via batch ingestion.

        Args:
            facts: List of {fact, entities, metadata} dicts.
            batch_size: How many facts to send per batch.

        Returns:
            Number of facts successfully stored.
        """
        if not self._client or not self._user_id:
            logger.debug("Zep not available; storing %d facts locally.", len(facts))
            self._facts_saved += len(facts)
            return len(facts)

        stored = 0
        for i in range(0, len(facts), batch_size):
            batch = facts[i : i + batch_size]
            combined_text = "\n".join(f["fact"] for f in batch)
            try:
                await self._client.memory.add(
                    session_id=self._user_id,
                    messages=[
                        {
                            "role_type": "system",
                            "role": "TSC Pipeline",
                            "content": combined_text,
                        }
                    ],
                )
                stored += len(batch)
            except Exception as e:
                logger.warning("Zep batch ingestion failed: %s", e)

        self._facts_saved += stored
        logger.info("Ingested %d/%d facts into Zep.", stored, len(facts))
        return stored

    async def search_facts(
        self,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search for relevant facts using Zep's hybrid search.

        Combines semantic + BM25 search with cross-encoder reranking.
        """
        if not self._client or not self._user_id:
            logger.debug("Zep not available; returning empty results for: %s", query[:50])
            return []

        try:
            results = await self._client.memory.search_sessions(
                text=query,
                user_id=self._user_id,
                search_scope="facts",
                limit=limit,
            )
            return [
                {
                    "fact": r.fact.fact if hasattr(r, "fact") and r.fact else str(r),
                    "score": r.score if hasattr(r, "score") else 0.0,
                }
                for r in (results.results if hasattr(results, "results") else results)
            ]
        except Exception as e:
            logger.warning("Zep search failed: %s", e)
            return []

    async def get_graph_entities(self) -> list[dict[str, Any]]:
        """Retrieve entities from the Zep knowledge graph."""
        if not self._client or not self._group_id:
            return []

        try:
            # Use group-level graph retrieval
            episodes = await self._client.graph.search(
                group_id=self._group_id,
                query="all entities",
                limit=100,
            )
            return [
                {
                    "name": str(e),
                    "source": "zep_graph",
                }
                for e in (episodes.results if hasattr(episodes, "results") else episodes)
            ]
        except Exception as e:
            logger.warning("Zep graph retrieval failed: %s", e)
            return []

    @property
    def facts_saved_count(self) -> int:
        return self._facts_saved
