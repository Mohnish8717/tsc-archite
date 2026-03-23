"""Hybrid fact retrieval from Zep memory."""

from __future__ import annotations

import logging
from typing import Any

from tsc.memory.zep_client import ZepMemoryClient

logger = logging.getLogger(__name__)


class FactRetriever:
    """High-level fact retrieval with sub-query expansion and deduplication.

    Provides:
    - Semantic search (vector-based)
    - Keyword-based search
    - Sub-query decomposition for complex questions
    - Result deduplication and ranking
    """

    def __init__(self, zep_client: ZepMemoryClient):
        self._zep = zep_client

    async def search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Simple semantic search wrapper."""
        return await self._zep.search_facts(query, limit=limit)

    async def multi_query_search(
        self,
        queries: list[str],
        limit_per_query: int = 10,
    ) -> list[dict[str, Any]]:
        """Run multiple queries and deduplicate results.

        Useful for complex questions decomposed into sub-queries.
        """
        all_results: dict[str, dict[str, Any]] = {}

        for query in queries:
            results = await self._zep.search_facts(query, limit=limit_per_query)
            for r in results:
                fact = r.get("fact", "")
                if fact and fact not in all_results:
                    all_results[fact] = r

        # Sort by score descending
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get("score", 0),
            reverse=True,
        )
        return sorted_results

    async def retrieve_for_gate(
        self,
        gate_name: str,
        feature_name: str,
        additional_queries: list[str] | None = None,
    ) -> list[str]:
        """Retrieve relevant facts for a specific gate evaluation."""
        queries = [
            f"{feature_name} technical feasibility",
            f"{feature_name} user impact",
            f"{feature_name} risks and constraints",
        ]
        if additional_queries:
            queries.extend(additional_queries)

        results = await self.multi_query_search(queries, limit_per_query=8)
        return [r["fact"] for r in results if r.get("fact")]
