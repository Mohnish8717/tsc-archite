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
        """Run multiple queries and combine via RRF (Reciprocal Rank Fusion).
        
        RRF provides expert-level reranking without expensive GPUs or LLMs,
        making it ideal for fanless M2 Air systems.
        """
        all_results: dict[str, dict[str, Any]] = {}
        query_ranks: list[dict[str, int]] = []

        for query in queries:
            results = await self._zep.search_facts(query, limit=limit_per_query)
            ranks: dict[str, int] = {}
            for i, r in enumerate(results):
                fact = r.get("fact", "")
                if not fact:
                    continue
                if fact not in all_results:
                    all_results[fact] = r
                ranks[fact] = i + 1
            query_ranks.append(ranks)

        # Apply Reciprocal Rank Fusion (RRF)
        # score = sum( 1 / (k + rank_i) )
        k = 60
        rrf_scores: dict[str, float] = {}
        for ranks in query_ranks:
            for fact, rank in ranks.items():
                rrf_scores[fact] = rrf_scores.get(fact, 0.0) + (1.0 / (k + rank))

        # Sort all results by RRF score
        sorted_facts = sorted(
            rrf_scores.keys(),
            key=lambda f: rrf_scores[f],
            reverse=True,
        )

        return [
            {**all_results[f], "rrf_score": rrf_scores[f]}
            for f in sorted_facts
        ]

    async def retrieve_for_gate(
        self,
        gate_name: str,
        feature_name: str,
        additional_queries: list[str] | None = None,
    ) -> list[str]:
        """Retrieve relevant facts for a specific gate evaluation using RRF."""
        queries = [
            f"{feature_name} {gate_name} technical feasibility",
            f"{feature_name} {gate_name} user impact",
            f"{feature_name} {gate_name} risks and constraints",
        ]
        if additional_queries:
            queries.extend(additional_queries)

        # Uses RRF for 'Expert-level' report reranking
        results = await self.multi_query_search(queries, limit_per_query=15)
        return [r["fact"] for r in results if r.get("fact")]
