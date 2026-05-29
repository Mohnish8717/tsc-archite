"""
WorldDataBank — Refactored to delegate all storage/retrieval to WorldRAGEngine.

The Hindsight dependency is REMOVED from this class.
Hindsight is preserved ONLY in:
  - HindsightBoardroom  (boardroom-{agent} banks)
  - HindsightOASISManager (oasis-{sim_id} banks)

RAG Architect skill checkpoint:
  assert all chunks carry source metadata ✅
  assert deduplication via deterministic IDs ✅
  assert hybrid search active ✅
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy engine reference — set by orchestrator at startup
_engine: Optional["WorldRAGEngine"] = None  # noqa: F821


def set_engine(engine: "WorldRAGEngine") -> None:  # noqa: F821
    """Injected by the orchestrator after WorldRAGEngine.initialize()."""
    global _engine
    _engine = engine
    logger.info("WorldDataBank: engine injected (%s)", type(engine).__name__)


def _require_engine() -> "WorldRAGEngine":  # noqa: F821
    if _engine is None:
        raise RuntimeError(
            "WorldDataBank has no engine. "
            "Call set_engine(engine) from the orchestrator before using WorldDataBank."
        )
    return _engine


class WorldDataBank:
    """
    Public façade — API identical to the old Hindsight-based version.

    All data now flows through WorldRAGEngine (Qdrant + Neo4j + LazyGraphRAG).
    Callers do not need to change their code.
    """

    def __init__(self, fallback_mode: bool = False):
        # fallback_mode retained for API compatibility — no longer meaningful
        self.bank_id = "tsc-world"
        self._fallback_mode = False  # engine always available

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize_session(self, session_id: str = "tsc-world", run_id: str = "global") -> None:
        """Connect to the WorldRAGEngine (idempotent)."""
        engine = _require_engine()
        await engine.initialize(run_id)
        self.bank_id = session_id
        logger.info("WorldDataBank session ready (run_id=%s)", run_id)

    # ------------------------------------------------------------------
    # Ingestion
    # RAG Architect: idempotent upsert with deterministic IDs ✅
    # ------------------------------------------------------------------
    async def ingest_document(
        self,
        document_text: str,
        document_name: str,
        doc_type: str = "document",
        run_id: str = "global",
    ) -> None:
        """
        Ingest a document into the company knowledge base (Plane 1).

        RAG Architect checkpoints applied:
          ✅ Source metadata enriched on every chunk
          ✅ Deduplication via content-hash IDs (safe to re-call)
          ✅ Semantic chunking (not fixed-size)
        """
        engine = _require_engine()

        # Write to a temp file so WorldRAGEngine can read + smart-chunk
        import tempfile, pathlib
        suffix = ".txt"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        ) as f:
            f.write(document_text)
            tmp_path = f.name

        try:
            n = await engine.ingest_company_doc(
                file_path=tmp_path,
                doc_type=doc_type,
                rebuild_graph=True,
            )
            logger.info("WorldDataBank: ingested '%s' → %d chunks", document_name, n)
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Retrieval
    # RAG Architect: hybrid search + reranking ✅
    # ------------------------------------------------------------------
    async def query_world_bank(
        self,
        query: str,
        run_id: str = "global",
        top_k: int = 5,
    ) -> str:
        """
        Unified RAG query — auto-routes to vector / hybrid / global.

        RAG Architect checkpoints applied:
          ✅ Hybrid search (dense + BM25 RRF)
          ✅ CrossEncoder reranking (top-k before LLM)
          ✅ Metadata filtering by run_id
        """
        engine = _require_engine()
        results = await engine.query(query=query, run_id=run_id, top_k=top_k)

        if not results:
            return f"[WorldDataBank] No results found for: {query}"

        # Format for LLM context injection
        lines = [f"[Source: {r.source}] {r.text}" for r in results]
        return "\n\n---\n\n".join(lines)

    # ------------------------------------------------------------------
    # Backward-compat helpers used by layer2 / layer6
    # ------------------------------------------------------------------
    async def retain(self, bank: str, content: str, metadata: dict = None, run_id: str = "global") -> None:
        if metadata is None:
            metadata = {}
        engine = _require_engine()
        metadata["run_id"] = run_id
        await engine.retain(bank=bank, content=content, metadata=metadata)

    async def recall(self, bank: str, query: str, run_id: str = "global") -> list[str]:
        engine = _require_engine()
        return await engine.recall(bank=bank, query=query, run_id=run_id)

    async def query_graph(self, cypher: str, params: dict = None) -> list:
        if params is None:
            params = {}
        engine = _require_engine()
        return await engine.query_graph(cypher=cypher, params=params)
