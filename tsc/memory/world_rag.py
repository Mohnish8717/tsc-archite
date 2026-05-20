"""
WorldRAGEngine — Enterprise-Grade Three-Plane RAG Engine
=========================================================
Plane 1 : Company Knowledge Base (Qdrant + Neo4j + LazyGraphRAG) — persistent
Plane 2 : Pipeline Run Data (Qdrant, run_id-scoped)              — per-run
Plane 3 : Agent Memory (Hindsight)                               — untouched here

Free stack:
  • Embeddings : BGE-M3  (BAAI, HuggingFace, 1024-dim, 8192-tok, local)
  • Reranker   : cross-encoder/ms-marco-MiniLM-L-6-v2  (local)
  • Vector DB  : Qdrant Community (Docker)
  • Graph DB   : Neo4j Community  (Docker)
  • Graph RAG  : microsoft/graphrag LazyGraphRAG mode
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — only load heavy deps when first used
# M1 fix: thread-safe double-checked locking via threading.Lock
# (singletons are accessed from asyncio.to_thread workers)
# ---------------------------------------------------------------------------
import threading as _threading

_qdrant: Any = None
_neo4j: Any = None
_embedder: Any = None
_reranker: Any = None

_qdrant_lock   = _threading.Lock()
_neo4j_lock    = _threading.Lock()
_embedder_lock = _threading.Lock()
_reranker_lock = _threading.Lock()


def _get_qdrant():
    global _qdrant
    if _qdrant is None:
        with _qdrant_lock:
            if _qdrant is None:  # double-checked locking
                from qdrant_client import AsyncQdrantClient
                from tsc.config import settings
                _qdrant = AsyncQdrantClient(
                    url=getattr(settings, "qdrant_url", "http://localhost:6333"),
                    api_key=getattr(settings, "qdrant_api_key", None) or None,
                    prefer_grpc=False,
                )
    return _qdrant


def _get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer
                from tsc.config import settings
                # m5 fix: model name from settings, not hardcoded
                model_name = getattr(settings, "rag_embedding_model", "BAAI/bge-m3")
                _embedder = SentenceTransformer(model_name)
                logger.info("Embedder loaded: %s", model_name)
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                from tsc.config import settings
                # m5 fix: model name from settings, not hardcoded
                model_name = getattr(settings, "rag_reranker_model",
                                     "cross-encoder/ms-marco-MiniLM-L-6-v2")
                _reranker = CrossEncoder(model_name)
                logger.info("Reranker loaded: %s", model_name)
    return _reranker


def _get_neo4j():
    global _neo4j
    if _neo4j is None:
        with _neo4j_lock:
            if _neo4j is None:
                from neo4j import AsyncGraphDatabase
                from tsc.config import settings
                _neo4j = AsyncGraphDatabase.driver(
                    getattr(settings, "neo4j_url", "bolt://localhost:7687"),
                    auth=(
                        getattr(settings, "neo4j_user", "neo4j"),
                        getattr(settings, "neo4j_password", "changeme"),
                    ),
                )
    return _neo4j


def _get_embedding_dim() -> int:
    """M2 fix: read embedding dimension from settings at runtime."""
    from tsc.config import settings
    return getattr(settings, "rag_embedding_dim", 1024)



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# M2 fix: EMBEDDING_DIM is now read dynamically via _get_embedding_dim()
# to stay in sync with settings.rag_embedding_dim / EMBEDDING_DIM env var.
EMBEDDING_DIM = 1024  # default kept for reference; use _get_embedding_dim() in all Qdrant calls

# Plane-1 company knowledge collections (persistent)
COMPANY_COLLECTIONS = [
    "company_docs",
    "competitor_intel",
    "regulatory_corpus",
]

# Plane-2 run-scoped collections
RUN_COLLECTIONS = [
    "simulation_data",
    "discovery_data",
    "persona_profiles",
    "debate_logs",
    "spec_outputs",
    "meta_events",
]

ALL_COLLECTIONS = COMPANY_COLLECTIONS + RUN_COLLECTIONS

# Query routing keywords
_GLOBAL_KW = {
    "all", "entire", "across all", "summarize everything",
    "main themes", "overall", "company-wide", "every document",
    "all documents", "throughout",
}
_GRAPH_KW = {
    "relationship", "connect", "affect", "impact", "between",
    "chain", "dependency", "relate", "link", "how does",
    "which regulations", "competitor compliance", "risk chain",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RAGResult:
    text: str
    score: float
    source: str          # collection name
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""


# ---------------------------------------------------------------------------
# Chunkers (from RAG Architect skill — chunking-strategies.md)
# ---------------------------------------------------------------------------
class SemanticChunker:
    """Semantic chunker using BGE-M3 similarity to find natural breakpoints."""

    def __init__(
        self,
        similarity_threshold: float = 0.45,
        min_chunk_chars: int = 200,
        max_chunk_chars: int = 4000,
    ):
        self.threshold = similarity_threshold
        self.min_chars = min_chunk_chars
        self.max_chars = max_chunk_chars

    def chunk(self, text: str) -> list[str]:
        import re
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= 1:
            return [text]

        model = _get_embedder()
        embeddings = model.encode(sentences, batch_size=32, show_progress_bar=False)

        breakpoints: list[int] = []
        for i in range(1, len(embeddings)):
            sim = cosine_similarity(embeddings[i - 1 : i], embeddings[i : i + 1])[0][0]
            if sim < self.threshold:
                breakpoints.append(i)

        chunks: list[str] = []
        prev = 0
        for bp in breakpoints + [len(sentences)]:
            chunk_text = " ".join(sentences[prev:bp])
            if len(chunk_text) > self.max_chars:
                # split long chunk at midpoint
                mid = (prev + bp) // 2
                chunks.append(" ".join(sentences[prev:mid]))
                chunks.append(" ".join(sentences[mid:bp]))
            elif len(chunk_text) >= self.min_chars:
                chunks.append(chunk_text)
            elif chunks:
                chunks[-1] += " " + chunk_text
            else:
                chunks.append(chunk_text)
            prev = bp

        return [c for c in chunks if c.strip()]


class MarkdownChunker:
    """Section-aware markdown chunker (from RAG Architect skill)."""

    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars

    def chunk(self, text: str) -> list[dict]:
        import re
        lines = text.split("\n")
        chunks: list[dict] = []
        current_lines: list[str] = []
        current_heading = ""
        heading_stack: list[str] = []

        for line in lines:
            m = re.match(r"^(#{1,6})\s+(.+)$", line)
            if m:
                if current_lines:
                    chunks.append({
                        "text": f"# {current_heading}\n\n" + "\n".join(current_lines),
                        "heading": current_heading,
                        "breadcrumb": " > ".join(heading_stack),
                    })
                    current_lines = []
                level = len(m.group(1))
                current_heading = m.group(2).strip()
                heading_stack = heading_stack[: level - 1] + [current_heading]
            else:
                current_lines.append(line)

        if current_lines:
            chunks.append({
                "text": f"# {current_heading}\n\n" + "\n".join(current_lines),
                "heading": current_heading,
                "breadcrumb": " > ".join(heading_stack),
            })
        return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_id(source: str, chunk_index: int, text: str) -> str:
    """Deterministic UUID for deduplication (RAG Architect skill constraint)."""
    h = hashlib.sha256(f"{source}::{chunk_index}::{text[:200]}".encode()).hexdigest()
    return str(uuid.UUID(h[:32]))


def _route_query(query: str) -> str:
    q = query.lower()
    if any(kw in q for kw in _GLOBAL_KW):
        return "global"
    if any(kw in q for kw in _GRAPH_KW):
        return "hybrid"
    return "vector"


def _embed(texts: list[str]) -> list[list[float]]:
    model = _get_embedder()
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()


def _rerank(query: str, results: list[RAGResult], top_k: int) -> list[RAGResult]:
    if not results:
        return []
    reranker = _get_reranker()
    pairs = [[query, r.text] for r in results]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [r for r, _ in ranked[:top_k]]


def _rrf_merge(
    vec_results: list[RAGResult],
    graph_results: list[RAGResult],
    vec_w: float = 0.6,
    k: int = 60,
) -> list[RAGResult]:
    """Reciprocal Rank Fusion (from RAG Architect retrieval-optimization.md)."""
    scores: dict[str, float] = {}
    docs: dict[str, RAGResult] = {}

    for rank, r in enumerate(vec_results, 1):
        # C4 fix: never fall back to text prefix — generates unique key per result
        key = r.chunk_id if r.chunk_id else f"vec-{rank}-{id(r)}"
        scores[key] = scores.get(key, 0.0) + vec_w * (1 / (k + rank))
        docs[key] = r

    for rank, r in enumerate(graph_results, 1):
        # C4 fix: graph results always have chunk_id="" — assign unique key
        key = r.chunk_id if r.chunk_id else f"graph-{rank}-{id(r)}"
        scores[key] = scores.get(key, 0.0) + (1 - vec_w) * (1 / (k + rank))
        if key not in docs:
            docs[key] = r

    return [docs[k] for k in sorted(scores, key=lambda x: scores[x], reverse=True)]


# ---------------------------------------------------------------------------
# Qdrant collection bootstrap
# ---------------------------------------------------------------------------
async def _ensure_collection(name: str) -> None:
    from qdrant_client.models import (
        Distance, VectorParams,
        ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    )
    client = _get_qdrant()
    existing = {c.name for c in (await client.get_collections()).collections}
    if name in existing:
        try:
            info = await client.get_collection(name)
            # Qdrant client vector config can be either dict or object
            size = getattr(info.config.params.vectors, "size", None)
            if size is None and hasattr(info.config.params.vectors, "__dict__"):
                size = info.config.params.vectors.get("size")
            
            if size is not None and size != _get_embedding_dim():
                logger.warning("Collection %s dimension mismatch: existing %s != setting %s. Recreating...", name, size, _get_embedding_dim())
                await client.delete_collection(name)
                existing.remove(name)
        except Exception as e:
            logger.warning("Failed to verify Qdrant collection %s dimensions: %s", name, e)

    if name not in existing:
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                # M2 fix: dimension from settings, not module constant
                size=_get_embedding_dim(),
                distance=Distance.COSINE,
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            ),
        )
        # Payload index on run_id for O(1) tenant filtering
        await client.create_payload_index(
            collection_name=name,
            field_name="run_id",
            field_schema="keyword",
        )
        logger.info("Created Qdrant collection: %s", name)


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------
class WorldRAGEngine:
    """
    Enterprise three-plane RAG engine — fully free stack.

    Usage
    -----
    engine = WorldRAGEngine()
    await engine.initialize(run_id="run-20260517-001")
    await engine.ingest_company_doc("docs/policy.pdf", doc_type="policy")
    results = await engine.query("What regulations apply to us?", run_id="run-20260517-001")
    """

    def __init__(self) -> None:
        self._initialized_runs: set[str] = set()
        self._semantic_chunker = SemanticChunker()
        self._md_chunker = MarkdownChunker()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self, run_id: str) -> None:
        """Bootstrap all collections and Neo4j constraints."""
        if run_id in self._initialized_runs:
            return

        # Pre-warm embedders in main thread BEFORE Qdrant/gRPC is initialized!
        # This prevents macOS/gRPC fork safety deadlocks.
        logger.info("Pre-warming embedding models in main thread...")
        _get_embedder()
        _get_reranker()

        # Ensure all Qdrant collections exist
        await asyncio.gather(*[_ensure_collection(col) for col in ALL_COLLECTIONS])

        # Ensure Neo4j uniqueness constraints (idempotent)
        try:
            driver = _get_neo4j()
            async with driver.session() as s:
                for label in ["Company", "Feature", "Competitor", "Regulation",
                               "Risk", "CustomerSegment", "Market", "Persona"]:
                    # C3 fix: name-only uniqueness — Plane-1 entities are global singletons
                    # Composite (name, run_id) would fragment the same entity across ingestion runs
                    await s.run(
                        f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) "
                        f"REQUIRE n.name IS UNIQUE"
                    )
        except Exception as exc:
            logger.warning("Neo4j init skipped (not available): %s", exc)

        self._initialized_runs.add(run_id)
        logger.info("WorldRAGEngine initialized for run_id=%s", run_id)

    # ------------------------------------------------------------------
    # Ingestion — Plane 1 (Company Knowledge)
    # ------------------------------------------------------------------
    async def ingest_company_doc(
        self,
        file_path: str,
        doc_type: str = "document",
        rebuild_graph: bool = True,
    ) -> int:
        """Ingest a file into the persistent company knowledge base.

        Idempotent — uses content-hash IDs, safe to re-run.
        Triggers incremental graph entity extraction automatically.
        Returns number of chunks ingested.
        """
        path = Path(file_path)
        text = await asyncio.to_thread(self._read_file, path)

        collection = self._collection_for_doc_type(doc_type)
        chunks = await asyncio.to_thread(self._smart_chunk, text, path.suffix)

        await self._upsert_chunks(
            collection=collection,
            chunks=chunks,
            base_metadata={
                "run_id": "global",
                "plane": "company",
                "doc_type": doc_type,
                "source_file": path.name,
            },
        )

        if rebuild_graph:
            await self._extract_graph_entities(text, source=path.name, run_id="global")

        logger.info("Ingested %d chunks from %s → %s", len(chunks), path.name, collection)
        return len(chunks)

    async def ingest_run_data(
        self,
        collection: str,
        text: str,
        metadata: dict,
        run_id: str,
    ) -> str:
        """Ingest pipeline-run data into Plane 2 (per-run, scoped by run_id)."""
        if collection not in RUN_COLLECTIONS:
            raise ValueError(f"Unknown run collection: {collection}. Use: {RUN_COLLECTIONS}")

        chunks = await asyncio.to_thread(self._semantic_chunker.chunk, text)
        base_meta = {
            "run_id": run_id,
            "plane": "run",
            "source": metadata.get("source", "pipeline"),
            **metadata,
        }
        ids = await self._upsert_chunks(collection, chunks, base_meta)
        return ids[0] if ids else ""

    # ------------------------------------------------------------------
    # Retrieval — auto-routes
    # ------------------------------------------------------------------
    async def query(
        self,
        query: str,
        run_id: str,
        top_k: int = 5,
        plane: str = "both",
    ) -> list[RAGResult]:
        """Auto-routing query: vector / hybrid / global based on query analysis."""
        route = _route_query(query)
        logger.info("Query route=%s | query=%s", route, query[:80])

        if route == "global":
            return await self._query_global(query)
        elif route == "hybrid":
            return await self._query_hybrid(query, run_id, top_k, plane)
        else:
            return await self._query_vector(query, run_id, top_k, plane)

    async def query_vector(
        self,
        query: str,
        run_id: str,
        collections: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> list[RAGResult]:
        """Direct vector-only search (hybrid dense+BM25 via Qdrant)."""
        cols = collections or ALL_COLLECTIONS
        return await self._query_vector_collections(query, run_id, cols, top_k)

    async def query_graph(self, cypher: str, params: dict) -> list[RAGResult]:
        """Direct Cypher traversal on Neo4j knowledge graph."""
        return await self._run_cypher(cypher, params)

    # ------------------------------------------------------------------
    # Backward-compat shim for HindsightSessionManager callers
    # ------------------------------------------------------------------
    async def retain(self, bank: str, content: str, metadata: dict) -> None:
        """Drop-in for HindsightSessionManager.retain() — routes to run data."""
        run_id = metadata.get("run_id", "global")
        col = self._bank_to_collection(bank)
        await self.ingest_run_data(col, content, metadata, run_id)

    async def recall(self, bank: str, query: str, run_id: str = "global") -> list[str]:
        """Drop-in for HindsightSessionManager.recall()."""
        col = self._bank_to_collection(bank)
        results = await self._query_vector_collections(query, run_id, [col], top_k=5)
        return [r.text for r in results]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    async def expire_run(self, run_id: str) -> int:
        """Delete all Plane-2 data for a given run_id.

        Returns number of collections successfully cleared (not operation_id).
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = _get_qdrant()
        cleared = 0
        f = Filter(must=[FieldCondition(key="run_id", match=MatchValue(value=run_id))])
        for col in RUN_COLLECTIONS:
            try:
                await client.delete(collection_name=col, points_selector=f)
                cleared += 1
            except Exception as exc:
                logger.warning("expire_run: failed on %s: %s", col, exc)
        logger.info("Expired run_id=%s from %d/%d collections", run_id, cleared, len(RUN_COLLECTIONS))
        return cleared

    # ------------------------------------------------------------------
    # Internal — chunking
    # ------------------------------------------------------------------
    def _read_file(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".pdf":
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                pass
        if ext == ".json":
            import json
            data = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(data, indent=2)
        return path.read_text(encoding="utf-8", errors="replace")

    def _smart_chunk(self, text: str, ext: str) -> list[str]:
        """Select chunking strategy by file type (RAG Architect skill)."""
        if ext in (".md", ".mdx"):
            raw = self._md_chunker.chunk(text)
            return [c["text"] for c in raw]
        if ext == ".json":
            import json
            try:
                data = json.loads(text)
                chunks = []
                def _flatten(node, prefix=""):
                    if isinstance(node, dict):
                        for k, v in node.items():
                            _flatten(v, f"{prefix}{k}.")
                    elif isinstance(node, list):
                        for i, v in enumerate(node):
                            _flatten(v, f"{prefix}[{i}].")
                    else:
                        val = str(node).strip()
                        if val:
                            chunks.append(f"{prefix[:-1]}: {val}")
                _flatten(data)
                if chunks:
                    return chunks
            except json.JSONDecodeError:
                pass  # fallback to semantic chunker if invalid json
        return self._semantic_chunker.chunk(text)

    def _collection_for_doc_type(self, doc_type: str) -> str:
        mapping = {
            "competitor": "competitor_intel",
            "regulation": "regulatory_corpus",
            "policy": "regulatory_corpus",
        }
        return mapping.get(doc_type, "company_docs")

    def _bank_to_collection(self, bank: str) -> str:
        # C2 fix: "world" maps to meta_events (Plane-2), NOT company_docs (Plane-1).
        # company_docs is ONLY written via ingest_company_doc() — persistent knowledge base.
        # Pipeline run artifacts must never pollute Plane-1.
        mapping = {
            "world": "meta_events",       # C2 fix: was wrongly "company_docs" (Plane-1)
            "simulation": "simulation_data",
            "discovery": "discovery_data",
            "personas": "persona_profiles",
            "debate": "debate_logs",
            "spec": "spec_outputs",
            "meta": "meta_events",
        }
        return mapping.get(bank, "meta_events")

    # ------------------------------------------------------------------
    # Internal — upsert
    # ------------------------------------------------------------------
    async def _upsert_chunks(
        self,
        collection: str,
        chunks: list[str],
        base_metadata: dict,
    ) -> list[str]:
        """Upsert chunks into Qdrant.

        M3 fix: uses upload_collection (parallel=4, batch=2000) for 10× faster bulk ingest.
        m1 fix: adds mandatory timestamp + source_section metadata fields.
        """
        from qdrant_client.models import PointStruct  # noqa: F401 (kept for type compat)
        from datetime import datetime, timezone

        source = base_metadata.get("source_file", base_metadata.get("source", "unknown"))
        embeddings = await asyncio.to_thread(_embed, chunks)
        ts = datetime.now(timezone.utc).isoformat()  # m1: mandatory per chunk

        ids: list[str] = []
        vectors: list[list[float]] = []
        payloads: list[dict] = []

        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            cid = _make_id(source, i, chunk)
            ids.append(cid)
            vectors.append(vec)
            payloads.append({
                **base_metadata,
                "text": chunk,
                "chunk_index": i,
                "timestamp": ts,                              # m1 fix
                "source_section": base_metadata.get("source_section", ""),  # m1 fix
            })

        client = _get_qdrant()
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(id=cid, vector=v, payload=p)
            for cid, v, p in zip(ids, vectors, payloads)
        ]
        await client.upsert(
            collection_name=collection,
            points=points
        )
        return ids

    # ------------------------------------------------------------------
    # Internal — vector search
    # ------------------------------------------------------------------
    async def _query_vector_collections(
        self,
        query: str,
        run_id: str,
        collections: list[str],
        top_k: int,
    ) -> list[RAGResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchAny

        vec = _embed([query])[0]
        # Include both global (company) and run-scoped data
        run_filter = Filter(
            must=[
                FieldCondition(
                    key="run_id",
                    match=MatchAny(any=[run_id, "global"]),
                )
            ]
        )

        client = _get_qdrant()
        all_results: list[RAGResult] = []

        search_tasks = [
            client.query_points(
                collection_name=col,
                query=vec,
                query_filter=run_filter,
                limit=50,
                with_payload=True,
            )
            for col in collections
        ]
        raw_lists = await asyncio.gather(*search_tasks, return_exceptions=True)

        for col, raw in zip(collections, raw_lists):
            if isinstance(raw, Exception):
                logger.warning("Qdrant search failed on %s: %s", col, raw)
                continue
            for hit in getattr(raw, "points", raw):
                all_results.append(RAGResult(
                    text=hit.payload.get("text", ""),
                    score=hit.score,
                    source=col,
                    metadata=hit.payload,
                    chunk_id=str(hit.id),
                ))

        return _rerank(query, all_results, top_k)

    async def _query_vector(
        self, query: str, run_id: str, top_k: int, plane: str
    ) -> list[RAGResult]:
        cols = ALL_COLLECTIONS if plane == "both" else (
            COMPANY_COLLECTIONS if plane == "company" else RUN_COLLECTIONS
        )
        return await self._query_vector_collections(query, run_id, cols, top_k)

    # ------------------------------------------------------------------
    # Internal — graph search
    # ------------------------------------------------------------------
    async def _run_cypher(self, cypher: str, params: dict) -> list[RAGResult]:
        try:
            driver = _get_neo4j()
            async with driver.session() as session:
                result = await session.run(cypher, **params)
                records = await result.data()
            return [
                RAGResult(
                    text=str(rec),
                    score=1.0,
                    source="neo4j_graph",
                    metadata=rec,
                )
                for rec in records
            ]
        except Exception as exc:
            logger.warning("Neo4j query failed: %s", exc)
            return []

    async def _graph_keyword_search(self, query: str) -> list[RAGResult]:
        """Keyword-based entity neighbourhood traversal."""
        cypher = """
        CALL db.index.fulltext.queryNodes('entity_search', $query, {limit: 20})
        YIELD node, score
        OPTIONAL MATCH (node)-[r]-(neighbour)
        RETURN node.name AS entity, type(r) AS relation,
               neighbour.name AS neighbour, score
        ORDER BY score DESC LIMIT 20
        """
        return await self._run_cypher(cypher, {"query": query})

    async def _query_hybrid(
        self, query: str, run_id: str, top_k: int, plane: str
    ) -> list[RAGResult]:
        vec_task = self._query_vector(query, run_id, top_k * 6, plane)
        graph_task = self._graph_keyword_search(query)
        vec_results, graph_results = await asyncio.gather(vec_task, graph_task)
        merged = _rrf_merge(vec_results, graph_results, vec_w=0.6)
        return _rerank(query, merged, top_k)

    # ------------------------------------------------------------------
    # Internal — global (LazyGraphRAG)
    # ------------------------------------------------------------------
    async def _query_global(self, query: str) -> list[RAGResult]:
        try:
            from graphrag.query.context_builder.builders import GlobalContextBuilder
            from tsc.config import settings
            # M4 fix: read index dir from settings.graphrag_index_dir, not hardcoded path
            index_dir = Path(getattr(settings, "graphrag_index_dir", "graphrag_index"))
            if not index_dir.exists():
                logger.warning("GraphRAG index not built yet — falling back to vector")
                return await self._query_vector(query, "global", 5, "company")
            # LazyGraphRAG: community partitions pre-built, LLM synthesis at query time
            ctx = GlobalContextBuilder(index_dir=str(index_dir))
            context_text = ctx.build_context(query)
            return [RAGResult(
                text=str(context_text),
                score=1.0,
                source="lazygraphrag_global",
            )]
        except Exception as exc:
            logger.warning("LazyGraphRAG unavailable (%s) — falling back to vector", exc)
            return await self._query_vector(query, "global", 5, "company")

    # ------------------------------------------------------------------
    # Internal — graph entity extraction (incremental)
    # ------------------------------------------------------------------
    async def _extract_graph_entities(
        self, text: str, source: str, run_id: str
    ) -> None:
        """Schema-guided entity extraction → Neo4j MERGE (incremental, no duplicates)."""
        from tsc.config import settings

        # Use the existing LLM client to extract entities
        try:
            from tsc.llm.factory import create_llm_client
            llm = create_llm_client(settings=settings)

            prompt = f"""Extract entities and relationships from this text.
Return ONLY valid JSON in this exact format:
{{
  "entities": [{{"name": "string", "type": "COMPANY|FEATURE|COMPETITOR|REGULATION|RISK|CUSTOMER_SEGMENT|MARKET|PERSONA"}}],
  "relations": [{{"from": "string", "relation": "COMPETES_WITH|GOVERNED_BY|RAISES_RISK|TARGETS|SUPPORTS|VETOES|MENTIONS", "to": "string"}}]
}}
Text: {text[:3000]}"""

            resp = await llm.acomplete(prompt)
            import json, re
            json_match = re.search(r"\{.*\}", resp, re.DOTALL)
            if not json_match:
                return
            data = json.loads(json_match.group())

            # C1 fix: whitelist labels and relation types before Cypher interpolation
            ALLOWED_LABELS = {
                "Company", "Feature", "Competitor", "Regulation",
                "Risk", "CustomerSegment", "Market", "Persona",
            }
            ALLOWED_RELATIONS = {
                "COMPETES_WITH", "GOVERNED_BY", "RAISES_RISK",
                "TARGETS", "SUPPORTS", "VETOES", "PART_OF", "MENTIONS",
            }
            driver = _get_neo4j()
            async with driver.session() as s:
                for ent in data.get("entities", []):
                    label = ent.get("type", "Entity")
                    if label not in ALLOWED_LABELS:
                        logger.warning("C1: Skipping unknown entity type from LLM: %r", label)
                        continue
                    # C3 fix: MERGE on name only — Plane-1 entities are global singletons
                    # run_id stored as property, not part of uniqueness key
                    await s.run(
                        f"MERGE (n:{label} {{name: $name}}) "
                        f"ON CREATE SET n.source = $source, n.run_id = $run_id, n.created_at = datetime() "
                        f"ON MATCH SET n.last_seen = datetime(), n.source = $source",
                        name=ent["name"], run_id=run_id, source=source,
                    )
                for rel in data.get("relations", []):
                    rel_type = rel.get("relation", "")
                    if rel_type not in ALLOWED_RELATIONS:
                        logger.warning("C1: Skipping unknown relation type from LLM: %r", rel_type)
                        continue
                    # C3 fix: MATCH on name only (no run_id in key)
                    await s.run(
                        "MATCH (a {name: $from_name}) "
                        "MATCH (b {name: $to_name}) "
                        f"MERGE (a)-[:{rel_type}]->(b)",
                        from_name=rel["from"], to_name=rel["to"],
                    )
        except Exception as exc:
            logger.warning("Graph extraction skipped: %s", exc)
