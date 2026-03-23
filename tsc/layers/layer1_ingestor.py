"""Layer 1: Contextual Ingestor.

Ingests raw input documents, normalizes, chunks semantically,
enriches with NLP metadata, and creates a unified ProblemContextBundle.

Critical fixes applied:
  1. Input validation layer (3-5 docs, required types)
  2. Chunk deduplication (cosine similarity threshold 0.9)
  3. Enrichment quality gates (coverage and confidence checks)
  4. Better error logging for embeddings

Optimizations applied:
  1. Lazy-loaded NLP models with timing
  2. Batch embedding encoding (batch_size=32)
  3. Parallel file loading (ThreadPoolExecutor for 3+ docs)
  4. Embedding cache by sentence hash
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from tsc.llm.base import LLMClient
from tsc.llm.prompts import ENRICHMENT_SYSTEM, ENRICHMENT_USER
from tsc.models.chunks import (
    ChunkEntity,
    EnrichedChunk,
    ExtractedMetric,
    GlobalStatistics,
    ProblemContextBundle,
    SentimentLabel,
    SentimentResult,
    SourceSummary,
    TopicCategory,
)
from tsc.models.inputs import (
    CompanyContext,
    DocumentType,
    FeatureProposal,
    FileType,
    InputDocument,
    LoadedDocument,
    NormalizedContent,
)

logger = logging.getLogger(__name__)


# ── Custom Exceptions ────────────────────────────────────────────────


class ValidationError(Exception):
    """Raised when input validation fails."""


# ─────────────────────────────────────────────────────────────────────


class ContextualIngestor:
    """Layer 1: Load, normalize, chunk, enrich, and bundle input documents."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._nlp = None  # Will be loaded on first use
        self._embedder = None  # Will be loaded on first use
        self._embedding_cache: dict[int, np.ndarray] = {}
        logger.info("ContextualIngestor initialized (models lazy-loaded)")

    # ── Public API ───────────────────────────────────────────────────

    async def process(
        self, documents: list[InputDocument]
    ) -> tuple[ProblemContextBundle, FeatureProposal, CompanyContext]:
        """Execute the full Layer 1 pipeline.

        Returns:
            (ProblemContextBundle, FeatureProposal, CompanyContext)
        """
        t0 = time.time()

        # Step 0: VALIDATE (CRITICAL FIX #1)
        self._validate_inputs(documents)
        logger.info("✓ Validated %d documents", len(documents))

        # Step 1.1: Load files (OPT-3: parallel for 3+ docs)
        if len(documents) >= 3:
            with ThreadPoolExecutor(max_workers=3) as executor:
                loaded = list(executor.map(self._load_file, documents))
            logger.info("✓ Loaded %d files (in parallel)", len(loaded))
        else:
            loaded = [self._load_file(doc) for doc in documents]
            logger.info("✓ Loaded %d files", len(loaded))

        # Step 1.2: Normalize
        normalized = [self._normalize(doc) for doc in loaded]
        logger.info("✓ Normalized %d documents", len(normalized))

        # Extract structured data
        feature = self._extract_feature_proposal(normalized)
        company = self._extract_company_context(normalized)
        persona_ctx = self._extract_persona_context(normalized)
        market_ctx = self._extract_market_context(normalized)
        logger.info(
            "✓ Extracted feature: %s, company: %s",
            feature.title,
            company.company_name,
        )

        # Step 1.3: Semantic chunking
        chunks = self._semantic_chunk(normalized)
        logger.info("✓ Created %d chunks (before dedup)", len(chunks))

        # CRITICAL FIX #2: Deduplicate
        chunks = self._deduplicate_chunks(chunks)
        logger.info("✓ Deduplicated to %d chunks", len(chunks))

        # Step 1.4: NLP Enrichment
        enriched = await self._enrich_chunks(chunks)
        logger.info("✓ Enriched %d chunks", len(enriched))

        # CRITICAL FIX #3: Validate quality
        quality = self._validate_enrichment_quality(enriched)
        logger.info(
            "✓ Quality check: %.1f%% with entities, %.1f%% with metrics, "
            "avg confidence %.3f",
            quality.get("pct_chunks_with_entities", 0),
            quality.get("pct_chunks_with_metrics", 0),
            quality.get("avg_entity_confidence", 0),
        )

        # Step 1.5: Bundle
        bundle = self._create_bundle(enriched, time.time() - t0, persona_ctx, market_ctx)
        logger.info(
            "✓ Layer 1 complete: %d chunks, %d entities, %.1fs",
            bundle.statistics.total_chunks,
            bundle.statistics.unique_entities,
            time.time() - t0,
        )

        return bundle, feature, company

    # ── CRITICAL FIX #1: Input Validation ────────────────────────────

    def _validate_inputs(self, documents: list[InputDocument]) -> None:
        """Validate input documents before processing.

        Raises:
            ValidationError: If any validation check fails.
        """
        if not documents:
            raise ValidationError("No documents provided")

        if len(documents) < 3:
            raise ValidationError(
                f"Need 3+ documents, got {len(documents)}"
            )

        types_present = {d.type for d in documents}

        if DocumentType.FEATURE_PROPOSAL not in types_present:
            raise ValidationError("Missing FEATURE_PROPOSAL document")

        if DocumentType.COMPANY_CONTEXT not in types_present:
            raise ValidationError("Missing COMPANY_CONTEXT document")

        content_types = {
            DocumentType.INTERVIEWS,
            DocumentType.SUPPORT_TICKETS,
            DocumentType.ANALYTICS,
        }
        if not any(t in types_present for t in content_types):
            raise ValidationError(
                "Need at least one content document "
                "(interviews, support_tickets, or analytics)"
            )

        for doc in documents:
            path = Path(doc.file_path)
            if not path.exists():
                raise ValidationError(f"File not found: {doc.file_path}")

    # ── CRITICAL FIX #2: Chunk Deduplication ─────────────────────────

    def _deduplicate_chunks(
        self,
        chunks: list[EnrichedChunk],
        similarity_threshold: float = 0.9,
    ) -> list[EnrichedChunk]:
        """Remove near-duplicate chunks by embedding cosine similarity.

        Args:
            chunks: List of chunks (may contain embeddings).
            similarity_threshold: Chunks above this similarity to the
                first occurrence are discarded.

        Returns:
            De-duplicated chunk list preserving original order.
        """
        if len(chunks) < 2:
            return chunks

        # Skip if no embeddings available
        if not any(c.embedding for c in chunks):
            logger.info("No embeddings available, skipping deduplication")
            return chunks

        keep_indices: set[int] = set(range(len(chunks)))

        for i in range(len(chunks)):
            if i not in keep_indices:
                continue
            if not chunks[i].embedding:
                continue

            emb_i = np.array(chunks[i].embedding)

            for j in range(i + 1, len(chunks)):
                if j not in keep_indices:
                    continue
                if not chunks[j].embedding:
                    continue

                emb_j = np.array(chunks[j].embedding)
                sim = self._cosine_similarity(emb_i, emb_j)

                if sim > similarity_threshold:
                    keep_indices.discard(j)

        deduplicated = [chunks[i] for i in sorted(keep_indices)]
        removed_count = len(chunks) - len(deduplicated)

        logger.info(
            "Deduplication: %d chunks → %d chunks (removed %d duplicates)",
            len(chunks),
            len(deduplicated),
            removed_count,
        )

        return deduplicated

    # ── CRITICAL FIX #3: Enrichment Quality Gates ────────────────────

    def _validate_enrichment_quality(
        self, chunks: list[EnrichedChunk]
    ) -> dict[str, Any]:
        """Check enrichment coverage and confidence, warn if below targets."""
        if not chunks:
            return {}

        stats: dict[str, Any] = {
            "total_chunks": len(chunks),
            "chunks_with_entities": 0,
            "chunks_with_metrics": 0,
            "chunks_with_sentiment": 0,
            "avg_entity_confidence": 0.0,
            "min_entity_confidence": 1.0,
        }

        confidences: list[float] = []

        for chunk in chunks:
            if chunk.entities:
                stats["chunks_with_entities"] += 1
                confs = [e.confidence for e in chunk.entities]
                confidences.extend(confs)
                stats["min_entity_confidence"] = min(
                    stats["min_entity_confidence"], min(confs)
                )

            if chunk.metrics:
                stats["chunks_with_metrics"] += 1

            if chunk.sentiment:
                stats["chunks_with_sentiment"] += 1

        if confidences:
            stats["avg_entity_confidence"] = round(
                float(np.mean(confidences)), 3
            )

        # Convert to percentages
        total = len(chunks)
        stats["pct_chunks_with_entities"] = round(
            100 * stats["chunks_with_entities"] / total, 1
        )
        stats["pct_chunks_with_metrics"] = round(
            100 * stats["chunks_with_metrics"] / total, 1
        )
        stats["pct_chunks_with_sentiment"] = round(
            100 * stats["chunks_with_sentiment"] / total, 1
        )

        # Warnings if below targets
        if stats["pct_chunks_with_entities"] < 60:
            logger.warning(
                "Low entity coverage: %.1f%% (target 60%%)",
                stats["pct_chunks_with_entities"],
            )

        if stats["pct_chunks_with_metrics"] < 20:
            logger.warning(
                "Low metric coverage: %.1f%% (target 20%%)",
                stats["pct_chunks_with_metrics"],
            )

        if stats["avg_entity_confidence"] < 0.70:
            logger.warning(
                "Low entity confidence: %.3f (target 0.70)",
                stats["avg_entity_confidence"],
            )

        return stats

    # ── Step 1.1: File Loading ───────────────────────────────────────

    def _load_file(self, doc: InputDocument) -> LoadedDocument:
        path = Path(doc.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        file_type = self._detect_file_type(path)
        content = ""
        json_parsed = None
        csv_rows = None

        if file_type in (FileType.TXT, FileType.MD):
            content = self._read_text(path)
        elif file_type == FileType.PDF:
            content = self._read_pdf(path)
        elif file_type == FileType.DOCX:
            content = self._read_docx(path)
        elif file_type == FileType.JSON:
            raw = path.read_text(encoding="utf-8")
            json_parsed = json.loads(raw)
            content = raw
        elif file_type == FileType.CSV:
            csv_rows = self._read_csv(path)
            content = "\n".join(
                ", ".join(f"{k}: {v}" for k, v in row.items()) for row in csv_rows
            )

        if not content.strip():
            raise ValueError(f"Empty file after loading: {path}")

        return LoadedDocument(
            file_path=str(path),
            document_type=doc.type,
            file_type=file_type,
            content=content,
            json_parsed=json_parsed,
            csv_rows=csv_rows,
            file_size_kb=path.stat().st_size / 1024,
        )

    def _detect_file_type(self, path: Path) -> FileType:
        ext = path.suffix.lower().lstrip(".")
        mapping = {
            "pdf": FileType.PDF,
            "txt": FileType.TXT,
            "md": FileType.MD,
            "docx": FileType.DOCX,
            "json": FileType.JSON,
            "csv": FileType.CSV,
        }
        ft = mapping.get(ext)
        if not ft:
            raise ValueError(f"Unsupported file type: {ext}")
        return ft

    def _read_text(self, path: Path) -> str:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not read text file with any encoding: {path}")

    def _read_pdf(self, path: Path) -> str:
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            logger.warning("pdfplumber not available, trying PyPDF2")
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _read_docx(self, path: Path) -> str:
        from docx import Document

        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    def _read_csv(self, path: Path) -> list[dict[str, Any]]:
        text = self._read_text(path)
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    # ── Step 1.2: Normalization ──────────────────────────────────────

    def _normalize(self, doc: LoadedDocument) -> NormalizedContent:
        text = doc.content
        applied: list[str] = []

        # Standardize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        applied.append("newline_standardization")

        # Remove problematic special characters (keep useful punctuation)
        text = re.sub(r"[^\w\s.,?!:;()\"\'\\-\n/]", " ", text)
        applied.append("special_char_removal")

        # Collapse multiple spaces
        text = re.sub(r"[ \t]+", " ", text)
        applied.append("space_collapsing")

        # Strip leading/trailing
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = re.sub(r"\n{3,}", "\n\n", text)
        applied.append("whitespace_stripping")

        # Quality score based on content length and ratio of normal chars
        alnum_ratio = sum(c.isalnum() for c in text) / max(len(text), 1)
        quality = min(1.0, alnum_ratio * 1.5)

        return NormalizedContent(
            document_type=doc.document_type,
            file_type=doc.file_type,
            normalized_text=text.strip(),
            json_parsed=doc.json_parsed,
            csv_rows=doc.csv_rows,
            normalization_applied=applied,
            quality_score=round(quality, 2),
        )

    # ── Step 1.3: Semantic Chunking ──────────────────────────────────

    def _semantic_chunk(
        self,
        normalized: list[NormalizedContent],
        similarity_threshold: float = 0.8,
        max_tokens: int = 500,
        min_tokens: int = 100,
    ) -> list[EnrichedChunk]:
        chunks: list[EnrichedChunk] = []
        chunk_idx = 0

        for norm in normalized:
            if not norm.normalized_text:
                continue

            # Split into sentences
            sentences = self._split_sentences(norm.normalized_text)
            if not sentences:
                continue

            # Try embedding-based chunking, fall back to size-based
            embeddings = self._get_embeddings(sentences)

            current_chunk_sents: list[str] = []
            current_chunk_emb: Optional[np.ndarray] = None

            for i, sent in enumerate(sentences):
                sent_tokens = len(sent.split())
                if sent_tokens < 3:
                    continue

                if not current_chunk_sents:
                    current_chunk_sents.append(sent)
                    if embeddings is not None:
                        current_chunk_emb = embeddings[i]
                    continue

                # Check similarity
                should_merge = True
                if embeddings is not None and current_chunk_emb is not None:
                    sim = self._cosine_similarity(current_chunk_emb, embeddings[i])
                    should_merge = sim > similarity_threshold

                current_tokens = sum(len(s.split()) for s in current_chunk_sents)

                if should_merge and current_tokens + sent_tokens <= max_tokens:
                    current_chunk_sents.append(sent)
                    if embeddings is not None:
                        # Update chunk embedding as running average
                        n = len(current_chunk_sents)
                        current_chunk_emb = (
                            current_chunk_emb * (n - 1) + embeddings[i]
                        ) / n
                else:
                    # Flush current chunk
                    if current_tokens >= min_tokens:
                        chunk_text = " ".join(current_chunk_sents)
                        chunks.append(
                            EnrichedChunk(
                                chunk_id=f"chunk_{chunk_idx:04d}",
                                text=chunk_text,
                                tokens=current_tokens,
                                embedding=(
                                    current_chunk_emb.tolist()
                                    if current_chunk_emb is not None
                                    else None
                                ),
                                source_file=norm.file_type.value,
                                source_type=norm.document_type.value,
                                sequence=chunk_idx,
                            )
                        )
                        chunk_idx += 1

                    current_chunk_sents = [sent]
                    if embeddings is not None:
                        current_chunk_emb = embeddings[i]

            # Flush remainder
            remaining_tokens = sum(len(s.split()) for s in current_chunk_sents)
            if current_chunk_sents and remaining_tokens >= min_tokens:
                chunk_text = " ".join(current_chunk_sents)
                chunks.append(
                    EnrichedChunk(
                        chunk_id=f"chunk_{chunk_idx:04d}",
                        text=chunk_text,
                        tokens=remaining_tokens,
                        embedding=(
                            current_chunk_emb.tolist()
                            if current_chunk_emb is not None
                            else None
                        ),
                        source_file=norm.file_type.value,
                        source_type=norm.document_type.value,
                        sequence=chunk_idx,
                    )
                )
                chunk_idx += 1

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        # Split on sentence terminators followed by whitespace and capital letter
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        # Also split on newlines that seem like paragraph breaks
        result: list[str] = []
        for s in sentences:
            parts = s.split("\n\n")
            result.extend(p.strip() for p in parts if p.strip())
        return result

    # ── CRITICAL FIX #4 + OPT-2 + OPT-4: Embeddings ────────────────

    def _get_embeddings(self, sentences: list[str]) -> Optional[np.ndarray]:
        """Get sentence embeddings using sentence-transformers.

        Includes:
        - Embedding cache by sentence hash (OPT-4)
        - Batch encoding with batch_size=32 (OPT-2)
        - Proper error-level logging (CRITICAL FIX #4)
        """
        # Separate cached vs uncached sentences
        cached_indices: dict[int, np.ndarray] = {}
        uncached_indices: list[int] = []
        uncached_sents: list[str] = []

        for i, sent in enumerate(sentences):
            sent_hash = hash(sent)
            if sent_hash in self._embedding_cache:
                cached_indices[i] = self._embedding_cache[sent_hash]
            else:
                uncached_indices.append(i)
                uncached_sents.append(sent)

        try:
            # OPT-1: Lazy load with timing
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading sentence-transformers model (first use)...")
                t0 = time.time()
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info(
                    "sentence-transformers model loaded in %.2fs", time.time() - t0
                )

            # Encode only uncached sentences (OPT-2: batch_size=32)
            new_embeddings: Optional[np.ndarray] = None
            if uncached_sents:
                new_embeddings = self._embedder.encode(
                    uncached_sents,
                    show_progress_bar=False,
                    batch_size=32,
                )
                # Cache them (OPT-4)
                for idx, sent, emb in zip(
                    uncached_indices, uncached_sents, new_embeddings
                ):
                    sent_hash = hash(sent)
                    self._embedding_cache[sent_hash] = emb
                
                # Enforce cache size limit (Priority 3.2)
                if len(self._embedding_cache) > 10000:
                    keys_to_remove = list(self._embedding_cache.keys())[:len(self._embedding_cache) - 10000]
                    for k in keys_to_remove:
                        del self._embedding_cache[k]

            # Determine embedding dimension
            if new_embeddings is not None:
                emb_dim = new_embeddings.shape[1]
            elif cached_indices:
                emb_dim = next(iter(cached_indices.values())).shape[0]
            else:
                return None

            # Reconstruct full embedding array in order
            result = np.zeros((len(sentences), emb_dim))
            for i, sent in enumerate(sentences):
                sent_hash = hash(sent)
                result[i] = self._embedding_cache[sent_hash]

            logger.info(
                "Embedding: %d cached, %d new, cache size: %d",
                len(cached_indices),
                len(uncached_sents),
                len(self._embedding_cache),
            )
            return result

        except ImportError as e:
            logger.error(
                "sentence-transformers not installed: %s. "
                "Install with: pip install sentence-transformers",
                e,
            )
            return None
        except Exception as e:
            logger.error(
                "Embedding failed (chunking quality will be reduced): %s",
                e,
                exc_info=True,
            )
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    # ── Step 1.4: NLP Enrichment ─────────────────────────────────────

    async def _enrich_chunks(
        self, chunks: list[EnrichedChunk]
    ) -> list[EnrichedChunk]:
        """Enrich chunks with NER, sentiment, urgency, topics."""
        # Try local NLP first, fall back to LLM
        try:
            return self._enrich_local(chunks)
        except Exception as e:
            logger.info("Local NLP not available (%s), using LLM enrichment", e)
            return await self._enrich_with_llm(chunks)

    def _enrich_local(self, chunks: list[EnrichedChunk]) -> list[EnrichedChunk]:
        """Enrich using local spaCy + transformers."""
        # OPT-1: Lazy load with timing
        if self._nlp is None:
            logger.info("Loading spaCy model (first use)...")
            import spacy

            t0 = time.time()
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded in %.2fs", time.time() - t0)

        for chunk in chunks:
            # Segment-aware enrichment (GAP 4)
            if chunk.source_type == "interviews":
                chunk.is_customer_perspective = True
            elif chunk.source_type == "company_context":
                chunk.is_customer_perspective = False
                
            doc = self._nlp(chunk.text)

            # NER
            entities: list[ChunkEntity] = []
            for ent in doc.ents:
                etype = self._map_spacy_entity(ent.label_)
                entities.append(
                    ChunkEntity(
                        text=ent.text,
                        type=etype,
                        confidence=0.85,
                    )
                )
            chunk.entities = entities

            # Regex-based metric extraction
            chunk.metrics = self._extract_metrics(chunk.text)

            # Urgency from keywords
            chunk.urgency = self._estimate_urgency(chunk.text)

            # Simple sentiment from keyword analysis
            chunk.sentiment = self._simple_sentiment(chunk.text)

            # Topic classification from keywords
            chunk.topic_category, chunk.topic_confidence = self._classify_topic(
                chunk.text
            )

            chunk.enrichment_timestamp = datetime.utcnow()

        return chunks

    async def _enrich_with_llm(
        self, chunks: list[EnrichedChunk]
    ) -> list[EnrichedChunk]:
        """Enrich using LLM when local NLP is unavailable."""
        for chunk in chunks:
            # Segment-aware enrichment (GAP 4)
            if chunk.source_type == "interviews":
                chunk.is_customer_perspective = True
            elif chunk.source_type == "company_context":
                chunk.is_customer_perspective = False
                
            try:
                prompt = ENRICHMENT_USER.render(
                    text=chunk.text,
                    source_file=chunk.source_file,
                    source_type=chunk.source_type,
                )
                result = await self._llm.analyze(
                    system_prompt=ENRICHMENT_SYSTEM,
                    user_prompt=prompt,
                    temperature=0.2,
                    max_tokens=1000,
                )
                self._apply_llm_enrichment(chunk, result)
            except Exception as e:
                logger.warning("LLM enrichment failed for %s: %s", chunk.chunk_id, e)
                # Apply basic enrichment
                chunk.metrics = self._extract_metrics(chunk.text)
                chunk.urgency = self._estimate_urgency(chunk.text)
                chunk.sentiment = self._simple_sentiment(chunk.text)
                chunk.topic_category, chunk.topic_confidence = self._classify_topic(
                    chunk.text
                )

            chunk.enrichment_timestamp = datetime.utcnow()

        return chunks

    def _apply_llm_enrichment(
        self, chunk: EnrichedChunk, result: dict[str, Any]
    ) -> None:
        """Apply LLM enrichment results to a chunk."""
        # Entities
        raw_entities = result.get("entities", [])
        chunk.entities = [
            ChunkEntity(
                text=e.get("text", ""),
                type=e.get("type", "PRODUCT"),
                confidence=e.get("confidence", 0.5),
                value=e.get("value"),
                unit=e.get("unit"),
            )
            for e in raw_entities
        ]

        # Sentiment
        sent = result.get("sentiment", {})
        chunk.sentiment = SentimentResult(
            label=sent.get("label", "NEUTRAL"),
            score=sent.get("score", 0.5),
        )

        # Urgency
        chunk.urgency = result.get("urgency", 3)

        # Topic
        chunk.topic_category = result.get("topic_category", "feedback")
        chunk.topic_confidence = result.get("topic_confidence", 0.5)

        # Metrics
        raw_metrics = result.get("metrics", [])
        chunk.metrics = [
            ExtractedMetric(
                value=m.get("value", 0),
                unit=m.get("unit", ""),
                context=m.get("context", ""),
            )
            for m in raw_metrics
        ]

    def _map_spacy_entity(self, label: str) -> str:
        mapping = {
            "PERSON": "PERSON",
            "ORG": "ORG",
            "PRODUCT": "PRODUCT",
            "GPE": "ORG",
            "CARDINAL": "METRIC",
            "PERCENT": "METRIC",
            "MONEY": "METRIC",
            "QUANTITY": "METRIC",
            "PAIN_POINT": "PAIN_POINT",
            "CONSTRAINT": "CONSTRAINT",
        }
        return mapping.get(label, "PRODUCT")

    def _extract_metrics(self, text: str) -> list[ExtractedMetric]:
        metrics: list[ExtractedMetric] = []
        patterns = [
            r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(\w+)",
            r"(\d+(?:\.\d+)?)\s+(users?|customers?|tickets?|crashes?|requests?|times?)",
            r"\$(\d+(?:,\d+)*(?:\.\d+)?)\s*([KkMm])?",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1).replace(",", ""))
                    unit = match.group(2) if len(match.groups()) > 1 else ""
                    context = text[max(0, match.start() - 30) : match.end() + 30]
                    metrics.append(
                        ExtractedMetric(value=value, unit=unit or "", context=context)
                    )
                except (ValueError, IndexError):
                    pass
        return metrics

    def _estimate_urgency(self, text: str) -> int:
        text_lower = text.lower()
        if any(
            w in text_lower
            for w in ("critical", "urgent", "blocking", "asap", "emergency")
        ):
            return 5
        if any(
            w in text_lower
            for w in ("important", "soon", "need", "must", "required")
        ):
            return 4
        if any(w in text_lower for w in ("would like", "nice to have", "want")):
            return 3
        if any(w in text_lower for w in ("maybe", "eventually", "consider")):
            return 2
        return 1

    def _simple_sentiment(self, text: str) -> SentimentResult:
        text_lower = text.lower()
        pos = sum(
            1
            for w in ("great", "love", "excellent", "happy", "awesome", "improve")
            if w in text_lower
        )
        neg = sum(
            1
            for w in (
                "crash",
                "bug",
                "broken",
                "frustrat",
                "pain",
                "fail",
                "error",
                "problem",
            )
            if w in text_lower
        )
        if neg > pos:
            return SentimentResult(
                label=SentimentLabel.NEGATIVE, score=min(0.9, 0.5 + neg * 0.1)
            )
        elif pos > neg:
            return SentimentResult(
                label=SentimentLabel.POSITIVE, score=min(0.9, 0.5 + pos * 0.1)
            )
        return SentimentResult(label=SentimentLabel.NEUTRAL, score=0.5)

    def _classify_topic(self, text: str) -> tuple[TopicCategory, float]:
        text_lower = text.lower()
        scores = {
            TopicCategory.FEATURE_REQUEST: sum(
                1
                for w in ("feature", "request", "add", "would like", "need", "want")
                if w in text_lower
            ),
            TopicCategory.BUG_REPORT: sum(
                1
                for w in ("bug", "crash", "error", "broken", "fail", "issue")
                if w in text_lower
            ),
            TopicCategory.QUESTION: sum(
                1
                for w in ("how", "what", "why", "when", "?")
                if w in text_lower
            ),
            TopicCategory.FEEDBACK: sum(
                1
                for w in ("feedback", "suggest", "opinion", "think", "feel")
                if w in text_lower
            ),
            TopicCategory.CONSTRAINT: sum(
                1
                for w in ("constraint", "limit", "cannot", "impossible", "blocker")
                if w in text_lower
            ),
        }
        best = max(scores, key=lambda k: scores[k])
        total = sum(scores.values()) or 1
        return best, round(scores[best] / total, 2)

    # ── Step 1.5: Bundle Creation ────────────────────────────────────

    def _create_bundle(
        self, chunks: list[EnrichedChunk], processing_time: float, persona_ctx: dict[str, Any], market_ctx: dict[str, Any]
    ) -> ProblemContextBundle:
        # Build indices
        by_chunk_id: dict[str, Any] = {c.chunk_id: c.model_dump() for c in chunks}
        by_entity: dict[str, list[str]] = defaultdict(list)
        by_topic: dict[str, list[str]] = defaultdict(list)
        by_urgency: dict[str, list[str]] = defaultdict(list)
        by_sentiment: dict[str, list[str]] = defaultdict(list)

        entity_counter: Counter = Counter()
        topic_counter: Counter = Counter()
        sentiment_counter: Counter = Counter()
        urgency_sum = 0

        for chunk in chunks:
            for ent in chunk.entities:
                by_entity[ent.text].append(chunk.chunk_id)
                entity_counter[ent.text] += 1
            by_topic[chunk.topic_category.value].append(chunk.chunk_id)
            topic_counter[chunk.topic_category.value] += 1
            by_urgency[str(chunk.urgency)].append(chunk.chunk_id)
            by_sentiment[chunk.sentiment.label.value].append(chunk.chunk_id)
            sentiment_counter[chunk.sentiment.label.value] += 1
            urgency_sum += chunk.urgency

        # Sources
        sources: dict[str, SourceSummary] = {}
        for chunk in chunks:
            st = chunk.source_type
            if st not in sources:
                sources[st] = SourceSummary()
            sources[st].count += 1
            sources[st].chunk_ids.append(chunk.chunk_id)

        return ProblemContextBundle(
            chunks=chunks,
            sources=sources,
            indices={
                "by_chunk_id": by_chunk_id,
                "by_entity": dict(by_entity),
                "by_topic": dict(by_topic),
                "by_urgency": dict(by_urgency),
                "by_sentiment": dict(by_sentiment),
            },
            statistics=GlobalStatistics(
                total_chunks=len(chunks),
                unique_entities=len(entity_counter),
                entity_summary=[
                    {"name": name, "mentions": count}
                    for name, count in entity_counter.most_common(20)
                ],
                topic_distribution=dict(topic_counter),
                sentiment_distribution=dict(sentiment_counter),
                average_urgency=round(urgency_sum / max(len(chunks), 1), 1),
            ),
            external_persona_facts=persona_ctx.get("external_facts", {}),
            internal_stakeholder_facts=persona_ctx.get("internal_facts", {}),
            customer_segments_identified=persona_ctx.get("segments", []),
            customer_pain_points=persona_ctx.get("pain_points", {}),
            market_context=market_ctx,
            processing_stats={
                "total_files": len(sources),
                "total_chunks": len(chunks),
                "processing_time_seconds": round(processing_time, 1),
            },
        )

    # ── Structured Extraction ────────────────────────────────────────

    def _extract_feature_proposal(
        self, normalized: list[NormalizedContent]
    ) -> FeatureProposal:
        for n in normalized:
            if n.document_type == DocumentType.FEATURE_PROPOSAL and n.json_parsed:
                data = n.json_parsed
                return FeatureProposal(
                    title=data.get("title", "Unknown Feature"),
                    description=data.get("description", ""),
                    target_users=data.get("target_users", ""),
                    target_user_count=data.get("target_user_count"),
                    effort_weeks_min=data.get("effort_weeks_min")
                    or data.get("effort_weeks"),
                    effort_weeks_max=data.get("effort_weeks_max")
                    or data.get("effort_weeks"),
                    affected_domains=data.get("affected_domains", []),
                    existing_features=data.get("existing_features", []),
                    tech_stack=data.get("tech_stack", []),
                    priority=data.get("priority"),
                    revenue_model=data.get("revenue_model"),
                    pricing_strategy=data.get("pricing_strategy"),
                    customer_segments=data.get("customer_segments", []),
                )
        return FeatureProposal(
            title="Unspecified Feature", description="No proposal found"
        )

    def _extract_company_context(
        self, normalized: list[NormalizedContent]
    ) -> CompanyContext:
        for n in normalized:
            if n.document_type == DocumentType.COMPANY_CONTEXT and n.json_parsed:
                data = n.json_parsed
                return CompanyContext(
                    company_name=data.get("company_name", ""),
                    team_size=data.get("team_size"),
                    budget=data.get("budget"),
                    tech_stack=data.get("tech_stack", []),
                    current_priorities=data.get("current_priorities", []),
                    competitors=data.get("competitors", []),
                    constraints=data.get("constraints", []),
                    stakeholders=data.get("stakeholders", []),
                )
        return CompanyContext()

    def _extract_persona_context(
        self, normalized: list[NormalizedContent]
    ) -> dict[str, Any]:
        """Extract facts for internal and external personas."""
        external_facts: dict[str, list[str]] = defaultdict(list)
        internal_facts: dict[str, list[str]] = defaultdict(list)
        segments: set[str] = set()
        pain_points: dict[str, list[str]] = defaultdict(list)
        
        for n in normalized:
            if n.document_type == DocumentType.COMPANY_CONTEXT and n.json_parsed:
                data = n.json_parsed
                for sh in data.get("stakeholders", []):
                    name = sh.get("name", "Unknown")
                    facts = sh.get("facts", [])
                    internal_facts[name].extend(facts)
            
            # Simulated parsing from text for interviews if json isn't available
            if n.document_type == DocumentType.INTERVIEWS and n.normalized_text:
                text = n.normalized_text
                sentences = self._split_sentences(text)
                for sent in sentences:
                    lower_sent = sent.lower()
                    if "technician" in lower_sent or "tech" in lower_sent:
                        segments.add("Field Technician")
                        external_facts["Field Technician"].append(sent)
                        if "pain" in lower_sent or "frustrat" in lower_sent or "issue" in lower_sent:
                            pain_points["Field Technician"].append(sent)
                    elif "manager" in lower_sent:
                        segments.add("Manager")
                        external_facts["Manager"].append(sent)
                        if "pain" in lower_sent or "frustrat" in lower_sent or "issue" in lower_sent:
                            pain_points["Manager"].append(sent)
                    elif "power user" in lower_sent:
                        segments.add("Power User")
                        external_facts["Power User"].append(sent)
                        if "pain" in lower_sent or "frustrat" in lower_sent or "issue" in lower_sent:
                            pain_points["Power User"].append(sent)
                        
        return {
            "external_facts": dict(external_facts),
            "internal_facts": dict(internal_facts),
            "segments": list(segments),
            "pain_points": dict(pain_points),
        }

    def _extract_market_context(
        self, normalized: list[NormalizedContent]
    ) -> dict[str, Any]:
        """Extract market conditions for Monte Carlo simulation."""
        pricing_tiers: list[str] = []
        competitors: list[str] = []
        connectivity_patterns: dict[str, str] = {}
        customer_geography: list[str] = []
        usage_patterns: dict[str, str] = {}
        
        for n in normalized:
            if n.document_type == DocumentType.COMPANY_CONTEXT and n.json_parsed:
                data = n.json_parsed
                pricing_tiers.extend(data.get("pricing_tiers", []))
                competitors.extend(data.get("competitors", []))
            
            if n.document_type == DocumentType.ANALYTICS and n.json_parsed:
                data = n.json_parsed
                geo = data.get("geography", [])
                customer_geography.extend(geo)
                conn = data.get("connectivity_patterns", {})
                connectivity_patterns.update(conn)
                
            if n.document_type == DocumentType.SUPPORT_TICKETS:
                pass
        
        if not customer_geography:
            customer_geography = ["Urban", "Suburban"]
        
        return {
            "pricing_tiers": pricing_tiers,
            "competitors": competitors,
            "connectivity_patterns": connectivity_patterns,
            "geography": customer_geography,
            "usage_patterns": usage_patterns,
        }
