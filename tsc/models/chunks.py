"""Chunk, enrichment, and Problem-Context Bundle schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class TopicCategory(str, Enum):
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    QUESTION = "question"
    FEEDBACK = "feedback"
    CONSTRAINT = "constraint"


class EntityType(str, Enum):
    PERSON = "PERSON"
    ORG = "ORG"
    PRODUCT = "PRODUCT"
    CONSTRAINT = "CONSTRAINT"
    PAIN_POINT = "PAIN_POINT"
    METRIC = "METRIC"


class ChunkEntity(BaseModel):
    """An entity detected within a chunk."""

    text: str
    type: EntityType
    value: Optional[float] = None
    unit: Optional[str] = None
    confidence: float = 0.0
    sentiment: Optional[SentimentLabel] = None


class SentimentResult(BaseModel):
    label: SentimentLabel = SentimentLabel.NEUTRAL
    score: float = 0.0


class ExtractedMetric(BaseModel):
    value: float
    unit: str = ""
    context: str = ""


class EnrichedChunk(BaseModel):
    """A semantically-chunked, NLP-enriched piece of text."""

    chunk_id: str
    text: str
    tokens: int = 0
    embedding: Optional[list[float]] = None
    topics: list[str] = Field(default_factory=list)
    start_pos: int = 0
    end_pos: int = 0
    source_file: str = ""
    source_type: str = ""
    sequence: int = 0
    coherence_score: float = 0.0

    # NLP enrichments
    entities: list[ChunkEntity] = Field(default_factory=list)
    sentiment: SentimentResult = Field(default_factory=SentimentResult)
    urgency: int = 3  # 1-5 scale
    topic_category: TopicCategory = TopicCategory.FEEDBACK
    topic_confidence: float = 0.0
    metrics: list[ExtractedMetric] = Field(default_factory=list)
    is_customer_perspective: bool = False
    enrichment_timestamp: Optional[datetime] = None


class SourceSummary(BaseModel):
    """Summary of chunks from a single source."""

    count: int = 0
    chunk_ids: list[str] = Field(default_factory=list)


class GlobalStatistics(BaseModel):
    """Aggregate statistics across all chunks."""

    total_chunks: int = 0
    unique_entities: int = 0
    entity_summary: list[dict[str, Any]] = Field(default_factory=list)
    topic_distribution: dict[str, int] = Field(default_factory=dict)
    sentiment_distribution: dict[str, int] = Field(default_factory=dict)
    average_urgency: float = 0.0


class ProblemContextBundle(BaseModel):
    """Unified output of Layer 1: all chunks, indices, and statistics."""

    chunks: list[EnrichedChunk] = Field(default_factory=list)
    sources: dict[str, SourceSummary] = Field(default_factory=dict)
    indices: dict[str, dict[str, Any]] = Field(default_factory=dict)
    statistics: GlobalStatistics = Field(default_factory=GlobalStatistics)
    
    # External persona generation context
    external_persona_facts: dict[str, list[str]] = Field(default_factory=dict)
    internal_stakeholder_facts: dict[str, list[str]] = Field(default_factory=dict)
    customer_segments_identified: list[str] = Field(default_factory=list)
    customer_pain_points: dict[str, list[str]] = Field(default_factory=dict)
    
    # Market context for Monte Carlo
    market_context: dict[str, Any] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processing_stats: dict[str, Any] = Field(default_factory=dict)
