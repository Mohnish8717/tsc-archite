"""Input document and file-loading schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    INTERVIEWS = "interviews"
    SUPPORT_TICKETS = "support_tickets"
    ANALYTICS = "analytics"
    COMPANY_CONTEXT = "company_context"
    FEATURE_PROPOSAL = "feature_proposal"


class FileType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    JSON = "json"
    CSV = "csv"
    MD = "md"


class InputDocument(BaseModel):
    """A raw input document to be processed."""

    type: DocumentType
    file_path: str
    description: str = ""


class LoadedDocument(BaseModel):
    """Result of loading and validating a single input file."""

    file_path: str
    document_type: DocumentType
    file_type: FileType
    content: str = ""
    json_parsed: Optional[dict[str, Any]] = None
    csv_rows: Optional[list[dict[str, Any]]] = None
    load_timestamp: datetime = Field(default_factory=datetime.utcnow)
    file_size_kb: float = 0
    encoding: str = "UTF-8"
    status: str = "loaded"


class NormalizedContent(BaseModel):
    """Content after normalization pass."""

    document_type: DocumentType
    file_type: FileType
    normalized_text: str = ""
    json_parsed: Optional[dict[str, Any]] = None
    csv_rows: Optional[list[dict[str, Any]]] = None
    normalization_applied: list[str] = Field(default_factory=list)
    quality_score: float = 0.0


class FeatureProposal(BaseModel):
    """The feature being evaluated — extracted from the proposal file."""

    title: str
    description: str
    target_users: str = ""
    target_user_count: Optional[int] = None
    effort_weeks_min: Optional[float] = None
    effort_weeks_max: Optional[float] = None
    affected_domains: list[str] = Field(default_factory=list)
    existing_features: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    priority: Optional[str] = None
    revenue_model: Optional[str] = None
    pricing_strategy: Optional[str] = None
    customer_segments: list[str] = Field(default_factory=list)


class CompanyContext(BaseModel):
    """Organizational context — extracted from company context file."""

    company_id: Optional[uuid.UUID] = None
    company_name: str = ""
    team_size: Optional[int] = None
    budget: Optional[str] = None
    tech_stack: list[str] = Field(default_factory=list)
    current_priorities: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    stakeholders: list[dict[str, str]] = Field(default_factory=list)
