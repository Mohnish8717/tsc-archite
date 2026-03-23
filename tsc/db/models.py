from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Float, ForeignKey, JSON, String, DateTime, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    industry: Mapped[Optional[str]] = mapped_column(String)
    revenue_range: Mapped[Optional[str]] = mapped_column(String)
    employee_count: Mapped[Optional[int]] = mapped_column(Integer)
    business_context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    features: Mapped[list[Feature]] = relationship("Feature", back_populates="company", cascade="all, delete-orphan")
    internal_personas: Mapped[list[InternalPersona]] = relationship("InternalPersona", back_populates="company", cascade="all, delete-orphan")
    simulation_runs: Mapped[list[SimulationRun]] = relationship("SimulationRun", back_populates="company", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Company(name='{self.name}', industry='{self.industry}')>"


class Feature(Base):
    __tablename__ = "features"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("companies.id", on_delete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    target_audience: Mapped[list[str]] = mapped_column(JSON, default=list)
    strategic_objective: Mapped[Optional[str]] = mapped_column(Text)
    effort_weeks_min: Mapped[Optional[int]] = mapped_column(Integer)
    effort_weeks_max: Mapped[Optional[int]] = mapped_column(Integer)
    business_context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    company: Mapped[Company] = relationship("Company", back_populates="features")
    predictions: Mapped[list[FeaturePersonaPrediction]] = relationship("FeaturePersonaPrediction", back_populates="feature", cascade="all, delete-orphan")
    simulation_runs: Mapped[list[SimulationRun]] = relationship("SimulationRun", back_populates="feature", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Feature(title='{self.title}', company_id='{self.company_id}')>"


class InternalPersona(Base):
    __tablename__ = "internal_personas"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("companies.id", on_delete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String)
    mbti_type: Mapped[Optional[str]] = mapped_column(String(4))
    psychological_profile: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    company: Mapped[Company] = relationship("Company", back_populates="internal_personas")

    def __repr__(self) -> str:
        return f"<InternalPersona(name='{self.name}', role='{self.role}')>"


class ExternalPersona(Base):
    __tablename__ = "external_personas"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)  # segment profile
    title: Mapped[Optional[str]] = mapped_column(String)
    mbti_type: Mapped[Optional[str]] = mapped_column(String(4))
    psychological_profile: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    use_case: Mapped[Optional[str]] = mapped_column(Text)
    demographics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<ExternalPersona(name='{self.name}', role='{self.role}')>"


class FeaturePersonaPrediction(Base):
    __tablename__ = "feature_persona_predictions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    feature_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("features.id", on_delete="CASCADE"), nullable=False)
    persona_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    persona_type: Mapped[str] = mapped_column(String(10), nullable=False)  # 'INTERNAL' or 'EXTERNAL'
    predicted_stance: Mapped[str] = mapped_column(String(20), nullable=False)
    objections: Mapped[list[str]] = mapped_column(JSON, default=list)
    conditions: Mapped[list[str]] = mapped_column(JSON, default=list)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    feature: Mapped[Feature] = relationship("Feature", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<Prediction(feature_id='{self.feature_id}', stance='{self.predicted_stance}')>"


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    feature_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("features.id", on_delete="CASCADE"), nullable=False)
    company_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("companies.id", on_delete="CASCADE"), nullable=False)
    approval_rate: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_assessment: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    recommendations: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    simulation_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    
    simulation_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    feature: Mapped[Feature] = relationship("Feature", back_populates="simulation_runs")
    company: Mapped[Company] = relationship("Company", back_populates="simulation_runs")

    def __repr__(self) -> str:
        return f"<SimulationRun(id='{self.id}', approval_rate={self.approval_rate:.2f})>"
