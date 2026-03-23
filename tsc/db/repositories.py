from __future__ import annotations

import logging
import uuid
from typing import Any, Optional, Sequence, TypeVar, Generic

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from tsc.db.models import Company, Feature, InternalPersona, ExternalPersona, FeaturePersonaPrediction, SimulationRun

logger = logging.getLogger(__name__)

T = TypeVar("T")

class BaseRepository(Generic[T]):
    """Base repository with common operations."""
    def __init__(self, session: AsyncSession):
        self.session = session


class PersonaRepository(BaseRepository):
    """Repository for internal and external personas."""

    async def save_internal_persona(self, company_id: uuid.UUID, persona_data: dict[str, Any]) -> uuid.UUID:
        """Save a new internal persona profile."""
        persona = InternalPersona(
            company_id=company_id,
            name=persona_data.get("name"),
            role=persona_data.get("role"),
            title=persona_data.get("title"),
            mbti_type=persona_data.get("mbti_type"),
            psychological_profile=persona_data.get("psychological_profile", {}),
            confidence_score=persona_data.get("confidence_score", 0.0)
        )
        self.session.add(persona)
        await self.session.flush()
        return persona.id

    async def get_internal_personas_by_company(self, company_id: uuid.UUID) -> Sequence[InternalPersona]:
        """Retrieve all internal personas for a specific company."""
        result = await self.session.execute(
            select(InternalPersona).where(InternalPersona.company_id == company_id)
        )
        return result.scalars().all()

    async def save_external_persona(self, persona_data: dict[str, Any]) -> uuid.UUID:
        """Save a new external customer persona."""
        persona = ExternalPersona(
            name=persona_data.get("name"),
            role=persona_data.get("role"),
            title=persona_data.get("title"),
            mbti_type=persona_data.get("mbti_type"),
            psychological_profile=persona_data.get("psychological_profile", {}),
            use_case=persona_data.get("use_case"),
            demographics=persona_data.get("demographics", {}),
            confidence_score=persona_data.get("confidence_score", 0.0)
        )
        self.session.add(persona)
        await self.session.flush()
        return persona.id

    async def get_external_personas_by_segment(self, segment: str) -> Sequence[ExternalPersona]:
        """Retrieve external personas by market segment (role)."""
        result = await self.session.execute(
            select(ExternalPersona).where(ExternalPersona.role == segment)
        )
        return result.scalars().all()

    async def update_persona(self, persona_id: uuid.UUID, is_internal: bool, updates: dict[str, Any]) -> bool:
        """Update an existing persona's data."""
        model = InternalPersona if is_internal else ExternalPersona
        await self.session.execute(
            update(model).where(model.id == persona_id).values(**updates)
        )
        return True

    async def delete_persona(self, persona_id: uuid.UUID, is_internal: bool) -> bool:
        """Delete a persona by ID."""
        model = InternalPersona if is_internal else ExternalPersona
        await self.session.execute(
            delete(model).where(model.id == persona_id)
        )
        return True


class FeatureRepository(BaseRepository):
    """Repository for feature proposals."""

    async def save_feature(self, company_id: uuid.UUID, feature_data: dict[str, Any]) -> uuid.UUID:
        """Save a new feature proposal."""
        feature = Feature(
            company_id=company_id,
            title=feature_data.get("title"),
            description=feature_data.get("description"),
            target_audience=feature_data.get("target_audience", []),
            strategic_objective=feature_data.get("strategic_objective"),
            effort_weeks_min=feature_data.get("effort_weeks_min"),
            effort_weeks_max=feature_data.get("effort_weeks_max"),
            business_context=feature_data.get("business_context", {})
        )
        self.session.add(feature)
        await self.session.flush()
        return feature.id

    async def get_feature_by_id(self, feature_id: uuid.UUID) -> Optional[Feature]:
        """Retrieve a specific feature by ID."""
        result = await self.session.get(Feature, feature_id)
        return result

    async def get_features_by_company(self, company_id: uuid.UUID) -> Sequence[Feature]:
        """Retrieve all features for a company."""
        result = await self.session.execute(
            select(Feature).where(Feature.company_id == company_id)
        )
        return result.scalars().all()


class PredictionRepository(BaseRepository):
    """Repository for feature-persona predictions."""

    async def save_prediction(self, feature_id: uuid.UUID, persona_id: uuid.UUID, persona_type: str, stance_data: dict[str, Any]) -> uuid.UUID:
        """Save a prediction for a feature-persona pair."""
        prediction = FeaturePersonaPrediction(
            feature_id=feature_id,
            persona_id=persona_id,
            persona_type=persona_type,
            predicted_stance=stance_data.get("predicted_stance"),
            objections=stance_data.get("objections", []),
            conditions=stance_data.get("conditions", []),
            confidence_score=stance_data.get("confidence_score", 0.0),
            reasoning=stance_data.get("reasoning")
        )
        self.session.add(prediction)
        await self.session.flush()
        return prediction.id

    async def get_predictions_for_feature(self, feature_id: uuid.UUID) -> Sequence[FeaturePersonaPrediction]:
        """Retrieve all predictions for a specific feature."""
        result = await self.session.execute(
            select(FeaturePersonaPrediction).where(FeaturePersonaPrediction.feature_id == feature_id)
        )
        return result.scalars().all()


class SimulationRepository(BaseRepository):
    """Repository for simulation runs."""

    async def save_simulation_run(self, feature_id: uuid.UUID, company_id: uuid.UUID, results: dict[str, Any]) -> uuid.UUID:
        """Save the results of a complete simulation run."""
        run = SimulationRun(
            feature_id=feature_id,
            company_id=company_id,
            approval_rate=results.get("approval_rate", 0.0),
            sentiment_score=results.get("sentiment_score", 0.0),
            risk_assessment=results.get("risk_assessment", {}),
            recommendations=results.get("recommendations", []),
            simulation_metadata=results.get("simulation_metadata", {}),
            duration_seconds=results.get("duration_seconds")
        )
        self.session.add(run)
        await self.session.flush()
        return run.id

    async def get_simulation_history(self, feature_id: uuid.UUID, limit: int = 10) -> Sequence[SimulationRun]:
        """Retrieve historical simulation runs for a feature."""
        result = await self.session.execute(
            select(SimulationRun)
            .where(SimulationRun.feature_id == feature_id)
            .order_by(SimulationRun.simulation_timestamp.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def get_company_simulations(self, company_id: uuid.UUID) -> Sequence[SimulationRun]:
        """Retrieve all simulations run for a specific company."""
        result = await self.session.execute(
            select(SimulationRun).where(SimulationRun.company_id == company_id)
        )
        return result.scalars().all()
