"""Prediction repository for feature-persona adoption forecasts."""

from __future__ import annotations

import logging
from typing import List, Sequence
from uuid import UUID

from sqlalchemy import select
from tsc.db.models import FeaturePersonaPrediction
from tsc.db.connection import DatabaseConnection

logger = logging.getLogger(__name__)


class PredictionRepository:
    """Repository for feature-persona predictions."""
    
    def __init__(self, db: DatabaseConnection):
        self.db = db

    async def save_prediction(self, prediction_data: dict) -> UUID:
        """Save a prediction for a feature-persona pair."""
        async with self.db.get_session() as session:
            prediction = FeaturePersonaPrediction(
                feature_id=prediction_data.get("feature_id"),
                persona_id=prediction_data.get("persona_id"),
                simulation_run_id=prediction_data.get("simulation_run_id"),
                predicted_adoption_rate=prediction_data.get("predicted_adoption_rate", 0.0),
                sentiment_score=prediction_data.get("sentiment_score", 0.0),
                qualitative_feedback=prediction_data.get("qualitative_feedback"),
                confidence_score=prediction_data.get("confidence_score", 0.0),
                metadata=prediction_data.get("metadata", {})
            )
            session.add(prediction)
            await session.flush()
            prediction_id = prediction.id
            
        logger.info(f"Saved prediction for feature {prediction_data.get('feature_id')} (ID: {prediction_id})")
        return prediction_id

    async def get_predictions_for_run(self, run_id: UUID) -> List[FeaturePersonaPrediction]:
        """Retrieve all predictions for a specific simulation run."""
        async with self.db.get_session() as session:
            stmt = select(FeaturePersonaPrediction).where(FeaturePersonaPrediction.simulation_run_id == run_id)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_predictions_for_feature(self, feature_id: UUID) -> List[FeaturePersonaPrediction]:
        """Retrieve all predictions for a specific feature."""
        async with self.db.get_session() as session:
            stmt = select(FeaturePersonaPrediction).where(FeaturePersonaPrediction.feature_id == feature_id)
            result = await session.execute(stmt)
            return list(result.scalars().all())
