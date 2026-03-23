"""Feature repository with database integration."""

from __future__ import annotations

import logging
from typing import Optional, List, Sequence
from uuid import UUID

from sqlalchemy import select
from tsc.db.models import Feature
from tsc.db.connection import DatabaseConnection

logger = logging.getLogger(__name__)


class FeatureRepository:
    """Repository for feature operations."""
    
    def __init__(self, db: DatabaseConnection):
        self.db = db

    async def save_feature(self, feature_data: dict) -> UUID:
        """Save a new feature proposal."""
        async with self.db.get_session() as session:
            feature = Feature(
                company_id=feature_data.get("company_id"),
                title=feature_data.get("title"),
                description=feature_data.get("description"),
                target_audience=feature_data.get("target_audience", []),
                strategic_objective=feature_data.get("strategic_objective"),
                effort_weeks_min=feature_data.get("effort_weeks_min"),
                effort_weeks_max=feature_data.get("effort_weeks_max"),
                business_context=feature_data.get("business_context", {}),
                metadata=feature_data.get("metadata", {})
            )
            session.add(feature)
            await session.flush()
            feature_id = feature.id
        
        logger.info(f"Saved feature: {feature_data.get('title')} (ID: {feature_id})")
        return feature_id

    async def get_feature_by_id(self, feature_id: UUID) -> Optional[Feature]:
        """Retrieve a specific feature by ID."""
        async with self.db.get_session() as session:
            return await session.get(Feature, feature_id)

    async def get_features_by_company(self, company_id: UUID) -> List[Feature]:
        """Retrieve all features for a company."""
        async with self.db.get_session() as session:
            stmt = select(Feature).where(Feature.company_id == company_id)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_feature_by_title(self, title: str, company_id: UUID) -> Optional[Feature]:
        """Retrieve a specific feature by title and company."""
        async with self.db.get_session() as session:
            stmt = select(Feature).where(
                Feature.title == title,
                Feature.company_id == company_id
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
