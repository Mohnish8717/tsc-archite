"""Simulation repository for persistent tracking of simulation outcomes."""

from __future__ import annotations

import logging
from typing import List, Sequence
from uuid import UUID

from sqlalchemy import select
from tsc.db.models import SimulationRun
from tsc.db.connection import DatabaseConnection

logger = logging.getLogger(__name__)


class SimulationRepository:
    """Repository for simulation runs."""
    
    def __init__(self, db: DatabaseConnection):
        self.db = db

    async def save_simulation_run(self, run_data: dict) -> UUID:
        """Save the results of a complete simulation run."""
        async with self.db.get_session() as session:
            run = SimulationRun(
                feature_id=run_data.get("feature_id"),
                company_id=run_data.get("company_id"),
                approval_rate=run_data.get("approval_rate", 0.0),
                sentiment_score=run_data.get("sentiment_score", 0.0),
                risk_assessment=run_data.get("risk_assessment", 0.5),
                recommendations=run_data.get("recommendations"),
                metadata_snapshot=run_data.get("metadata_snapshot", {})
            )
            session.add(run)
            await session.flush()
            run_id = run.id
        
        logger.info(f"Saved simulation run: {run_id}")
        return run_id

    async def get_runs_by_feature(self, feature_id: UUID) -> List[SimulationRun]:
        """Retrieve historical simulation runs for a feature."""
        async with self.db.get_session() as session:
            stmt = select(SimulationRun).where(SimulationRun.feature_id == feature_id).order_by(SimulationRun.run_timestamp.desc())
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_runs_by_company(self, company_id: UUID) -> List[SimulationRun]:
        """Retrieve all simulations run for a specific company."""
        async with self.db.get_session() as session:
            stmt = select(SimulationRun).where(SimulationRun.company_id == company_id).order_by(SimulationRun.run_timestamp.desc())
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_run_by_id(self, run_id: UUID) -> Optional[SimulationRun]:
        """Retrieve a specific simulation run by ID."""
        async with self.db.get_session() as session:
            return await session.get(SimulationRun, run_id)
