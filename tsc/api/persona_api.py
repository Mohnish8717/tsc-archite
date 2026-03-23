from __future__ import annotations

import uuid
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from tsc.db.connection import get_db
from tsc.repositories.persona_repository import PersonaRepository
from tsc.repositories.feature_repository import FeatureRepository
from tsc.repositories.simulation_repository import SimulationRepository
from tsc.caching.lru_cache import PersonaCache
from tsc.models.inputs import CompanyContext, FeatureProposal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2")

# Global app state (simplified for API)
_cache = PersonaCache()
_db = get_db(cache=_cache)

@router.get("/personas/company/{company_id}")
async def get_company_personas(company_id: uuid.UUID):
    """Retrieve all internal personas for a company."""
    repo = PersonaRepository(_db, cache=_cache)
    personas = await repo.get_internal_personas_by_company(company_id)
    return {"personas": [str(p) for p in personas]}

@router.get("/personas/market/{segment}")
async def get_market_personas(segment: str):
    """Retrieve all external personas for a market segment."""
    repo = PersonaRepository(_db, cache=_cache)
    personas = await repo.get_external_personas_by_segment(segment)
    return {"personas": [str(p) for p in personas]}

@router.get("/simulations/history/{company_id}")
async def get_simulation_history(company_id: uuid.UUID):
    """Retrieve simulation history for a company."""
    repo = SimulationRepository(_db)
    runs = await repo.get_runs_by_company(company_id)
    return {"history": [str(r) for r in runs]}

@router.get("/simulations/run/{run_id}")
async def get_simulation_run(run_id: uuid.UUID):
    """Retrieve details for a single simulation run."""
    repo = SimulationRepository(_db)
    run = await repo.get_run_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run": str(run)}
