"""Persona repository with automatic cache invalidation."""

from __future__ import annotations

import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from tsc.db.models import InternalPersona, ExternalPersona
from tsc.caching.lru_cache import PersonaCache
from tsc.db.connection import DatabaseConnection

logger = logging.getLogger(__name__)


class PersonaRepository:
    """
    Repository for persona operations with cache-aware write safety.
    
    All write operations automatically invalidate the cache
    to maintain consistency.
    """
    
    def __init__(self, db: DatabaseConnection, cache: Optional[PersonaCache] = None):
        self.db = db
        self.cache = cache

    # ──────────────────────────────────────────────────────
    # INTERNAL PERSONAS
    # ──────────────────────────────────────────────────────

    async def save_internal_persona(
        self,
        company_id: UUID,
        persona_data: dict
    ) -> UUID:
        """
        Save or update internal persona.
        
        Automatically invalidates cache after successful save.
        
        Args:
            company_id: Company ID
            persona_data: Persona data dictionary
            
        Returns:
            Persona ID
        """
        async with self.db.get_session() as session:
            # Check if exists
            stmt = select(InternalPersona).where(
                InternalPersona.company_id == company_id,
                InternalPersona.name == persona_data.get("name"),
                InternalPersona.role == persona_data.get("role"),
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update
                for key, value in persona_data.items():
                    setattr(existing, key, value)
                persona_obj = existing
            else:
                # Create new
                persona_obj = InternalPersona(company_id=company_id, **persona_data)
                session.add(persona_obj)
            
            await session.flush()  # Get ID before commit
            persona_id = persona_obj.id
        
        # ✅ Cache invalidation AFTER successful commit
        await self.db.invalidate_persona(persona_id)
        logger.info(f"Saved internal persona {persona_data['name']} (ID: {persona_id})")
        
        return persona_id

    async def get_internal_persona(
        self,
        persona_id: UUID
    ) -> Optional[InternalPersona]:
        """
        Get internal persona by ID (cache-aware).
        
        Checks cache first, falls back to DB.
        
        Args:
            persona_id: Persona ID
            
        Returns:
            InternalPersona or None
        """
        # Try cache first
        if self.cache:
            cached = await self.cache.get_persona(str(persona_id))
            if cached:
                return cached
        
        # Fall back to DB
        async with self.db.get_session() as session:
            stmt = select(InternalPersona).where(InternalPersona.id == persona_id)
            result = await session.execute(stmt)
            persona = result.scalar_one_or_none()
            
            # Cache for next time
            if persona and self.cache:
                await self.cache.set_internal(str(persona_id), persona)
            
            return persona

    async def get_internal_personas_by_company(
        self,
        company_id: UUID
    ) -> List[InternalPersona]:
        """
        Get all internal personas for company (cache-aware list).
        
        Args:
            company_id: Company ID
            
        Returns:
            List of InternalPersona
        """
        # Check list cache
        cache_key = f"personas:company:{company_id}:internal"
        if self.cache:
            cached = await self.cache.get_list(cache_key)
            if cached:
                return cached
        
        # Query DB
        async with self.db.get_session() as session:
            stmt = select(InternalPersona).where(
                InternalPersona.company_id == company_id
            )
            result = await session.execute(stmt)
            personas = result.scalars().all()
            
            # Cache list
            if self.cache:
                await self.cache.set_list(cache_key, personas)
            
            return personas

    async def update_internal_persona(
        self,
        persona_id: UUID,
        updates: dict
    ) -> None:
        """
        Update internal persona (cache-aware).
        
        Automatically invalidates cache.
        
        Args:
            persona_id: Persona ID
            updates: Fields to update
        """
        async with self.db.get_session() as session:
            stmt = (
                update(InternalPersona)
                .where(InternalPersona.id == persona_id)
                .values(**updates, updated_at=datetime.utcnow())
            )
            await session.execute(stmt)
        
        # ✅ Invalidate cache after DB commit
        await self.db.invalidate_persona(persona_id)
        logger.info(f"Updated internal persona {persona_id}")

    async def delete_internal_persona(self, persona_id: UUID) -> None:
        """
        Delete internal persona (cache-aware).
        
        Args:
            persona_id: Persona ID
        """
        async with self.db.get_session() as session:
            stmt = select(InternalPersona).where(InternalPersona.id == persona_id)
            result = await session.execute(stmt)
            persona = result.scalar_one_or_none()
            
            if persona:
                await session.delete(persona)
        
        # ✅ Invalidate cache
        await self.db.invalidate_persona(persona_id)
        logger.info(f"Deleted internal persona {persona_id}")

    # ──────────────────────────────────────────────────────
    # EXTERNAL PERSONAS
    # ──────────────────────────────────────────────────────

    async def save_external_persona(
        self,
        persona_data: dict
    ) -> UUID:
        """Save or update external persona (cache-aware)."""
        async with self.db.get_session() as session:
            stmt = select(ExternalPersona).where(
                ExternalPersona.segment_type == persona_data.get("segment_type"),
                ExternalPersona.persona_name == persona_data.get("persona_name"),
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                for key, value in persona_data.items():
                    setattr(existing, key, value)
                persona_obj = existing
            else:
                persona_obj = ExternalPersona(**persona_data)
                session.add(persona_obj)
            
            await session.flush()
            persona_id = persona_obj.id
        
        # ✅ Cache invalidation
        await self.db.invalidate_persona(persona_id)
        logger.info(f"Saved external persona {persona_data['persona_name']}")
        
        return persona_id

    async def get_external_personas_by_segment(
        self,
        segment_type: str
    ) -> List[ExternalPersona]:
        """Get external personas by segment (cache-aware list)."""
        cache_key = f"personas:segment:{segment_type}"
        
        if self.cache:
            cached = await self.cache.get_list(cache_key)
            if cached:
                return cached
        
        async with self.db.get_session() as session:
            stmt = select(ExternalPersona).where(
                ExternalPersona.segment_type == segment_type
            )
            result = await session.execute(stmt)
            personas = result.scalars().all()
            
            if self.cache:
                await self.cache.set_list(cache_key, personas)
            
            return personas

    async def get_external_persona(
        self,
        persona_id: UUID
    ) -> Optional[ExternalPersona]:
        """Get external persona by ID (cache-aware)."""
        if self.cache:
            cached = await self.cache.get_persona(str(persona_id))
            if cached:
                return cached
        
        async with self.db.get_session() as session:
            stmt = select(ExternalPersona).where(ExternalPersona.id == persona_id)
            result = await session.execute(stmt)
            persona = result.scalar_one_or_none()
            
            if persona and self.cache:
                await self.cache.set_external(str(persona_id), persona)
            
            return persona
