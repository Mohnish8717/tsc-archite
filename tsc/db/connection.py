"""Production-ready async database connection with cache integration."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

if TYPE_CHECKING:
    from tsc.caching.lru_cache import PersonaCache

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages async database connections with:
    - Connection pooling
    - Session management
    - Cache invalidation on writes
    - Schema initialization
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        cache: Optional[PersonaCache] = None,
    ):
        """
        Initialize database connection.
        
        Args:
            database_url: Database URL (defaults to env var or SQLite)
            cache: Optional PersonaCache for invalidation on writes
        """
        if not database_url:
            database_url = os.getenv(
                "DATABASE_URL",
                "sqlite+aiosqlite:///./tsc_dev.db"
            )
        
        self._cache = cache  # ✅ Cache reference for invalidation
        self._database_url = database_url
        self._initialized = False
        
        # PostgreSQL-specific settings
        is_postgresql = database_url.startswith("postgresql")
        engine_args = {}
        if is_postgresql:
            engine_args = {
                "pool_size": int(os.getenv("DATABASE_POOL_SIZE", "10")),
                "max_overflow": int(os.getenv("DATABASE_MAX_OVERFLOW", "20")),
                "pool_timeout": int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
                "pool_recycle": 3600,  # Recycle connections hourly
            }
        
        self._engine = create_async_engine(
            database_url,
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            **engine_args
        )
        
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
            class_=AsyncSession
        )
        
        db_identifier = database_url.split("@")[-1] if "@" in database_url else database_url
        logger.info(f"DatabaseConnection initialized for: {db_identifier}")

    async def init_schema(self, base: type[DeclarativeBase]) -> None:
        """
        Initialize database schema.
        
        Creates all tables defined in the declarative base.
        Safe to call multiple times.
        
        Args:
            base: SQLAlchemy declarative base
            
        Raises:
            Exception: If schema initialization fails
        """
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(base.metadata.create_all)
            self._initialized = True
            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            if self._cache:
                await self._cache.clear()
            raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provides an async database session via context manager.
        
        Automatically commits on success, rolls back on error.
        
        Usage:
            async with db.get_session() as session:
                result = await session.execute(query)
        
        Yields:
            AsyncSession: Active database session
            
        Raises:
            Exception: If session operations fail (propagated after rollback)
        """
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()  # ✅ Explicit commit
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()

    async def invalidate_persona(self, persona_id: str) -> None:
        """
        Invalidate persona in cache after DB update.
        
        Call this after successfully updating a persona
        in the database to maintain cache consistency.
        
        Args:
            persona_id: ID of persona to invalidate
        """
        if self._cache:
            await self._cache.delete(persona_id)
            logger.debug(f"Invalidated cache for persona: {persona_id}")

    async def invalidate_persona_lists(self, company_id: str = "") -> None:
        """
        Invalidate persona list caches.
        
        Call after persona list changes to maintain consistency.
        
        Args:
            company_id: Prefix to match (empty = invalidate all lists)
        """
        if self._cache:
            await self._cache.invalidate_lists(f"personas:{company_id}")
            logger.debug(f"Invalidated persona lists for company: {company_id}")

    async def close(self) -> None:
        """
        Close all database connections.
        
        Call this on application shutdown.
        """
        await self._engine.dispose()
        logger.info("Database connection closed")

    @property
    def is_initialized(self) -> bool:
        """Check if database schema has been initialized."""
        return self._initialized


# Singleton-like instance management
_db_instance: Optional[DatabaseConnection] = None


def get_db(cache: Optional[PersonaCache] = None) -> DatabaseConnection:
    """
    Get or create the global DatabaseConnection instance.
    
    Args:
        cache: Optional PersonaCache (only used on first call)
        
    Returns:
        Global DatabaseConnection instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection(cache=cache)
    return _db_instance


async def init_db(base: type[DeclarativeBase]) -> DatabaseConnection:
    """
    Initialize database on application startup.
    
    Args:
        base: SQLAlchemy declarative base
        
    Returns:
        Initialized DatabaseConnection instance
    """
    db = get_db()
    await db.init_schema(base)
    return db
