"""Production-ready LRU Cache with async safety and TTL support."""

from __future__ import annotations

import asyncio
import time
import collections
from typing import Any, Optional, TypeVar, Generic

T = TypeVar("T")

class LRUCache(Generic[T]):
    """
    A production-ready LRU Cache with:
    - TTL (Time-To-Live) support
    - Async-safe operations (thread-safe with asyncio.Lock)
    - Proper expiry handling
    - Capacity management
    """
    
    def __init__(self, capacity: int = 100, ttl_seconds: int = 300):
        """
        Initialize LRU Cache.
        
        Args:
            capacity: Maximum number of items in cache
            ttl_seconds: Time-to-live for cached items in seconds
        """
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache = collections.OrderedDict()
        self.expiry = {}
        self._lock = asyncio.Lock()  # ✅ Async-safe locking
        self._hit_count = 0
        self._miss_count = 0

    async def get(self, key: str) -> Optional[T]:
        """
        Get value from cache (async-safe).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self.cache:
                self._miss_count += 1
                return None
            
            # Check expiry
            current_time = time.time()
            if current_time > self.expiry.get(key, 0):
                self._delete_unsafe(key)  # Use unsafe version (already locked)
                self._miss_count += 1
                return None
            
            # Move to end (mark as recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self._hit_count += 1
            return value

    async def set(self, key: str, value: T) -> None:
        """
        Set value in cache (async-safe).
        
        Args:
            key: Cache key
            value: Value to cache
        """
        async with self._lock:
            # Remove if exists
            if key in self.cache:
                self.cache.pop(key)
            # Evict oldest if at capacity
            elif len(self.cache) >= self.capacity:
                oldest_key, _ = self.cache.popitem(last=False)
                self.expiry.pop(oldest_key, None)
            
            # Add to cache
            self.cache[key] = value
            self.expiry[key] = time.time() + self.ttl

    async def delete(self, key: str) -> None:
        """
        Delete key from cache (async-safe).
        
        Args:
            key: Cache key to delete
        """
        async with self._lock:
            self._delete_unsafe(key)

    def _delete_unsafe(self, key: str) -> None:
        """Delete without locking (call only when already locked)."""
        if key in self.cache:
            self.cache.pop(key)
            self.expiry.pop(key, None)

    async def clear(self) -> None:
        """Clear entire cache (async-safe)."""
        async with self._lock:
            self.cache.clear()
            self.expiry.clear()

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total = self._hit_count + self._miss_count
            hit_rate = (self._hit_count / total * 100) if total > 0 else 0
            return {
                "capacity": self.capacity,
                "size": len(self.cache),
                "hits": self._hit_count,
                "misses": self._miss_count,
                "hit_rate": f"{hit_rate:.1f}%",
                "ttl_seconds": self.ttl,
            }


class PersonaCache:
    """
    Specialized cache for personas with multiple LRU caches
    for different persona types.
    
    Features:
    - Separate caches for internal and external personas
    - List query caching with shorter TTL
    - Transaction-aware (invalidation support)
    - Hit/miss statistics
    """
    
    def __init__(self, capacity: int = 200, ttl: int = 600):
        """
        Initialize PersonaCache.
        
        Args:
            capacity: Items per cache
            ttl: Time-to-live in seconds
        """
        self._internal = LRUCache(capacity=capacity, ttl_seconds=ttl)
        self._external = LRUCache(capacity=capacity, ttl_seconds=ttl)
        self._lists = LRUCache(capacity=50, ttl_seconds=300)  # Shorter TTL for lists
        self._lock = asyncio.Lock()  # For coordination

    async def get_persona(self, persona_id: str) -> Optional[Any]:
        """Get persona by ID from either internal or external cache."""
        # Try internal first
        result = await self._internal.get(persona_id)
        if result is not None:
            return result
        
        # Then external
        return await self._external.get(persona_id)

    async def set_internal(self, persona_id: str, persona: Any) -> None:
        """
        Cache internal persona.
        
        NOTE: Only call after DB transaction commits
        to ensure consistency.
        """
        await self._internal.set(persona_id, persona)

    async def set_external(self, persona_id: str, persona: Any) -> None:
        """
        Cache external persona.
        
        NOTE: Only call after DB transaction commits
        to ensure consistency.
        """
        await self._external.set(persona_id, persona)

    async def get_list(self, query_key: str) -> Optional[list[Any]]:
        """Get cached list result for a query."""
        return await self._lists.get(query_key)

    async def set_list(self, query_key: str, persona_list: list[Any]) -> None:
        """
        Cache list query result.
        
        NOTE: Only call after DB transaction commits.
        """
        await self._lists.set(query_key, persona_list)

    async def delete(self, persona_id: str) -> None:
        """
        Delete persona from both caches.
        
        Use this when persona is updated in DB
        to maintain consistency.
        """
        await asyncio.gather(
            self._internal.delete(persona_id),
            self._external.delete(persona_id),
        )

    async def invalidate_lists(self, prefix: str = "") -> None:
        """
        Invalidate list caches matching prefix.
        
        Example: invalidate_lists("company_123")
        removes all list queries for that company.
        """
        async with self._lists._lock:
            keys_to_delete = [k for k in self._lists.cache.keys() if k.startswith(prefix)]
            for key in keys_to_delete:
                self._lists._delete_unsafe(key)

    async def clear(self) -> None:
        """Clear all caches."""
        await asyncio.gather(
            self._internal.clear(),
            self._external.clear(),
            self._lists.clear(),
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        internal_stats = await self._internal.get_stats()
        external_stats = await self._external.get_stats()
        lists_stats = await self._lists.get_stats()
        
        return {
            "internal": internal_stats,
            "external": external_stats,
            "lists": lists_stats,
        }
