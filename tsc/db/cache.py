from __future__ import annotations

import time
import collections
from typing import Any, Optional, TypeVar, Generic

T = TypeVar("T")

class LRUCache(Generic[T]):
    """A Simple LRU Cache with TTL support."""
    
    def __init__(self, capacity: int = 100, ttl_seconds: int = 300):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache = collections.OrderedDict()
        self.expiry = {}

    def get(self, key: str) -> Optional[T]:
        if key not in self.cache:
            return None
            
        # Check expiry
        if time.time() > self.expiry.get(key, 0):
            self.delete(key)
            return None
            
        # Move to end (most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def set(self, key: str, value: T):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Remove oldest (least recently used)
            oldest_key, _ = self.cache.popitem(last=False)
            self.expiry.pop(oldest_key, None)
            
        self.cache[key] = value
        self.expiry[key] = time.time() + self.ttl

    def delete(self, key: str):
        if key in self.cache:
            self.cache.pop(key)
            self.expiry.pop(key, None)

    def clear(self):
        self.cache.clear()
        self.expiry.clear()

class PersonaCache:
    """Specialized cache for personas."""
    
    def __init__(self, capacity: int = 200, ttl: int = 600):
        self._internal = LRUCache(capacity=capacity, ttl_seconds=ttl)
        self._external = LRUCache(capacity=capacity, ttl_seconds=ttl)
        self._lists = LRUCache(capacity=50, ttl_seconds=300) # For list queries

    def get_persona(self, persona_id: str) -> Optional[Any]:
        return self._internal.get(persona_id) or self._external.get(persona_id)

    def set_internal(self, persona_id: str, persona: Any):
        self._internal.set(persona_id, persona)

    def set_external(self, persona_id: str, persona: Any):
        self._external.set(persona_id, persona)

    def get_list(self, query_key: str) -> Optional[list[Any]]:
        return self._lists.get(query_key)

    def set_list(self, query_key: str, persona_list: list[Any]):
        self._lists.set(query_key, persona_list)
