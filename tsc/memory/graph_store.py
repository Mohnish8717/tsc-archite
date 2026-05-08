"""
GraphStore — Lightweight Stub
===============================

Minimal stub for the deprecated GraphStore. The knowledge graph builder
and legacy persona generator still reference this interface. In v3.0,
Hindsight is the primary memory backbone — this stub provides the minimum
contract to keep the pipeline importable.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GraphStore:
    """Stub graph store — wraps WorldDataBank for legacy compatibility."""

    def __init__(self, world_bank: Optional[Any] = None):
        self._world_bank = world_bank
        logger.debug("GraphStore stub initialized (Hindsight is primary)")

    async def store_graph(self, graph: Any) -> None:
        """Store knowledge graph (no-op in stub)."""
        logger.debug("GraphStore.store_graph: stub no-op")

    async def retrieve_stakeholder_context(
        self, name: str, role: str
    ) -> dict[str, Any]:
        """Retrieve context for a stakeholder from the graph."""
        # In the stub, return empty context — the grounded persona prompt
        # will work with whatever evidence is available
        return {
            "personal_facts": [],
            "org_context": [],
            "constraint_context": [],
        }

    async def retrieve_customer_context(
        self, segment: str, use_case: str
    ) -> dict[str, Any]:
        """Retrieve context for an external customer persona."""
        return {
            "personal_facts": [],
            "org_context": [],
            "constraint_context": [],
        }

    async def query(self, query: str) -> list[dict[str, Any]]:
        """Query the graph store."""
        return []
