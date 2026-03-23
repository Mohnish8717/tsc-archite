"""Zep Graph Memory Updater for strategic batching and summarization.

Reduces Zep Cloud credit usage by 98% through 50:1 batching and LLM-based 
summarization into 'Strategic Insights'.
"""

from __future__ import annotations

import logging
from typing import Any, List, Dict
from datetime import datetime

from tsc.llm.base import LLMClient
from tsc.memory.zep_client import ZepMemoryClient

logger = logging.getLogger(__name__)

class ZepGraphMemoryUpdater:
    """Manages buffered ingestion of simulation actions into Zep Graph.
    
    Features:
    - 25-50 action buffer.
    - LLM-based summarization (Strategic Insights).
    - Filter logic for non-meaningful actions (DO_NOTHING).
    - Platform-specific tagging (Twitter, Reddit).
    """

    def __init__(
        self, 
        zep_client: ZepMemoryClient, 
        llm: LLMClient,
        buffer_size: int = 50
    ):
        self._zep = zep_client
        self._llm = llm
        self._buffer_size = buffer_size
        self._buffer: List[Dict[str, Any]] = []
        self._total_actions_processed = 0
        self._total_episodes_sent = 0

    async def add_action(self, action: Dict[str, Any]) -> None:
        """Add an action to the buffer. Triggers sync if buffer full."""
        # 1. Filtering
        action_type = action.get("type", "UNKNOWN")
        if action_type in ("DO_NOTHING", "IDLE", "WAIT"):
            return

        self._buffer.append(action)
        self._total_actions_processed += 1

        if len(self._buffer) >= self._buffer_size:
            await self.flush()

    async def flush(self) -> None:
        """Summarize buffer via LLM and send as a single Zep Episode."""
        if not self._buffer:
            return

        logger.info(
            "Flushing Zep buffer: %d actions -> 1 Strategic Insight", 
            len(self._buffer)
        )

        # 2. Natural Language Compression & Aggregation
        raw_text = self._compress_actions(self._buffer)
        
        # 3. Strategic Summarization (LLM)
        try:
            insight = await self._summarize_batch(raw_text)
            
            # 4. Zep Ingestion (client.graph.add)
            await self._zep.ingest_facts([
                {
                    "fact": insight,
                    "metadata": {
                        "type": "strategic_insight",
                        "action_count": len(self._buffer),
                        "timestamp": datetime.utcnow().isoformat(),
                        "platforms": list(set(a.get("platform", "unknown") for a in self._buffer))
                    }
                }
            ])
            self._total_episodes_sent += 1
            
        except Exception as e:
            logger.error("Failed to summarize/ingest Zep batch: %s", e)
            # Fallback: ingest raw compressed text if LLM fails
            await self._zep.ingest_facts([{"fact": raw_text[:2000], "metadata": {"type": "raw_batch_fallback"}}])

        self._buffer = []

    def _compress_actions(self, actions: List[Dict[str, Any]]) -> str:
        """Convert list of action dicts into a dense newline-separated string."""
        lines = []
        for a in actions:
            agent = a.get("agent_name", f"Agent_{a.get('agent_id', 'unknown')}")
            act = a.get("type", "acted")
            target = a.get("target_id", "content")
            platform = a.get("platform", "web")
            lines.append(f"[{platform}] {agent}: {act} on {target}")
        return "\n".join(lines)

    async def _summarize_batch(self, raw_actions: str) -> str:
        """Use LLM (Nemotron optimization logic) to distill actions into insight."""
        system_prompt = (
            "You are a strategic analyst. Summarize the following batch of user simulation "
            "actions into a single, high-density 'Strategic Insight' that captures the "
            "prevailing sentiment, key blockers, and adoption trends. Be concise (max 3 sentences)."
        )
        user_prompt = f"SIMULATION ACTIONS:\n{raw_actions}\n\nSTRATEGIC INSIGHT:"
        
        # Optimization: Nemotron-style reasoning
        return await self._llm.analyze(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=200
        )

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "actions_processed": self._total_actions_processed,
            "episodes_sent": self._total_episodes_sent,
            "buffer_current": len(self._buffer)
        }
