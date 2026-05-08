"""
HindsightSessionManager — Unified Memory Backbone for v3.0
====================================================================

Single session manager that all pipeline layers share. Replaces the
fragmented approach (WorldDataBank, HindsightOASISManager, HindsightBoardroom
each creating their own clients) with namespaced banks within one session.

Banks:
    world       — Raw documents, company context, interview transcripts
    simulation  — OASIS agent interactions and behavioral insights
    discovery   — Feature discovery analysis and proposed features
    personas    — Generated persona profiles and belief vectors
    debate      — Boardroom debate transcripts and votes
    spec        — Generated PRD and task breakdown
    meta        — Pipeline metadata, timing, provenance chain

Two modes:
    HINDSIGHT — Connected to Hindsight server (Cloud or Local Docker)
    EMBEDDED  — In-memory dict fallback for development without server
"""

import os
import re
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Patch for nested async (AG2 + Hindsight both use asyncio)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass


BANK_NAMES = ["world", "simulation", "discovery", "personas", "debate", "spec", "meta"]


class HindsightSessionManager:
    """Unified Hindsight session for the entire pipeline run.

    Usage:
        session = HindsightSessionManager()
        await session.initialize("run-2026-05-02-001")

        # Any layer can retain data:
        await session.retain("world", "Customer interview transcript...", {"type": "interview"})

        # Any layer can recall data:
        context = await session.recall("world", "What are the top customer complaints?")

        # Cross-bank queries:
        evidence = await session.cross_recall("pricing concerns", ["world", "simulation"])
    """

    def __init__(self, hindsight_url: Optional[str] = None, api_key: Optional[str] = None):
        self.run_id: str = ""
        self._hindsight = None
        self._mode = "EMBEDDED"
        self._lock = threading.RLock()

        # Embedded fallback storage: bank_name -> list of {content, metadata, timestamp}
        self._embedded: Dict[str, List[dict]] = {bank: [] for bank in BANK_NAMES}

        # Try to connect to Hindsight
        url = hindsight_url or os.getenv("HINDSIGHT_URL", "")
        key = api_key or os.getenv("HINDSIGHT_API_KEY", "")
        if url:
            try:
                from hindsight_client import Hindsight
                self._hindsight = Hindsight(base_url=url, api_key=key if key else None)
                self._mode = "HINDSIGHT"
                logger.info(f"🧠 HindsightSession: Connected to {url}")
            except ImportError:
                logger.warning("hindsight_client not installed. Using EMBEDDED fallback.")
            except Exception as e:
                logger.warning(f"Hindsight connection failed ({e}). Using EMBEDDED fallback.")
        else:
            logger.info("🧠 HindsightSession: No HINDSIGHT_URL set. Using EMBEDDED mode.")

    @property
    def is_connected(self) -> bool:
        """Check if Hindsight server is available."""
        return self._mode == "HINDSIGHT" and self._hindsight is not None

    async def initialize(self, run_id: str) -> None:
        """Initialize the session with a unique run ID and create all banks.

        Args:
            run_id: Unique identifier for this pipeline run (e.g., "run-2026-05-02-001")
        """
        self.run_id = run_id
        logger.info(f"🧠 Initializing session '{run_id}' (mode={self._mode})")

        if not self.is_connected:
            logger.info(f"🧠 EMBEDDED mode: {len(BANK_NAMES)} virtual banks ready")
            return

        # Create namespaced banks in Hindsight
        for bank_name in BANK_NAMES:
            bank_id = self._bank_id(bank_name)
            try:
                # Delete any pre-existing bank from a previous run with same ID
                try:
                    await asyncio.to_thread(self._hindsight.delete_bank, bank_id=bank_id)
                except Exception:
                    pass  # Bank didn't exist — fine

                await asyncio.to_thread(
                    self._hindsight.create_bank,
                    bank_id=bank_id,
                    name=f"{bank_name} — {run_id}",
                    background=self._bank_background(bank_name),
                    retain_mission=self._bank_retain_mission(bank_name),
                    enable_observations=True,
                    observations_mission=f"Synthesize patterns and insights from {bank_name} data.",
                )
                logger.debug(f"  ✓ Bank '{bank_id}' created")
            except Exception as e:
                if "already exists" in str(e).lower() or "409" in str(e):
                    logger.debug(f"  ✓ Bank '{bank_id}' already exists — reusing")
                else:
                    logger.warning(f"  ✗ Bank '{bank_id}' creation failed: {e}")

        logger.info(f"🧠 Session '{run_id}': {len(BANK_NAMES)} banks initialized in Hindsight")

    def _bank_id(self, bank_name: str) -> str:
        """Generate the namespaced bank ID."""
        return f"pre-{self.run_id}-{bank_name}"

    async def retain(self, bank: str, content: str, metadata: Optional[dict] = None) -> None:
        """Store content into a specific bank.

        Args:
            bank: Bank name (e.g., "world", "simulation", "debate")
            content: Text content to store
            metadata: Optional metadata dict (e.g., {"type": "interview", "file": "data.txt"})
        """
        if not content or not content.strip():
            return

        if bank not in BANK_NAMES:
            logger.warning(f"Unknown bank '{bank}'. Valid: {BANK_NAMES}")
            return

        record = {
            "content": content[:10000],  # Cap at 10K chars per record
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        # Always store in embedded (as backup and for cross_recall)
        with self._lock:
            self._embedded[bank].append(record)

        # Also store in Hindsight if connected
        if self.is_connected:
            try:
                bank_id = self._bank_id(bank)
                tags = [bank]
                if metadata:
                    tags.extend([f"{k}:{v}" for k, v in metadata.items() if isinstance(v, str)])

                await asyncio.to_thread(
                    self._hindsight.retain,
                    bank_id=bank_id,
                    content=content[:10000],
                    tags=tags[:10],  # Hindsight tag limit
                    timestamp=datetime.now(),
                )
            except Exception as e:
                logger.debug(f"Hindsight retain failed for bank '{bank}': {e}")

    async def recall(self, bank: str, query: str, max_tokens: int = 800) -> str:
        """Retrieve semantically relevant content from a bank.

        Args:
            bank: Bank name to query
            query: Natural language query
            max_tokens: Max tokens in response

        Returns:
            Relevant context as a string
        """
        if bank not in BANK_NAMES:
            return f"Unknown bank '{bank}'."

        # Try Hindsight first
        if self.is_connected:
            try:
                bank_id = self._bank_id(bank)
                result = await asyncio.to_thread(
                    self._hindsight.recall,
                    bank_id=bank_id,
                    query=query,
                    budget="low",
                    max_tokens=max_tokens,
                )
                if result:
                    text = getattr(result, 'answer', getattr(result, 'text', str(result)))
                    return str(text)[:max_tokens * 4]
            except Exception as e:
                logger.debug(f"Hindsight recall failed for bank '{bank}': {e}")

        # Embedded fallback: simple keyword matching
        return self._embedded_recall(bank, query)

    async def reflect(self, bank: str, query: str) -> str:
        """Deep reflection over a bank's contents (slower, more thorough).

        Args:
            bank: Bank name to reflect on
            query: Reflection query

        Returns:
            Synthesized reflection
        """
        if bank not in BANK_NAMES:
            return f"Unknown bank '{bank}'."

        if self.is_connected:
            try:
                bank_id = self._bank_id(bank)
                result = await asyncio.to_thread(
                    self._hindsight.reflect,
                    bank_id=bank_id,
                    query=query,
                    budget="high",
                    include_facts=True,
                )
                if result:
                    return getattr(result, 'answer', getattr(result, 'text', str(result)))
            except Exception as e:
                logger.debug(f"Hindsight reflect failed for bank '{bank}': {e}")

        # Embedded fallback
        return self._embedded_recall(bank, query)

    async def cross_recall(self, query: str, banks: List[str]) -> str:
        """Query multiple banks simultaneously and merge results.

        Args:
            query: Natural language query
            banks: List of bank names to query

        Returns:
            Merged context from all banks
        """
        results = []
        for bank in banks:
            if bank in BANK_NAMES:
                r = await self.recall(bank, query)
                if r and r.strip() and "no documents" not in r.lower() and "unknown bank" not in r.lower():
                    results.append(f"[{bank.upper()}]\n{r}")

        if not results:
            return "No relevant data found across the specified banks."
        return "\n\n".join(results)

    async def get_bank_summary(self, bank: str) -> dict:
        """Get metadata about a bank's contents.

        Returns:
            Dict with record_count and sample content
        """
        with self._lock:
            records = self._embedded.get(bank, [])
            return {
                "bank": bank,
                "mode": self._mode,
                "record_count": len(records),
                "latest_timestamp": records[-1]["timestamp"] if records else None,
            }

    async def get_session_summary(self) -> dict:
        """Get an overview of all banks in this session."""
        summaries = {}
        for bank in BANK_NAMES:
            summaries[bank] = await self.get_bank_summary(bank)
        return {
            "run_id": self.run_id,
            "mode": self._mode,
            "banks": summaries,
            "total_records": sum(s["record_count"] for s in summaries.values()),
        }

    # ─── Embedded Fallback ─────────────────────────────────────────────

    def _embedded_recall(self, bank: str, query: str) -> str:
        """Simple keyword-based recall for embedded mode."""
        with self._lock:
            records = self._embedded.get(bank, [])

        if not records:
            return f"No data available in '{bank}' bank."

        # Score records by keyword overlap
        query_words = set(re.findall(r'\w+', query.lower()))
        scored = []
        for rec in records:
            content = rec["content"].lower()
            content_words = set(re.findall(r'\w+', content))
            overlap = len(query_words & content_words)
            scored.append((overlap, rec))

        # Return top 5 most relevant records
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:5]

        results = []
        for score, rec in top:
            preview = rec["content"][:500]
            meta = rec.get("metadata", {})
            meta_str = f" [{meta.get('type', '')}]" if meta.get('type') else ""
            results.append(f"{meta_str} {preview}")

        return "\n---\n".join(results)

    # ─── Bank Configuration ────────────────────────────────────────────

    @staticmethod
    def _bank_background(bank_name: str) -> str:
        backgrounds = {
            "world": "Repository of raw customer data: interview transcripts, support tickets, usage analytics, and company context documents.",
            "simulation": "OASIS social simulation data: agent interactions, behavioral patterns, product usage feedback from synthetic users.",
            "discovery": "Feature discovery analysis: identified pain points, proposed features, customer evidence citations.",
            "personas": "Generated user and stakeholder personas with psychological profiles, belief vectors, and predicted stances.",
            "debate": "AG2 adversarial boardroom debate: stakeholder positions, votes, challenges, vetoes, and consensus formation.",
            "spec": "Generated product specifications: PRD sections, UI proposals, data model changes, workflow modifications, development tasks.",
            "meta": "Pipeline execution metadata: timing, token usage, provenance chains, run configuration.",
        }
        return backgrounds.get(bank_name, f"Data bank for {bank_name}")

    @staticmethod
    def _bank_retain_mission(bank_name: str) -> str:
        missions = {
            "world": "Extract customer pain points, feature requests, sentiment, product feedback, and competitive intelligence from raw documents.",
            "simulation": "Extract user behavior patterns, feature adoption signals, rejection reasons, and social contagion dynamics from simulation logs.",
            "discovery": "Extract feature proposals, supporting evidence, customer quotes, and prioritization rationale.",
            "personas": "Extract persona traits, domain expertise, predicted behaviors, and belief systems.",
            "debate": "Extract stakeholder positions, commitments, vetoes, challenges, concessions, and evolving consensus.",
            "spec": "Extract development tasks, UI specifications, data model changes, and acceptance criteria.",
            "meta": "Extract pipeline execution metrics, timing data, and configuration parameters.",
        }
        return missions.get(bank_name, f"Extract and organize data for {bank_name}")

    async def cleanup(self) -> int:
        """Delete all banks created during this session.
        
        Called when a NEW simulation starts to reclaim storage.
        Works for both local Docker and cloud Hindsight.
        """
        deleted = 0
        if self.is_connected:
            for bank_name in BANK_NAMES:
                bank_id = self._bank_id(bank_name)
                try:
                    await asyncio.to_thread(self._hindsight.delete_bank, bank_id=bank_id)
                    deleted += 1
                except Exception:
                    pass  # Bank may not exist
        logger.info(f"🧹 Session '{self.run_id}': Cleaned up {deleted} banks")
        return deleted

    def close(self) -> None:
        """Cleanup Hindsight connection."""
        if self._hindsight:
            try:
                self._hindsight.close()
            except Exception:
                pass
            self._hindsight = None
        logger.info(f"🧠 Session '{self.run_id}' closed")
