"""Token-bucket rate limiter for Groq (and other TPM-limited) LLM providers.

Implements a dual token + request bucket with proportional refill.
Thread-safe via asyncio.Lock for use in concurrent coroutines.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """Dual token + request rate limiter with proportional refill."""

    tpm_limit: int = 4000       # 33% safety margin under Groq free-tier 6,000
    rpm_limit: int = 20         # 33% safety margin under Groq free-tier 30

    refill_interval: float = 60.0  # seconds

    # Internal state (not constructor args)
    _tokens: float = field(init=False, default=0.0)
    _requests: float = field(init=False, default=0.0)
    _last_refill: float = field(init=False, default=0.0)
    _lock: Optional[asyncio.Lock] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._tokens = float(self.tpm_limit)
        self._requests = float(self.rpm_limit)
        self._last_refill = time.monotonic()
        # Lock is created lazily to avoid event-loop issues
        self._lock = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazily create the asyncio.Lock inside the running event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _refill(self) -> None:
        """Proportionally refill tokens and requests based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill

        # Guard against clock going backward (NTP adjustment)
        if elapsed < 0:
            elapsed = 0.0
            self._last_refill = now
            return

        fraction = elapsed / self.refill_interval
        self._tokens = min(self.tpm_limit, self._tokens + self.tpm_limit * fraction)
        self._requests = min(self.rpm_limit, self._requests + self.rpm_limit * fraction)
        self._last_refill = now

    async def acquire(self, estimated_tokens: int, timeout: float = 120.0) -> None:
        """Wait until enough tokens and request capacity are available.

        Args:
            estimated_tokens: Estimated total tokens for this call.
            timeout: Maximum seconds to wait before force-proceeding.
        """
        # Floor at 1 to avoid acquire(0) issues
        if estimated_tokens <= 0:
            logger.debug("acquire() called with %d tokens — treating as 1", estimated_tokens)
            estimated_tokens = 1

        if timeout <= 0:
            logger.warning("acquire() timeout=0 — force-proceeding immediately")
            return

        deadline = time.monotonic() + timeout
        lock = self._get_lock()

        async with lock:
            while True:
                self._refill()

                if self._tokens >= estimated_tokens and self._requests >= 1:
                    self._tokens -= estimated_tokens
                    self._requests -= 1
                    logger.debug(
                        "Rate limiter: acquired %d tokens (remaining: %.0f tokens, %.0f requests)",
                        estimated_tokens, self._tokens, self._requests,
                    )
                    return

                # Calculate wait time
                token_wait = 0.0
                if self._tokens < estimated_tokens:
                    deficit = estimated_tokens - self._tokens
                    token_wait = (deficit / self.tpm_limit) * self.refill_interval

                request_wait = 0.0
                if self._requests < 1:
                    request_wait = (1.0 / self.rpm_limit) * self.refill_interval

                wait_secs = min(max(token_wait, request_wait) + 0.1, 15.0)

                # Check if we'd exceed the deadline
                if time.monotonic() + wait_secs > deadline:
                    logger.warning(
                        "Rate limiter: timeout approaching — force-proceeding "
                        "(tokens=%.0f, requests=%.0f, needed=%d)",
                        self._tokens, self._requests, estimated_tokens,
                    )
                    # Deduct what we can
                    self._tokens = max(0, self._tokens - estimated_tokens)
                    self._requests = max(0, self._requests - 1)
                    return

                logger.debug(
                    "Rate limiter: waiting %.1fs (need %d tokens, have %.0f)",
                    wait_secs, estimated_tokens, self._tokens,
                )
                # Release the lock during sleep so other coroutines can check
                # Actually, we keep the lock to serialize — asyncio.sleep yields
                # to the event loop but no other acquire() can proceed
                await asyncio.sleep(wait_secs)


# ── Module-level singleton ──────────────────────────────────────────

_groq_bucket: Optional[TokenBucket] = None


def get_groq_bucket(
    tpm_limit: int | None = None,
    rpm_limit: int | None = None,
) -> TokenBucket:
    """Get or create the singleton Groq rate limiter.

    Reads limits from environment variables if not provided:
      TSC_GROQ_TPM_LIMIT (default: 5500)
      TSC_GROQ_RPM_LIMIT (default: 14000)
    """
    global _groq_bucket

    if _groq_bucket is None:
        tpm = tpm_limit or int(os.getenv("TSC_GROQ_TPM_LIMIT", "5500"))
        rpm = rpm_limit or int(os.getenv("TSC_GROQ_RPM_LIMIT", "14000"))
        _groq_bucket = TokenBucket(tpm_limit=tpm, rpm_limit=rpm)
        logger.info("Groq rate limiter initialized: %d TPM, %d RPM", tpm, rpm)

    return _groq_bucket


def reset_groq_bucket() -> None:
    """Reset singleton (for testing)."""
    global _groq_bucket
    _groq_bucket = None
