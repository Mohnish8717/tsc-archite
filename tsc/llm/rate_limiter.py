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
from typing import Any, Optional

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

        while True:
            async with lock:
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

            # Sleep outside the lock so other tasks can check/refill concurrently
            await asyncio.sleep(wait_secs)


# ── Module-level singletons ──────────────────────────────────────────

_groq_bucket: Optional[TokenBucket] = None
_gemini_bucket: Optional[TokenBucket] = None


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


def get_gemini_bucket(
    tpm_limit: int | None = None,
    rpm_limit: int | None = None,
) -> TokenBucket:
    """Get or create the singleton Gemini rate limiter.

    Reads limits from environment variables if not provided:
      TSC_GEMINI_TPM_LIMIT (default: 1000000)
      TSC_GEMINI_RPM_LIMIT (default: 12)  # Safely under the 15 RPM free tier
    """
    global _gemini_bucket

    if _gemini_bucket is None:
        tpm = tpm_limit or int(os.getenv("TSC_GEMINI_TPM_LIMIT", "1000000"))
        # Sync with GEMINI_FREE_RPM to ensure matching global rate limiting
        env_rpm = os.getenv("TSC_GEMINI_RPM_LIMIT") or os.getenv("GEMINI_FREE_RPM")
        rpm = rpm_limit or int(env_rpm) if env_rpm else 12
        _gemini_bucket = TokenBucket(tpm_limit=tpm, rpm_limit=rpm)
        logger.info("Gemini rate limiter initialized: %d TPM, %d RPM", tpm, rpm)

    return _gemini_bucket


def reset_groq_bucket() -> None:
    """Reset singletons (for testing)."""
    global _groq_bucket, _gemini_bucket
    _groq_bucket = None
    _gemini_bucket = None


def patch_openai_globally() -> None:
    """Monkey-patch openai's synchronous and asynchronous chat completions
    to globally enforce our custom rate limiters. This ensures AutoGen (ag2)
    and all sub-systems are perfectly rate-limited without bypasses.
    """
    try:
        import openai
        import openai.resources.chat.completions as chat_completions
        
        # Avoid double patching
        if hasattr(chat_completions.Completions.create, "_is_patched"):
            return
            
        orig_sync_create = chat_completions.Completions.create
        orig_async_create = chat_completions.AsyncCompletions.create
        
        def patched_sync_create(self, *args, **kwargs):
            model = kwargs.get("model", "")
            base_url = str(getattr(self._client, "base_url", ""))
            is_gemini = "generativelanguage" in base_url or "gemini" in model.lower() or "gemma" in model.lower()
            is_nvidia = "nvidia" in base_url or os.getenv("TSC_LLM_PROVIDER") == "nvidia"
            
            if is_gemini:
                rpm_limit = int(os.getenv("GEMINI_FREE_RPM", "10"))
                delay = max(4.0, 60.0 / max(1, rpm_limit))
                logger.info(f"⏳ Globally rate-limiting synchronous Gemini call ({model}) with a {delay:.1f}s delay...")
                time.sleep(delay)
            elif is_nvidia:
                delay = 2.0
                time.sleep(delay)
            elif "groq" in base_url or "llama" in model.lower():
                time.sleep(0.5)
                
            # Robust retry loop to prevent synchronous RateLimitErrors (e.g. from Autogen/AG2 debate)
            import openai
            max_sync_retries = 5
            backoff = 2.0
            for attempt in range(max_sync_retries):
                try:
                    return orig_sync_create(self, *args, **kwargs)
                except openai.RateLimitError as e:
                    if attempt == max_sync_retries - 1:
                        raise e
                    logger.warning(
                        f"⏳ Sync Rate Limit hit for {model} (attempt {attempt+1}/{max_sync_retries}). "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff = min(60.0, backoff * 2.0)
            
        async def patched_async_create(self, *args, **kwargs):
            model = kwargs.get("model", "")
            base_url = str(getattr(self._client, "base_url", ""))
            is_gemini = "generativelanguage" in base_url or "gemini" in model.lower() or "gemma" in model.lower()
            is_nvidia = "nvidia" in base_url or os.getenv("TSC_LLM_PROVIDER") == "nvidia"
            
            async def _execute_with_retry():
                import openai
                max_async_retries = 8
                backoff = 2.0
                for attempt in range(max_async_retries):
                    try:
                        return await orig_async_create(self, *args, **kwargs)
                    except openai.RateLimitError as e:
                        if attempt == max_async_retries - 1:
                            raise e
                        logger.warning(
                            f"⏳ Async Rate Limit hit for {model} (attempt {attempt+1}/{max_async_retries}). "
                            f"Retrying in {backoff:.1f}s..."
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(60.0, backoff * 1.5)
            
            if is_gemini:
                bucket = get_gemini_bucket()
                messages = kwargs.get("messages", [])
                input_chars = sum(len(str(m.get("content", ""))) for m in messages)
                estimated = (input_chars // 4) + kwargs.get("max_tokens", 1000)
                
                logger.debug(f"⏳ Globally acquiring capacity for async Gemini call ({model}, estimated {estimated} tokens)...")
                await bucket.acquire(estimated)
                
                # Space out async Gemini calls through the leaky bucket singleton
                leaky = get_leaky_bucket()
                return await leaky.call(_execute_with_retry())
            elif is_nvidia:
                # NVIDIA NIM limits are often strict; use a leaky bucket to space calls out
                global _nvidia_leaky
                if '_nvidia_leaky' not in globals():
                    _nvidia_leaky = LeakyBucketQueue(rpm=20) # 20 RPM for NVIDIA NIM
                    asyncio.create_task(_nvidia_leaky.start())
                return await _nvidia_leaky.call(_execute_with_retry())
            elif "groq" in base_url or "llama" in model.lower():
                bucket = get_groq_bucket()
                messages = kwargs.get("messages", [])
                input_chars = sum(len(str(m.get("content", ""))) for m in messages)
                estimated = (input_chars // 4) + kwargs.get("max_tokens", 1000)
                await bucket.acquire(estimated)
                
            return await _execute_with_retry()
            
        patched_sync_create._is_patched = True
        patched_async_create._is_patched = True
        
        chat_completions.Completions.create = patched_sync_create
        chat_completions.AsyncCompletions.create = patched_async_create
        logger.info("🛡️ Globally monkey-patched OpenAI chat completions for rate-limiting protection.")
    except Exception as e:
        logger.warning(f"Failed to apply global OpenAI completions patch: {e}")

# Run the patch on import immediately
patch_openai_globally()


# ── Leaky Bucket Queue ───────────────────────────────────────────────────────
# A disciplined FIFO queue that drains one LLM coroutine at a time,
# spaced exactly (60 / RPM) seconds apart.  Unlike asyncio.gather, this
# guarantees no burst even when many callers are waiting simultaneously.

class LeakyBucketQueue:
    """FIFO leaky-bucket queue for LLM coroutines.

    Usage:
        bucket = LeakyBucketQueue(rpm=10)
        await bucket.start()
        result = await bucket.call(my_llm_coro())
        await bucket.stop()

    The drain loop runs as a background task and processes one enqueued
    coroutine per interval, regardless of how many callers are waiting.
    """

    def __init__(self, rpm: int = 10) -> None:
        self.rpm = max(1, rpm)
        self.interval = 60.0 / self.rpm          # seconds between calls
        self._queue: asyncio.Queue = asyncio.Queue()
        self._drain_task: Optional[asyncio.Task] = None
        self._running = False
        logger.info(
            "LeakyBucketQueue created: %d RPM → %.2fs between calls", rpm, self.interval
        )

    async def start(self) -> None:
        """Start the background drain loop."""
        if self._running:
            return
        self._running = True
        self._drain_task = asyncio.ensure_future(self._drain_loop())
        logger.info("LeakyBucketQueue drain loop started")

    async def stop(self) -> None:
        """Gracefully stop the drain loop."""
        self._running = False
        if self._drain_task and not self._drain_task.done():
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
        logger.info("LeakyBucketQueue drain loop stopped")

    async def _drain_loop(self) -> None:
        """Single consumer — pops one coroutine per interval and dispatches it.

        Each coroutine is given a hard 300-second timeout. We dispatch the
        coroutine concurrently as a background task, and then sleep for exactly
        the interval to ensure we never burst API limits while still allowing
        concurrent LLM executions.
        """
        while self._running:
            try:
                coro, fut = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if fut.cancelled():
                self._queue.task_done()
                continue

            async def _execute(c, f):
                try:
                    result = await asyncio.wait_for(c, timeout=300.0)
                    if not f.done():
                        f.set_result(result)
                except asyncio.TimeoutError as exc:
                    logger.warning("LeakyBucketQueue: coroutine timed out after 300s")
                    if not f.done():
                        f.set_exception(exc)
                except Exception as exc:  # noqa: BLE001
                    if not f.done():
                        f.set_exception(exc)
                finally:
                    self._queue.task_done()

            # Dispatch the API call concurrently so the queue can keep draining
            asyncio.create_task(_execute(coro, fut))

            # Enforce exact minimum spacing between request dispatches
            await asyncio.sleep(self.interval)

    async def call(self, coro) -> Any:
        """Enqueue a coroutine and await its result through the leaky bucket.

        Args:
            coro: An awaitable (coroutine) to run through the rate limiter.

        Returns:
            Whatever the coroutine returns.
        """
        if not self._running:
            await self.start()

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put((coro, fut))
        return await fut


# ── Module-level LeakyBucketQueue singleton ─────────────────────────────────

_leaky_bucket: Optional["LeakyBucketQueue"] = None


def get_leaky_bucket(rpm: int | None = None) -> "LeakyBucketQueue":
    """Get or create the singleton LeakyBucketQueue.

    Reads RPM from environment:
      GEMINI_FREE_RPM (default: 10)
    """
    global _leaky_bucket
    if _leaky_bucket is None:
        effective_rpm = rpm or int(os.getenv("GEMINI_FREE_RPM", "10"))
        _leaky_bucket = LeakyBucketQueue(rpm=effective_rpm)
        logger.info("LeakyBucketQueue singleton created: %d RPM", effective_rpm)
    return _leaky_bucket


def reset_leaky_bucket() -> None:
    """Reset singleton (for testing)."""
    global _leaky_bucket
    _leaky_bucket = None
