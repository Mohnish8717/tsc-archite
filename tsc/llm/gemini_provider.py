"""Google Gemini / Gemma 4 API provider — using the new google-genai SDK.

Migrated from the deprecated `google.generativeai` package to `google.genai`
to fix HTTP 500 errors caused by the old SDK sending a legacy request schema
that newer models (Gemma 4) reject.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from google import genai
from google.genai import types as genai_types

from tsc.llm.base import LLMClient
from tsc.llm.rate_limiter import get_gemini_bucket, get_leaky_bucket

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "gemma-4-26b-a4b-it"


class GeminiClient(LLMClient):
    """Client for Google Gemini / Gemma 4 via AI Studio (new google-genai SDK)."""

    def __init__(self, api_key: str, model: str, **kwargs: Any):
        super().__init__(api_key, model, **kwargs)
        self._client = genai.Client(api_key=api_key)
        self._rate_limiter = get_gemini_bucket()

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_model(name: str) -> str:
        """Ensure model name has the 'models/' prefix the API expects."""
        if not name.startswith("models/"):
            return f"models/{name}"
        return name

    def _estimate_tokens(self, system_prompt: str, user_prompt: str, max_tokens: int = 0) -> int:
        """Rough token estimate for rate-limiter pre-accounting."""
        input_chars = len(system_prompt) + len(user_prompt)
        estimated = (input_chars // 4) + max_tokens
        return max(50, estimated)

    def _build_config(
        self,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        *,
        is_gemma_legacy: bool = False,
    ) -> genai_types.GenerateContentConfig:
        """Build a GenerateContentConfig for the new SDK.

        For legacy (pre-Gemma 4) Gemma models system_instruction is not
        supported, so we omit it here and prepend it to user_prompt instead
        (handled by the caller).
        """
        cfg_kwargs: dict[str, Any] = {
            "temperature": temperature,
            # Disable Automatic Function Calling — we never send tools,
            # so AFC just adds an unnecessary round-trip and log noise.
            "automatic_function_calling": genai_types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        }
        if not is_gemma_legacy:
            cfg_kwargs["system_instruction"] = system_prompt
        return genai_types.GenerateContentConfig(**cfg_kwargs)

    # ── core retry/fallback logic ────────────────────────────────────────

    async def _call_with_retry(
        self,
        model_name: str,
        contents: str,
        config: genai_types.GenerateContentConfig,
        max_retries: int = 5,
        base_backoff: float = 4.0,
        timeout: float = 300.0,
    ) -> Any | None:
        """Try *model_name* up to *max_retries* times.

        Returns the API response on success, or ``None`` if all retries
        are exhausted (caller should fall back to the next model).
        """
        leaky = get_leaky_bucket()

        for attempt in range(max_retries):
            try:
                coro = self._client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                response = await asyncio.wait_for(leaky.call(coro), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                logger.warning(
                    "⚠️ Model %s timed out after %.0fs (attempt %d/%d).",
                    model_name, timeout, attempt + 1, max_retries,
                )
            except Exception as e:
                err_str = str(e).lower()
                is_retryable = any(
                    kw in err_str
                    for kw in ["500", "internal error", "429", "resource",
                               "exhausted", "quota", "limit", "503",
                               "unavailable", "overloaded"]
                )
                if is_retryable and attempt < max_retries - 1:
                    sleep_time = base_backoff * (2 ** attempt)
                    logger.warning(
                        "⚠️ Model %s call failed (%s). "
                        "Retry %d/%d in %.1fs…",
                        model_name, e, attempt + 1, max_retries, sleep_time,
                    )
                    await asyncio.sleep(sleep_time)
                elif is_retryable:
                    logger.warning(
                        "⚠️ All %d retries exhausted for %s (%s).",
                        max_retries, model_name, e,
                    )
                else:
                    raise  # non-retryable → surface immediately

        return None  # all retries exhausted

    # ── public API ───────────────────────────────────────────────────────

    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        """Get structured JSON output from Gemini/Gemma."""
        start_time = time.time()

        is_gemma = "gemma" in self.model.lower()
        is_gemma4 = "gemma-4" in self.model.lower() or "gemma4" in self.model.lower()
        is_legacy_gemma = is_gemma and not is_gemma4

        model_name = self._normalize_model(self.model)

        # For legacy Gemma, merge system prompt into user prompt
        if is_legacy_gemma:
            effective_user_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            effective_user_prompt = user_prompt

        config = self._build_config(
            system_prompt, temperature, max_tokens,
            is_gemma_legacy=is_legacy_gemma,
        )

        # Rate-limiter pre-accounting
        estimated = self._estimate_tokens(system_prompt, user_prompt, max_tokens)
        await self._rate_limiter.acquire(estimated)

        # ── Phase 1: Primary model ───────────────────────────────────────
        response = await self._call_with_retry(
            model_name, effective_user_prompt, config,
        )

        # ── Phase 2: Fallback to FALLBACK_MODEL ─────────────────────────
        if response is None:
            fallback_name = self._normalize_model(FALLBACK_MODEL)
            logger.info("🤖 Executing dynamic fallback to %s…", fallback_name)

            fallback_config = self._build_config(
                system_prompt, temperature, max_tokens,
                is_gemma_legacy=False,
            )
            response = await self._call_with_retry(
                fallback_name, user_prompt, fallback_config,
                max_retries=2,
            )

        # ── Phase 3: Ultimate last-resort → Groq Llama ──────────────────
        if response is None:
            logger.info(
                "🤖 Executing ultimate last-resort fallback to "
                "Groq Llama (llama-3.3-70b-versatile)…"
            )
            import os
            from tsc.llm.groq_provider import GroqClient

            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                try:
                    groq_client = GroqClient(
                        api_key=groq_key, model="llama-3.3-70b-versatile"
                    )
                    result = await groq_client.analyze(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        json_schema=json_schema,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return result
                except Exception as groq_exc:
                    logger.error("❌ Ultimate Groq fallback failed: %s", groq_exc)
                    raise
            else:
                logger.error(
                    "❌ Groq fallback requested but GROQ_API_KEY is not set."
                )
                raise RuntimeError(
                    "All models timed out/failed and Groq API key is missing."
                )

        text = response.text
        elapsed = time.time() - start_time
        usage = response.usage_metadata
        self._log_call(
            "analyze",
            usage.prompt_token_count or 0,
            usage.candidates_token_count or 0,
            elapsed,
        )
        return self._parse_json_response(text)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """Get free-form text output from Gemini/Gemma."""
        start_time = time.time()

        is_gemma = "gemma" in self.model.lower()
        is_gemma4 = "gemma-4" in self.model.lower() or "gemma4" in self.model.lower()
        is_legacy_gemma = is_gemma and not is_gemma4

        model_name = self._normalize_model(self.model)

        if is_legacy_gemma:
            effective_user_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            effective_user_prompt = user_prompt

        config = self._build_config(
            system_prompt, temperature, max_tokens,
            is_gemma_legacy=is_legacy_gemma,
        )

        # Rate-limiter pre-accounting
        estimated = self._estimate_tokens(system_prompt, user_prompt, max_tokens)
        await self._rate_limiter.acquire(estimated)

        # ── Phase 1: Primary model ───────────────────────────────────────
        response = await self._call_with_retry(
            model_name, effective_user_prompt, config,
        )

        # ── Phase 2: Fallback to FALLBACK_MODEL ─────────────────────────
        if response is None:
            fallback_name = self._normalize_model(FALLBACK_MODEL)
            logger.info("🤖 Executing dynamic fallback to %s…", fallback_name)

            fallback_config = self._build_config(
                system_prompt, temperature, max_tokens,
                is_gemma_legacy=False,
            )
            response = await self._call_with_retry(
                fallback_name, user_prompt, fallback_config,
                max_retries=2,
            )

        # ── Phase 3: Ultimate last-resort → Groq Llama ──────────────────
        if response is None:
            logger.info(
                "🤖 Executing ultimate last-resort fallback to "
                "Groq Llama (llama-3.3-70b-versatile)…"
            )
            import os
            from tsc.llm.groq_provider import GroqClient

            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                try:
                    groq_client = GroqClient(
                        api_key=groq_key, model="llama-3.3-70b-versatile"
                    )
                    result = await groq_client.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return result
                except Exception as groq_exc:
                    logger.error("❌ Ultimate Groq fallback failed: %s", groq_exc)
                    raise
            else:
                logger.error(
                    "❌ Groq fallback requested but GROQ_API_KEY is not set."
                )
                raise RuntimeError(
                    "All models timed out/failed and Groq API key is missing."
                )

        text = response.text
        elapsed = time.time() - start_time
        usage = response.usage_metadata
        self._log_call(
            "generate",
            usage.prompt_token_count or 0,
            usage.candidates_token_count or 0,
            elapsed,
        )
        return text
