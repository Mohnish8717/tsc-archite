"""Google Gemini / Gemma 4 API provider."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import google.generativeai as genai
from tsc.llm.base import LLMClient
from tsc.llm.rate_limiter import get_gemini_bucket

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """Client for Google Gemini / Gemma 4 via AI Studio."""

    def __init__(self, api_key: str, model: str, **kwargs: Any):
        super().__init__(api_key, model, **kwargs)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name=model)
        self._rate_limiter = get_gemini_bucket()

    def _estimate_tokens(self, system_prompt: str, user_prompt: str, max_tokens: int = 0) -> int:
        """Estimate total tokens for a request (input + output)."""
        input_chars = len(system_prompt) + len(user_prompt)
        estimated = (input_chars // 4) + max_tokens
        return max(50, estimated)

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

        # Gemma models served via the Gemini API wrapper ignore system_instruction.
        # Prepend the system prompt directly into the user turn to guarantee compliance.
        if self.model.startswith("gemma"):
            effective_user_prompt = f"{system_prompt}\n\n{user_prompt}"
            model = genai.GenerativeModel(model_name=self.model)
        else:
            effective_user_prompt = user_prompt
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
            )

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Enforce rate limiting
        estimated = self._estimate_tokens(system_prompt, user_prompt, max_tokens)
        await self._rate_limiter.acquire(estimated)

        # Use async API to avoid blocking the asyncio event loop
        response = await model.generate_content_async(
            effective_user_prompt,
            generation_config=generation_config,
        )

        text = response.text
        elapsed = time.time() - start_time
        usage = response.usage_metadata
        self._log_call("analyze", usage.prompt_token_count, usage.candidates_token_count, elapsed)
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

        if self.model.startswith("gemma"):
            effective_user_prompt = f"{system_prompt}\n\n{user_prompt}"
            model = genai.GenerativeModel(model_name=self.model)
        else:
            effective_user_prompt = user_prompt
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
            )

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Enforce rate limiting
        estimated = self._estimate_tokens(system_prompt, user_prompt, max_tokens)
        await self._rate_limiter.acquire(estimated)

        # Use async API to avoid blocking the asyncio event loop
        response = await model.generate_content_async(
            effective_user_prompt,
            generation_config=generation_config,
        )

        text = response.text
        elapsed = time.time() - start_time
        usage = response.usage_metadata
        self._log_call("generate", usage.prompt_token_count, usage.candidates_token_count, elapsed)
        return text
