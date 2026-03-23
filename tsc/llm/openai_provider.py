"""OpenAI LLM provider."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import openai

from tsc.llm.base import LLMClient
from tsc.llm.rate_limiter import get_groq_bucket

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """OpenAI API backend (GPT-4o, etc.)."""

    def __init__(self, api_key: str, model: str = "gpt-4o", **kwargs: Any):
        super().__init__(api_key, model, **kwargs)
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._rate_limiter = get_groq_bucket()

    def _estimate_tokens(self, messages: list[dict[str, str]], max_tokens: int = 0) -> int:
        """Estimate total tokens for a request (input + output)."""
        input_chars = sum(len(str(m.get("content", ""))) for m in messages)
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
        t0 = time.time()

        sys_content = system_prompt + "\n\nYou MUST respond with valid JSON only."
        if json_schema:
            sys_content += f"\n\nJSON Schema:\n{json_schema}"

        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_prompt},
        ]

        # Rate limit
        estimated = self._estimate_tokens(messages, max_tokens)
        await self._rate_limiter.acquire(estimated)

        logger.debug(f"\n[LLM PROMPT - analyze]\nSYSTEM: {sys_content}\nUSER: {user_prompt}\n" + "="*40)

        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=messages,
        )

        text = response.choices[0].message.content or "{}"
        logger.debug(f"\n[LLM RESPONSE - analyze]\n{text}\n" + "="*40)

        elapsed = time.time() - t0
        usage = response.usage
        self._log_call(
            "analyze",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            elapsed,
        )
        return self._parse_json_response(text)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        # model kwarg intentionally removed — use self.model (set at construction)
    ) -> str:
        t0 = time.time()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Rate limit
        estimated = self._estimate_tokens(messages, max_tokens)
        await self._rate_limiter.acquire(estimated)

        logger.debug(f"\n[LLM PROMPT - generate]\nSYSTEM: {system_prompt}\nUSER: {user_prompt}\n" + "="*40)

        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )

        text = response.choices[0].message.content or ""
        logger.debug(f"\n[LLM RESPONSE - generate]\n{text}\n" + "="*40)

        elapsed = time.time() - t0
        usage = response.usage
        self._log_call(
            "generate",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            elapsed,
        )
        return text
