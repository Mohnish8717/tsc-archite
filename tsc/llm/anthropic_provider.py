"""Anthropic (Claude) LLM provider."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import anthropic

from tsc.llm.base import LLMClient

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """Claude API backend via the Anthropic SDK."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", **kwargs: Any):
        super().__init__(api_key, model, **kwargs)
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        t0 = time.time()
        messages = [{"role": "user", "content": user_prompt}]

        # Ask for JSON in the system prompt
        sys_suffix = "\n\nYou MUST respond with valid JSON only. No markdown, no explanation."
        if json_schema:
            sys_suffix += f"\n\nJSON Schema:\n{json_schema}"

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt + sys_suffix,
            messages=messages,
        )

        text = response.content[0].text
        elapsed = time.time() - t0
        self._log_call(
            "analyze",
            response.usage.input_tokens,
            response.usage.output_tokens,
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
        t0 = time.time()

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = response.content[0].text
        elapsed = time.time() - t0
        self._log_call(
            "generate",
            response.usage.input_tokens,
            response.usage.output_tokens,
            elapsed,
        )
        return text
