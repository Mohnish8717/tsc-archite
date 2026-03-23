"""OpenAI LLM provider."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import openai

from tsc.llm.base import LLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """OpenAI API backend (GPT-4o, etc.)."""

    def __init__(self, api_key: str, model: str = "gpt-4o", **kwargs: Any):
        super().__init__(api_key, model, **kwargs)
        self._client = openai.AsyncOpenAI(api_key=api_key)

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

        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = response.choices[0].message.content or "{}"
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
        max_tokens: int = 4000,
    ) -> str:
        t0 = time.time()

        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = response.choices[0].message.content or ""
        elapsed = time.time() - t0
        usage = response.usage
        self._log_call(
            "generate",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            elapsed,
        )
        return text
