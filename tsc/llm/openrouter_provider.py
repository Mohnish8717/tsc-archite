"""OpenRouter LLM provider — uses OpenAI-compatible API."""

from __future__ import annotations

import logging
from typing import Any

import openai

from tsc.llm.openai_provider import OpenAIClient

logger = logging.getLogger(__name__)


class OpenRouterClient(OpenAIClient):
    """OpenRouter backend — access any model via OpenAI-compatible endpoint."""

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-sonnet-4-20250514",
        **kwargs: Any,
    ):
        super(OpenAIClient, self).__init__(api_key, model, **kwargs)
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://tsc-pipeline.local",
                "X-Title": "TSC v2.0 Pipeline",
            },
        )
