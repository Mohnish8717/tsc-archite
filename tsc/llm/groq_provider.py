"""Groq LLM provider — uses OpenAI-compatible API."""

from __future__ import annotations

import logging
from typing import Any

import openai

from tsc.llm.openai_provider import OpenAIClient

logger = logging.getLogger(__name__)


class GroqClient(OpenAIClient):
    """Groq backend — fast inference via OpenAI-compatible endpoint."""

    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        **kwargs: Any,
    ):
        super().__init__(api_key, model, **kwargs)
        # Re-initialize with custom base URL
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=self.GROQ_BASE_URL,
        )
