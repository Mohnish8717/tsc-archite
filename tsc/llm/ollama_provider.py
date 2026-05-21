"""Ollama local LLM provider (OpenAI compatible)."""

from __future__ import annotations

import logging
from typing import Any

import openai

from tsc.llm.openai_provider import OpenAIClient

logger = logging.getLogger(__name__)


class OllamaClient(OpenAIClient):
    """Ollama local API client leveraging the OpenAI compatibility layer."""

    def __init__(
        self,
        api_key: str = "ollama",
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        **kwargs: Any,
    ):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
