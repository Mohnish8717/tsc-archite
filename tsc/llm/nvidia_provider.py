"""NVIDIA NIM LLM provider — uses OpenAI-compatible API."""

from __future__ import annotations

import logging
from typing import Any

import openai

from tsc.llm.openai_provider import OpenAIClient

logger = logging.getLogger(__name__)


class NvidiaClient(OpenAIClient):
    """NVIDIA NIM backend — inference via OpenAI-compatible endpoint."""

    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemma-2-9b-it",
        **kwargs: Any,
    ):
        logger.info(f"INIT NvidiaClient for model: {model}")
        super().__init__(api_key, model, **kwargs)
        # Re-initialize with custom base URL
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=self.NVIDIA_BASE_URL,
            timeout=300.0,
        )
