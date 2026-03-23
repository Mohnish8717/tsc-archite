"""TSC v2.0 configuration via environment variables and .env file."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    OPENROUTER = "openrouter"


class Settings(BaseSettings):
    """Global settings loaded from environment / .env file."""

    # --- LLM Provider ---
    llm_provider: LLMProvider = Field(
        default=LLMProvider.ANTHROPIC,
        alias="TSC_LLM_PROVIDER",
        description="LLM provider to use",
    )
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        alias="TSC_LLM_MODEL",
        description="Model name (provider-specific)",
    )

    # API keys
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")

    # --- Memory (Zep) ---
    zep_api_key: Optional[str] = Field(default=None, alias="ZEP_API_KEY")

    # --- NLP ---
    spacy_model: str = Field(default="en_core_web_sm", alias="TSC_SPACY_MODEL")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="TSC_EMBEDDING_MODEL",
    )

    # --- Pipeline ---
    max_chunks: int = Field(default=500, alias="TSC_MAX_CHUNKS")
    chunk_similarity_threshold: float = Field(
        default=0.8, alias="TSC_CHUNK_SIMILARITY_THRESHOLD"
    )
    gate_fail_threshold: float = Field(default=0.5, alias="TSC_GATE_FAIL_THRESHOLD")
    monte_carlo_simulations: int = Field(
        default=3000, alias="TSC_MONTE_CARLO_SIMULATIONS"
    )

    # --- Web ---
    web_host: str = Field(default="0.0.0.0", alias="TSC_WEB_HOST")
    web_port: int = Field(default=8000, alias="TSC_WEB_PORT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    def get_api_key(self, provider: Optional[LLMProvider] = None) -> str:
        """Return the API key for the given (or configured) provider."""
        p = provider or self.llm_provider
        key_map = {
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.OPENROUTER: self.openrouter_api_key,
        }
        key = key_map.get(p)
        if not key:
            raise ValueError(
                f"No API key configured for provider '{p.value}'. "
                f"Set the corresponding environment variable."
            )
        return key


# Singleton – import this everywhere
settings = Settings()
