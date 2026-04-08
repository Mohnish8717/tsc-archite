"""Pipeline configuration — single source of truth for all tuning parameters."""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    GOOGLE = "google"


class PipelineConfig(BaseSettings):
    """Centralized configuration for the TSC evaluation pipeline.

    All numeric parameters are on 0–1 scale unless noted otherwise.
    Override via environment variables (prefixed with TSC_) or .env file.
    """

    # ── LLM Provider ────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(default=LLMProvider.GROQ, alias="TSC_LLM_PROVIDER")
    llm_model: str = Field(default="llama-3.1-8b-instant", alias="TSC_LLM_MODEL")

    # API keys
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    zep_api_key: Optional[str] = Field(default=None, alias="ZEP_API_KEY")

    # ── Memory & NLP ────────────────────────────────────────────────
    spacy_model: str = Field(default="en_core_web_sm", alias="TSC_SPACY_MODEL")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="TSC_EMBEDDING_MODEL")

    # ── Rate limiting ────────────────────────────────────────────────
    groq_tpm_limit: int = Field(default=6000, alias="TSC_GROQ_TPM_LIMIT")
    groq_rpm_limit: int = Field(default=14000, alias="TSC_GROQ_RPM_LIMIT")

    # ── Layer 1 — Chunking ───────────────────────────────────────────
    chunk_min_tokens: int = 80
    chunk_max_tokens: int = 4000
    max_chunks: int = Field(default=500, alias="TSC_MAX_CHUNKS")
    chunk_similarity_threshold: float = 0.65

    # ── Layer 3 — Personas ───────────────────────────────────────────
    persona_max_tokens: int = 1200
    persona_max_count: int = 4          # 2 internal + 2 external
    persona_parallel: bool = False

    # ── Layer 4 — Gates ──────────────────────────────────────────────
    gate_parallel: bool = False
    gate_inter_call_delay: float = 2.0  # seconds between sequential gate calls
    gate_fail_threshold: float = Field(default=0.5, alias="TSC_GATE_FAIL_THRESHOLD")
    monte_carlo_agents: int = 300

    # ── Layer 5 — Refinement ─────────────────────────────────────────
    refinement_threshold: float = 0.45
    refinement_selective: bool = True   # use process_failed_only()

    # ── Layer 6 — Debate ─────────────────────────────────────────────
    debate_parallel_rounds: bool = False
    debate_statement_max_tokens: int = 400

    # ── Layer 7 — Spec ───────────────────────────────────────────────
    spec_max_tokens: int = 3000

    # ── Layer 8 — Handoff ────────────────────────────────────────────
    summary_max_tokens: int = 200

    # ── Web ──────────────────────────────────────────────────────────
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
            LLMProvider.GOOGLE: self.google_api_key,
        }
        key = key_map.get(p)
        if not key:
            # Fallback to direct env check if not in Pydantic yet
            env_map = {
                LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
                LLMProvider.OPENAI: "OPENAI_API_KEY",
                LLMProvider.GROQ: "GROQ_API_KEY",
                LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
                LLMProvider.GOOGLE: "GEMINI_API_KEY",
            }
            key = os.getenv(env_map.get(p, ""))

        if not key:
            raise ValueError(f"No API key configured for provider '{p.value}'")
        return key
