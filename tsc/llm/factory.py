"""Provider factory — runtime LLM provider selection."""

from __future__ import annotations

from typing import Optional

from tsc.config import LLMProvider, Settings
from tsc.llm.base import LLMClient


def create_llm_client(
    provider: Optional[LLMProvider] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> LLMClient:
    """Create an LLM client for the specified (or configured) provider.

    Args:
        provider: Override provider (default: from settings).
        model: Override model name (default: from settings).
        api_key: Override API key (default: from settings).
        settings: Settings instance (default: global singleton).

    Returns:
        Configured LLMClient instance.
    """
    from tsc.config import settings as default_settings

    cfg = settings or default_settings
    p = provider or cfg.llm_provider
    m = model or cfg.llm_model
    k = api_key or cfg.get_api_key(p)

    import os
    proxy_url = os.environ.get("LITELLM_PROXY_URL")
    # Do not intercept if the provider is explicitly local Ollama (unless routed via LiteLLM)
    if proxy_url and p != LLMProvider.OLLAMA:
        from tsc.llm.openai_provider import OpenAIClient
        # Globally intercept all LLM requests if a local proxy is defined
        return OpenAIClient(api_key=k or "litellm-dummy-key", model=m, base_url=proxy_url)

    if p == LLMProvider.ANTHROPIC:
        from tsc.llm.anthropic_provider import AnthropicClient
        return AnthropicClient(api_key=k, model=m)

    elif p == LLMProvider.OPENAI:
        from tsc.llm.openai_provider import OpenAIClient
        return OpenAIClient(api_key=k, model=m)

    elif p == LLMProvider.GROQ:
        from tsc.llm.groq_provider import GroqClient
        return GroqClient(api_key=k, model=m)

    elif p == LLMProvider.OPENROUTER:
        from tsc.llm.openrouter_provider import OpenRouterClient
        return OpenRouterClient(api_key=k, model=m)

    elif p == LLMProvider.GOOGLE:
        from tsc.llm.gemini_provider import GeminiClient
        return GeminiClient(api_key=k, model=m)

    elif p == LLMProvider.OLLAMA:
        from tsc.llm.ollama_provider import OllamaClient
        return OllamaClient(api_key=k or "ollama", model=m or "llama3.2")

    elif p == LLMProvider.NVIDIA:
        from tsc.llm.nvidia_provider import NvidiaClient
        return NvidiaClient(api_key=k, model=m)

    else:
        raise ValueError(f"Unknown LLM provider: {p}")
