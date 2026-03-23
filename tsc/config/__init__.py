"""TSC configuration package."""

from tsc.config.pipeline_config import LLMProvider, PipelineConfig

# Alias for backward compatibility with orchestrator.py
Settings = PipelineConfig
settings = PipelineConfig()

__all__ = ["Settings", "settings", "PipelineConfig", "LLMProvider"]
