"""Google Gemini / Gemma 4 API provider."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import google.generativeai as genai
from tsc.llm.base import LLMClient

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """Client for Google Gemini / Gemma 4 via AI Studio."""

    def __init__(self, api_key: str, model: str, **kwargs: Any):
        super().__init__(api_key, model, **kwargs)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name=model)

    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        """Get structured JSON output from Gemini."""
        start_time = time.time()
        
        # Merge prompts for Gemini (or use system_instruction if supported in SDK version)
        # Assuming latest SDK supports system_instruction
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt
        )
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_schema else "text/plain"
        )
        
        response = model.generate_content(
            user_prompt,
            generation_config=generation_config
        )
        
        text = response.text
        elapsed = time.time() - start_time
        
        # Token usage (Gemini SDK uses usage_metadata)
        usage = response.usage_metadata
        self._log_call("analyze", usage.prompt_token_count, usage.candidates_token_count, elapsed)
        
        return self._parse_json_response(text)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """Get free-form text output from Gemini."""
        start_time = time.time()
        
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt
        )
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        response = model.generate_content(
            user_prompt,
            generation_config=generation_config
        )
        
        text = response.text
        elapsed = time.time() - start_time
        
        usage = response.usage_metadata
        self._log_call("generate", usage.prompt_token_count, usage.candidates_token_count, elapsed)
        
        return text
