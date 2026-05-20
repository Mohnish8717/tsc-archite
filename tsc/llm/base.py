"""Abstract base class for all LLM providers."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TokenUsage(BaseModel):
    """Cumulative token usage tracker."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


class LLMClient(ABC):
    """Interface that all LLM providers must implement."""

    def __init__(self, api_key: str, model: str, **kwargs: Any):
        self.api_key = api_key
        self.model = model
        self._usage = TokenUsage()
        self._call_count = 0

    @abstractmethod
    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        """Get structured JSON output from the LLM.

        Args:
            system_prompt: System context / instructions.
            user_prompt: The actual question / data.
            json_schema: Optional JSON-Schema to constrain output.
            temperature: Creativity parameter (0.0–1.0).
            max_tokens: Maximum output tokens.

        Returns:
            Parsed JSON dict.
        """
        ...

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """Get free-form text output from the LLM.

        Used for long-form generation (persona profiles, specs, debate statements).
        """
        ...

    def get_usage(self) -> TokenUsage:
        """Return cumulative token usage."""
        return self._usage

    @property
    def call_count(self) -> int:
        return self._call_count

    # ── Helpers ──────────────────────────────────────────────────────

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Extract JSON from LLM response, tolerating markdown fences and conversational wrappers."""
        text = text.strip()
        
        # 1. First try parsing the raw text directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Try to extract JSON from markdown code blocks
        import re
        block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if block_match:
            try:
                return json.loads(block_match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. Try to find the largest valid JSON object or array in the text
        # We find all outermost {..} and [..] blocks and try parsing them.
        # This prevents picking up stray `[SPEAKER]` tags if they aren't valid JSON.
        
        candidates = []
        start_dict = text.find("{")
        if start_dict != -1:
            end_dict = text.rfind("}") + 1
            if end_dict > start_dict:
                candidates.append(text[start_dict:end_dict])
                
        start_list = text.find("[")
        if start_list != -1:
            end_list = text.rfind("]") + 1
            if end_list > start_list:
                candidates.append(text[start_list:end_list])

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")

    def _log_call(self, method: str, input_tok: int, output_tok: int, elapsed: float) -> None:
        self._call_count += 1
        self._usage.add(input_tok, output_tok)
        logger.debug(
            "[%s] %s call #%d: %d in / %d out tokens (%.1fs)",
            self.__class__.__name__,
            method,
            self._call_count,
            input_tok,
            output_tok,
            elapsed,
        )
