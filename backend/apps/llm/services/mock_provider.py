from __future__ import annotations

import json
from .base_provider import BaseLLMProvider, LLMResult


class MockLLMProvider(BaseLLMProvider):
    """Safe offline provider for learning and UI testing."""

    name = "mock"

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 1200) -> LLMResult:
        payload = {
            "summary": "AI provider is running in mock mode. Add an API key and provider to enable real LLM responses.",
            "suggestions": [
                {
                    "title": "Start with rule-based recommendations",
                    "reason": "They are reliable, fast, and explainable before adding generative AI.",
                    "confidence": 0.9,
                }
            ],
            "actions": [],
        }
        return LLMResult(text=json.dumps(payload), raw=payload, provider=self.name, model="mock")
