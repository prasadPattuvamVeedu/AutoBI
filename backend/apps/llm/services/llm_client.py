from __future__ import annotations

from django.conf import settings

from .gemini_provider import GeminiProvider
from .mock_provider import MockLLMProvider
from .openai_provider import OpenAIProvider
from .response_parser import parse_json_object


class LLMConfigurationError(Exception):
    """Raised when a provider is selected but not configured."""


class LLMServiceError(Exception):
    """Raised for provider failures that should be safe for frontend display."""


def get_provider():
    provider_name = getattr(settings, "LLM_PROVIDER", "mock").lower()
    if provider_name == "gemini":
        if not getattr(settings, "GEMINI_API_KEY", ""):
            raise LLMConfigurationError("Gemini API key is not configured.")
        return GeminiProvider(
            api_key=getattr(settings, "GEMINI_API_KEY", ""),
            model=getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash"),
        )
    if provider_name == "openai":
        if not getattr(settings, "OPENAI_API_KEY", ""):
            raise LLMConfigurationError("OpenAI API key is not configured.")
        return OpenAIProvider(
            api_key=getattr(settings, "OPENAI_API_KEY", ""),
            model=getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
        )
    return MockLLMProvider()


def generate_text(prompt: str, *, temperature: float = 0.2, max_tokens: int = 1200):
    provider = get_provider()
    try:
        return provider.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    except LLMConfigurationError:
        raise
    except Exception as exc:
        raise LLMServiceError("AI service is temporarily unavailable. Please try again later.") from exc


def generate_json(prompt: str, *, temperature: float = 0.2, max_tokens: int = 1200):
    result = generate_text(prompt, temperature=temperature, max_tokens=max_tokens)
    parsed = parse_json_object(result.text)
    parsed.setdefault("provider", result.provider)
    parsed.setdefault("model", result.model)
    return parsed
