from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMResult:
    text: str
    raw: Optional[Dict[str, Any]] = None
    provider: str = "unknown"
    model: str = "unknown"


class BaseLLMProvider(ABC):
    """Provider interface. Keep AutoBI business logic out of this layer."""

    name = "base"

    def __init__(self, api_key: str = "", model: str = "") -> None:
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 1200) -> LLMResult:
        raise NotImplementedError
