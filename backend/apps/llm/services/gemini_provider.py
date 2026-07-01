from __future__ import annotations

import json
import urllib.request
import urllib.error

from .base_provider import BaseLLMProvider, LLMResult


class GeminiProvider(BaseLLMProvider):
    name = "gemini"

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 1200) -> LLMResult:
        if not self.api_key:
            raise ValueError("Gemini API key is not configured.")

        model = self.model or "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError("Gemini request failed.") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Gemini request failed.") from exc

        candidates = raw.get("candidates") or []
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts)
        return LLMResult(text=text, raw=raw, provider=self.name, model=model)
