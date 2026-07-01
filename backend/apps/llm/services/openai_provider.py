from __future__ import annotations

import json
import urllib.request
import urllib.error

from .base_provider import BaseLLMProvider, LLMResult


class OpenAIProvider(BaseLLMProvider):
    name = "openai"

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 1200) -> LLMResult:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is missing.")

        model = self.model or "gpt-4o-mini"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI request failed: {exc.code} {detail}") from exc

        text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        return LLMResult(text=text, raw=raw, provider=self.name, model=model)
