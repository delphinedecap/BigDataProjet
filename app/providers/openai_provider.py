import os
from typing import Any, Dict, Optional

from openai import OpenAI

from app.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY manquante dans l'environnement.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        config = config or {}
        temperature = config.get("temperature", 0.0)
        max_tokens = config.get("max_tokens", 80)
        top_p = config.get("top_p", 1.0)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        text = response.choices[0].message.content or ""
        return text.strip()