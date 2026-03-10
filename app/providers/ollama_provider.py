import requests
from typing import Any, Dict, Optional

from app.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        config = config or {}
        temperature = config.get("temperature", 0.0)
        top_p = config.get("top_p", 1.0)
        max_tokens = config.get("max_tokens", 80)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()