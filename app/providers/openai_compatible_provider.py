import requests
from typing import Any, Dict, Optional

from app.providers.base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    """
    Provider générique pour une API compatible OpenAI.
    Peut être utilisé avec LM Studio, LocalAI, vLLM, etc.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "not-needed",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict:
        """
        Provider générique pour une API compatible OpenAI.
        Peut être utilisé avec LM Studio, LocalAI, vLLM, etc.
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def check_connection(self) -> None:
        """
        Vérifie que l'API compatible OpenAI répond.
        """
        response = requests.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()

    def list_models(self) -> list[str]:
        """
        Retourne la liste des modèles exposés par l'API.
        """
        response = requests.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return [model["id"] for model in data.get("data", [])]

    def preload_model(self, model_name: Optional[str] = None) -> None:
        """
        Pas de préchargement spécifique ici.
        Méthode présente pour garder la même interface que les autres providers.
        """
        return None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Envoie un prompt au format chat/completions et retourne la réponse.
        """
        config = config or {}

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": config.get("temperature", 0.0),
            "top_p": config.get("top_p", 1.0),
            "max_tokens": config.get("max_tokens", 80),
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()