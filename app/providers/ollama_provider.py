import requests
from typing import Any, Dict, Optional

from app.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    """
    Provider permettant de communiquer avec un modèle local servi par Ollama.
    """

    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def check_connection(self) -> None:
        """
        Vérifie que le serveur Ollama répond.
        """
        response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()

    def list_models(self) -> list[str]:
        """
        Retourne la liste des modèles disponibles dans Ollama.
        """
        response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]

    def pull_model(self, model_name: str) -> None:
        """
        Télécharge un modèle dans Ollama si nécessaire.
        """
        response = requests.post(
            f"{self.base_url}/api/pull",
            json={"model": model_name, "stream": False},
            timeout=600,
        )
        response.raise_for_status()

    def preload_model(self, model_name: Optional[str] = None, keep_alive: str = "60m") -> None:
        """
        Précharge un modèle pour éviter un temps d'attente trop long
        lors de la première génération.
        """
        payload = {
            "model": model_name or self.model,
            "keep_alive": keep_alive,
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )

        if not response.ok:
            raise ValueError(
                f"Erreur preload Ollama {response.status_code} : {response.text}"
            )

        response.raise_for_status()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Envoie un prompt à Ollama et retourne la réponse texte du modèle.
        """
        config = config or {}

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "keep_alive": config.get("keep_alive", "10m"),
            "options": {
                "temperature": config.get("temperature", 0.0),
                "top_p": config.get("top_p", 1.0),
                "num_predict": config.get("max_tokens", 80),
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