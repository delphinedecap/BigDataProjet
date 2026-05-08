from typing import Any, Dict

from app.prompts.base import PromptVariant


class SystemPromptVariant(PromptVariant):
    """
    Variante avec consigne globale passée au modèle via system_prompt.
    Le prompt utilisateur original n'est pas modifié.
    """

    name = "system_prompt"

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def build_prompt(self, question: str, **kwargs) -> Dict[str, Any]:
        return {
            "system": self.system_prompt,
            "user": question,
            "meta": {
                "variant": self.name,
                "transformed": True,
                "description": "Ajout d'une consigne globale via system_prompt.",
                "parameters": {
                    "system_prompt": self.system_prompt,
                },
            },
        }