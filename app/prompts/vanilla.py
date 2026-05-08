from typing import Any, Dict

from app.prompts.base import PromptVariant


class VanillaPrompt(PromptVariant):
    """
    Variante vanilla : retourne le prompt original sans modification.
    Sert de baseline de comparaison avec les variantes de prompting.
    """

    name = "vanilla"

    def build_prompt(self, question: str, **kwargs) -> Dict[str, Any]:
        return {
            "system": None,
            "user": question,
            "meta": {
                "variant": self.name,
                "transformed": False,
                "description": "Aucune transformation du prompt original.",
                "parameters": {},
            },
        }