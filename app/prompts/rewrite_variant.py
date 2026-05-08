from typing import Any, Dict

from app.prompts.base import PromptVariant


class RewriteVariant(PromptVariant):
    """
    Variante déterministe de reformulation légère.
    Elle ajoute une consigne explicite autour du prompt original.
    """

    name = "rewrite"

    def __init__(self, prefix: str = "", suffix: str = "") -> None:
        self.prefix = prefix.strip()
        self.suffix = suffix.strip()

    def build_prompt(self, question: str, **kwargs) -> Dict[str, Any]:
        parts = []

        if self.prefix:
            parts.append(self.prefix)

        parts.append(question.strip())

        if self.suffix:
            parts.append(self.suffix)

        rewritten_prompt = "\n\n".join(parts)

        return {
            "system": None,
            "user": rewritten_prompt,
            "meta": {
                "variant": self.name,
                "transformed": True,
                "description": "Reformulation déterministe par ajout d'un préfixe et/ou d'un suffixe.",
                "parameters": {
                    "prefix": self.prefix,
                    "suffix": self.suffix,
                },
            },
        }