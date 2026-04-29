class VanillaPrompt:

    """retourne le prompt original sans modification"""

    name = "vanilla"

    def build_prompt(self, question: str, **kwargs) -> dict:
        return {
            "system": None,
            "user": question,
            "meta": {
                "variant": self.name,
                "transformed": False,
                "description": "Raw question, no transformation applied. #Vanilla ",
            },
        }