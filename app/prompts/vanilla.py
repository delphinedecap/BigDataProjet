class VanillaPrompt:

    """
    Variant vanilla:
    retourne le prompt original sans modification
    sert de baseline de comparaison avec les variants de prompting
    
    """

    name = "vanilla"

    def build_prompt(self, question: str, **kwargs) -> dict:
        return {
            "system": None,
            "user": question,
            "meta": {
                "variant": self.name,
                "transformed": False,
                "description": "Aucune transformation #Vanilla .",
            },
        }