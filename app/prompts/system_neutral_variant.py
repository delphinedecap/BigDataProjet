class SystemNeutralVariant:

    """
    Variant neutre:

    Ce variant est neutre, il ne prend pas en compte une conscience culturelle,
    il évite les stéreotypes sur les différentes cultures,
    et répond de façon factuelle et générale. 
    
    """

    name = "neutral"

    def build_prompt(self, question: str, **kwargs) -> dict:

        system = (
            "You are a neutral assistant. "
            "Answer in one concise sentence. "
            "Avoid cultural stereotypes. "
            "Be factual and general."
        )

        return {
            "system": system,
            "user": question,
            "meta": {
                "variant": self.name,
                "transformed": True,
                "description": "Neutralité et généralité face au context culturel."
            },
        }