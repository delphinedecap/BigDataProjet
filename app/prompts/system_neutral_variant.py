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
           "Provide a general answer that is not specific to any culture unless explicitly stated. "
           "Avoid assumptions about cultural practices. "
           "Follow the user's instructions strictly."
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