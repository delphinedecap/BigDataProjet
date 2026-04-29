class SystemPromptVariant:
    """ Variant culturel """

    name = "system_prompt"

    def __init__(self):
        self.system_text = (
             "You are a culturally aware assistant. "
             "Answer the user's question in exactly one short sentence, "
             "taking into account the cultural context implied by the language or "
             "explicitly mentioned in the question. "
             "Always reply in the same language as the question."
        )

    def build_prompt(self, question: str, **kwargs) -> dict:
        return {
            "system": self.system_text,
            "user": question,
            "meta": {
                "variant": self.name,
                "transformed": False,
                "system_text": self.system_text,
                "description": "Adds an expert role via system prompt.",
            },
        }