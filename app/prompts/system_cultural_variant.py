class SystemCulturalVariant:

    """ 
    Variant culturel :

    Ce variant permet reprendre le prompt original, en apportant un contexte
    au modèle, lui imposant une conscience culturelle, et de s'adapter au langage employé
    dans le prompt originel, ainsi qu'au contexte culturel implicitement ou explicitement
    indiqué dans le prompt originel.
    Le LLM doit également repondre dans la langue de la question posée.
    """

    name = "cultural"


    def build_prompt(self, question: str, **kwargs) -> dict:

        system = (
             "You are a culturally aware assistant. "
             "Adapt your answer to the cultural context implied by the question. "
             "If a country or culture is mentioned, base your answer on it. "
             "If not, infer a plausible cultural context from the language of the question. "
             "Follow the user's instructions strictly."
        )

        return {
            "system": system,
            "user": question,
            "meta": {
                "variant": self.name,
                "transformed": False,
                "description": "Modèle conscient du contexte culturel. Respect de la langues et des instructions",
            },
        }