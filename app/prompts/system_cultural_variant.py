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
             "Respect the user's instructions strictly. "
             "Do not mention forbidden elements. "
             "Focus on the specific country mentioned if mentioned. "
             "Answer in one sentence and in the same language."
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