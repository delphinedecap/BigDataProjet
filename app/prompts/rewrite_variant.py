class RewriteVariant:


    """ 
    Variant de réecriture :

    permet de modifier le prompt envoyé au modèle,
    en ajoutant une contrainte de réponse,
    à l'aide d'un préfixe : réponse en une phrase courte
    à """

    name = "rewrite"

    def build_prompt(self, question: str, **kwargs) -> dict:
        rewritten = (
            "Task: Answer in one short sentence.\n"
            "Requirement: Be practical and clear.\n"
            f"Question: {question}"
        )

        return {
            "system": None,
            "user": rewritten,
            "meta": {
                "variant": self.name,
                "transformed": True,
                "description": "Prompt structuré, clair et concis.",
            },
        }

    