class RewriteVariant:

    name = "rewrite"

    def build_prompt(self, question: str, **kwargs) -> dict:
        rewritten = f"Answer clearly in one short sentence: {question}"

        return {
            "system": None,
            "user": rewritten,
            "meta": {
                "variant": self.name,
                "transformed": True,
                "original_question": question,
                "rewritten_question": rewritten,
                "description": "Simple manual rewrite adding instruction.",
            },
        }