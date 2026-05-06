from app.prompts.system_neutral_variant import SystemNeutralVariant
from app.prompts.system_cultural_variant import SystemCulturalVariant
from app.prompts.vanilla import VanillaPrompt
from app.prompts.rewrite_variant import RewriteVariant


def create_variant(name: str):
    if name == "neutral":
        return SystemNeutralVariant()
    elif name == "cultural":
        return SystemCulturalVariant()
    elif name == "rewrite":
        return RewriteVariant()
    else:
        return VanillaPrompt()
    