from app.prompts.rewrite_variant import RewriteVariant
from app.prompts.system_prompt_variant import SystemPromptVariant
from app.prompts.system_cultural_variant import SystemCulturalVariant
from app.prompts.system_neutral_variant import SystemNeutralVariant
from app.prompts.vanilla import VanillaPrompt


def create_prompt_variant(config: dict):
    config = config or {}
    variant_name = config.get("name", "vanilla")

    if variant_name == "vanilla":
        return VanillaPrompt()

    if variant_name == "system_prompt":
        return SystemPromptVariant(
            system_prompt=config.get(
                "system_prompt",
                "Réponds de manière courte, neutre et culturellement prudente, en une seule phrase.",
            )
        )

    if variant_name == "rewrite":
        return RewriteVariant(
            prefix=config.get("prefix", ""),
            suffix=config.get("suffix", ""),
        )

    if variant_name == "cultural":
        return SystemCulturalVariant()

    if variant_name == "neutral":
        return SystemNeutralVariant()

    raise ValueError(f"Variante de prompt non supportée : {variant_name}")