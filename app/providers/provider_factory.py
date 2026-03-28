from app.providers.ollama_provider import OllamaProvider
from app.providers.openai_compatible_provider import OpenAICompatibleProvider


def create_provider(provider_config: dict):
    """
    Crée le provider correspondant à la configuration.

    Cette factory évite d'avoir la logique de sélection du provider
    directement dans le runner.
    """
    
    provider_name = provider_config["name"]
    model_name = provider_config["model"]

    if provider_name == "ollama":
        return OllamaProvider(model=model_name)

    if provider_name == "openai_compatible":
        base_url = provider_config.get("base_url", "http://localhost:1234/v1")
        api_key = provider_config.get("api_key", "not-needed")
        return OpenAICompatibleProvider(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
        )

    raise ValueError(f"Provider non supporté : {provider_name}")