from datetime import datetime
from typing import Any, Dict

from app.pipeline.dataset_loader import load_jsonl
from app.pipeline.exporter import save_jsonl, save_metadata
from app.providers.ollama_provider import OllamaProvider
from app.utils.logger import setup_logger


def build_provider(provider_config: Dict[str, Any]):
    provider_name = provider_config["name"]
    model_name = provider_config["model"]

    if provider_name == "ollama":
        return OllamaProvider(model=model_name)

    raise ValueError(f"Provider non supporté : {provider_name}")


def run_experiment(config: Dict[str, Any]) -> None:
    logger = setup_logger(config.get("log_file", "runs/app.log"))

    input_path = config["input_path"]
    output_path = config["output_path"]
    metadata_path = config["metadata_path"]

    generation_config = config.get("generation", {})
    system_prompt = config.get("system_prompt")

    provider = build_provider(config["provider"])

    logger.info("Chargement du dataset : %s", input_path)
    rows = load_jsonl(input_path)
    logger.info("Nombre de prompts chargés : %s", len(rows))

    output_rows = []

    for index, row in enumerate(rows, start=1):
        prompt = row.get("prompt", "").strip()

        if not prompt:
            logger.warning("Prompt vide à l'index %s", index)
            row["answer"] = ""
            row["error"] = "Prompt vide"
            output_rows.append(row)
            continue

        try:
            answer = provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                config=generation_config,
            )
            row["answer"] = answer
            logger.info("[%s/%s] Succès", index, len(rows))

        except Exception as e:
            logger.error("[%s/%s] Erreur : %s", index, len(rows), str(e))
            row["answer"] = ""
            row["error"] = str(e)

        output_rows.append(row)

    logger.info("Sauvegarde des résultats dans : %s", output_path)
    save_jsonl(output_rows, output_path)

    metadata = {
        "team": config.get("team", "mon-equipe"),
        "system": config.get("system", "baseline-system"),
        "submissionid": config.get("submissionid", "baseline-run"),
        "label": config.get("label", "eloquent-2026-cultural"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "languages": config.get("languages", []),
        "dataset_type": config.get("dataset_type"),
        "provider": config["provider"],
        "generation": generation_config,
        "system_prompt": system_prompt,
        "notes": config.get("notes", ""),
    }

    logger.info("Sauvegarde des métadonnées dans : %s", metadata_path)
    save_metadata(metadata, metadata_path)

    logger.info("Run terminé.")