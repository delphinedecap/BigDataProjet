from datetime import datetime
from typing import Any, Dict

from app.pipeline.dataset_loader import load_jsonl
from app.pipeline.exporter import save_jsonl, save_metadata
from app.providers.provider_factory import create_provider
from app.utils.logger import setup_logger
from app.prompts.prompt_factory import create_prompt_variant


def run_experiment(config: Dict[str, Any]) -> None:
    """
    Lance un run complet à partir d'une configuration YAML :
    - initialisation du logger ;
    - création du provider ;
    - vérification du provider ;
    - chargement du dataset ;
    - génération des réponses ;
    - sauvegarde des résultats et métadonnées.
    """
    logger = setup_logger(config.get("log_file", "runs/app.log"))

    input_path = config["input_path"]
    output_path = config["output_path"]
    metadata_path = config["metadata_path"]

    generation_config = config.get("generation", {})
    system_prompt = config.get("system_prompt")
    prompt_variant = create_prompt_variant(config.get("prompt_variant"))

    provider = create_provider(config["provider"])
    model_name = config["provider"]["model"]

    try:
        logger.info("Vérification de la connexion au provider...")
        provider.check_connection()

        logger.info("Récupération des modèles disponibles...")
        available_models = provider.list_models()
        logger.info("Modèles disponibles : %s", available_models)

        if model_name not in available_models:
            raise ValueError(
                f"Le modèle '{model_name}' n'est pas disponible. "
                f"Modèles disponibles : {available_models}"
            )

        logger.info("Préchargement du modèle : %s", model_name)
        try:
            provider.preload_model(model_name)
        except Exception as e:
            logger.warning("Préchargement ignoré : %s", str(e))

    except Exception as e:
        logger.error("Impossible d'initialiser le provider : %s", str(e))
        raise

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
            built_prompt = prompt_variant.build_prompt(prompt)

            answer = provider.generate(
                prompt=built_prompt["user"],
                system_prompt=built_prompt["system"] or system_prompt,
                config=generation_config,
            )

            row["original_prompt"] = prompt
            row["prompt_variant"] = built_prompt["meta"]
            row["transformed_prompt"] = built_prompt["user"]
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
        "team": config.get("team", "Master MIAGE Toulouse - DECAP SOUTRIC ZAIT AMBLARD"),
        "system": config.get("system", "baseline-system"),
        "submissionid": config.get("submissionid", "baseline-run"),
        "label": config.get("label", "Master MIAGE Toulouse"),
        "date": datetime.now().isoformat(timespec="seconds"),
        "config_file": config.get("config_file"),
        "languages": config.get("languages", []),
        "dataset_type": config.get("dataset_type"),
        "input_path": input_path,
        "output_path": output_path,
        "metadata_path": metadata_path,
        "prompt_count": len(rows),
        "success_count": sum(1 for row in output_rows if row.get("answer")),
        "error_count": sum(1 for row in output_rows if row.get("error")),
        "provider": config["provider"],
        "generation": generation_config,
        "system_prompt": system_prompt,
        "prompt_variant": config.get("prompt_variant", {"name": "vanilla"}),
        "notes": config.get("notes", ""),
    }

    logger.info("Sauvegarde des métadonnées dans : %s", metadata_path)
    save_metadata(metadata, metadata_path)

    logger.info("Run terminé.")