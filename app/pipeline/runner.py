from datetime import datetime
from typing import Any, Dict

from app.pipeline.dataset_loader import load_jsonl
from app.pipeline.exporter import save_jsonl, save_metadata
from app.providers.openai_provider import OpenAIProvider


def run_experiment(config: Dict[str, Any]) -> None:
    input_path = config["input_path"]
    output_path = config["output_path"]
    metadata_path = config["metadata_path"]

    provider_name = config["provider"]["name"]
    model_name = config["provider"]["model"]

    generation_config = config.get("generation", {})
    system_prompt = config.get("system_prompt")

    if provider_name == "openai":
        provider = OpenAIProvider(model=model_name)
    else:
        raise ValueError(f"Provider non supporté : {provider_name}")

    rows = load_jsonl(input_path)
    output_rows = []

    for index, row in enumerate(rows, start=1):
        prompt = row.get("prompt", "").strip()

        if not prompt:
            row["answer"] = ""
            output_rows.append(row)
            continue

        try:
            answer = provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                config=generation_config,
            )
            row["answer"] = answer
        except Exception as e:
            row["answer"] = ""
            row["error"] = str(e)

        output_rows.append(row)
        print(f"[{index}/{len(rows)}] terminé")

    save_jsonl(output_rows, output_path)

    metadata = {
        "team": config.get("team", "team-name"),
        "system": config.get("system", "baseline-system"),
        "model": model_name,
        "submissionid": config.get("submissionid", "baseline-run"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "label": config.get("label", "eloquent-2026-cultural"),
        "languages": config.get("languages", []),
        "modifications": {
            "system_prompt": system_prompt,
            "generation_params": generation_config,
            "notes": config.get("notes", "Baseline vanilla run"),
        },
    }

    save_metadata(metadata, metadata_path)