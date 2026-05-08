import sys
from pathlib import Path

import yaml

from app.pipeline.runner import run_experiment


DEFAULT_CONFIG = "app/config/baseline_fr_unspecific.yaml"


def load_config(path: str) -> dict:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration introuvable : {path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["config_file"] = str(config_path)
    return config


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)
    run_experiment(config)