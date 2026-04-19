import sys
import yaml

from app.pipeline.runner import run_experiment


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config_path = "app/config/baseline_fr_unspecific.yaml"

    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = load_config(config_path)
    run_experiment(config)