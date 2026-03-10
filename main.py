import yaml

from app.pipeline.runner import run_experiment


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config("app/config/default.yaml")
    run_experiment(config)