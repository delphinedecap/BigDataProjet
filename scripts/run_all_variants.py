from pathlib import Path
import sys
import yaml

from app.pipeline.runner import run_experiment


CONFIG_DIR = Path("app/config")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["config_file"] = str(path)
    return config


def main() -> None:
    config_files = sorted(CONFIG_DIR.glob("variant_*.yaml"))

    if not config_files:
        print("Aucune configuration de variante trouvée dans app/config.")
        sys.exit(1)

    print(f"{len(config_files)} configurations de variantes trouvées.")

    failed_runs = []

    for config_file in config_files:
        print(f"\n=== Lancement : {config_file} ===")

        try:
            config = load_config(config_file)
            run_experiment(config)
        except Exception as error:
            failed_runs.append((config_file, str(error)))
            print(f"Erreur pendant le run {config_file} : {error}")

    if failed_runs:
        print("\nCertains runs ont échoué :")
        for config_file, error in failed_runs:
            print(f"- {config_file} : {error}")
        sys.exit(1)

    print("\nTous les runs de variantes sont terminés avec succès.")


if __name__ == "__main__":
    main()