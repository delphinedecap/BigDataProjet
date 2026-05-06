import json
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: str) -> List[Dict]:
    """
    Charge un fichier JSONL ou tous les fichiers JSONL d'un dossier
    et retourne une liste de dictionnaires.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier ou dossier introuvable : {path}")

    rows = []

    if file_path.is_file():
        files = [file_path]
    elif file_path.is_dir():
        files = sorted(file_path.glob("*.jsonl"))
    else:
        raise ValueError(f"Chemin non valide : {path}")

    if not files:
        raise ValueError(f"Aucun fichier JSONL trouvé dans : {path}")

    for current_file in files:
        with current_file.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                    row["_source_file"] = current_file.name
                    rows.append(row)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Ligne {line_number} invalide dans {current_file}: {e}"
                    ) from e

    return rows