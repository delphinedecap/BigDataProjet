import json
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: str) -> List[Dict]:
    """
    Charge un fichier JSONL et retourne son contenu
    sous forme d'une liste de dictionnaires.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Ligne {line_number} invalide dans {path}: {e}") from e

    return rows