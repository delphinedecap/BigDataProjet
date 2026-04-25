import os
import pandas as pd

from stats import compute_basic_stats
from embeddings import compute_similarity
from qualitative import extract_extreme_cases


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "output")


def load_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_json(path, lines=True)


def run_multiple_analysis(paths):
    results = {}

    for name, path in paths.items():
        print(f"\n===== ANALYSE: {name} =====")

        df = load_jsonl(path)

        if "answer" not in df.columns:
            raise ValueError(f"Le fichier {name} ne contient pas de colonne 'answer'")

        df["answer"] = df["answer"].fillna("")

        # Statistiques
        stats, df = compute_basic_stats(df)

        # Similarité
        sim_matrix = compute_similarity(df)
        avg_similarity = sim_matrix.mean()

        # Analyse qualitative
        qualitative = extract_extreme_cases(df, sim_matrix)

        print("\n--- Statistiques ---")
        print(stats)

        print("\n--- Similarité ---")
        print("Matrix shape:", sim_matrix.shape)
        print("Average similarity:", avg_similarity)

        print("\n--- Analyse qualitative ---")
        print(qualitative)

        print("\n--- Réponses  ---")
        print(df["answer"].head(3))

        # Stockage pour comparaison
        results[name] = {
            "avg_word_count": stats["avg_word_count"],
            "avg_char_length": stats["avg_char_length"],
            "avg_similarity": avg_similarity
        }

    return results


if __name__ == "__main__":
    print("Current working dir:", os.getcwd())

    paths = {
        "fr_unspecific": os.path.join(DATA_DIR, "fr_unspecific_output.jsonl"),
        "fr_specific": os.path.join(DATA_DIR, "fr_specific_output.jsonl"),
        "en_unspecific": os.path.join(DATA_DIR, "en_unspecific_output.jsonl"),
        "en_specific": os.path.join(DATA_DIR, "en_specific_output.jsonl"),
    }

    results = run_multiple_analysis(paths)

    # Comparaison finale
    print("\n\n===== Comparaison finale =====")

    for name, metrics in results.items():
        print(f"\n{name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")