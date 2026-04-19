import os
import pandas as pd

from stats import compute_basic_stats
from embeddings import compute_similarity
from qualitative import extract_extreme_cases


# 🔹 Remonter à la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "output")


def load_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_json(path, lines=True)


def run_multiple_analysis(paths):
    results = {}

    for name, path in paths.items():
        print(f"\n===== ANALYSIS: {name} =====")

        df = load_jsonl(path)

        # sécurité
        if "answer" not in df.columns:
            raise ValueError(f"Le fichier {name} ne contient pas de colonne 'answer'")

        df["answer"] = df["answer"].fillna("")

        # 🔹 Stats
        stats, df = compute_basic_stats(df)

        # 🔹 Similarité
        sim_matrix = compute_similarity(df)
        avg_similarity = sim_matrix.mean()

        # 🔹 Qualitatif
        qualitative = extract_extreme_cases(df, sim_matrix)

        # 🔹 Print détaillé
        print("\n--- STATS ---")
        print(stats)

        print("\n--- SIMILARITY ---")
        print("Matrix shape:", sim_matrix.shape)
        print("Average similarity:", avg_similarity)

        print("\n--- QUALITATIVE ---")
        print(qualitative)

        print("\n--- SAMPLE ANSWERS ---")
        print(df["answer"].head(3))

        # 🔹 Stockage pour comparaison
        results[name] = {
            "avg_word_count": stats["avg_word_count"],
            "avg_char_length": stats["avg_char_length"],
            "avg_similarity": avg_similarity
        }

    return results


if __name__ == "__main__":
    print("Current working dir:", os.getcwd())

    # 🔥 Mets ici tous tes fichiers générés
    paths = {
        "fr_unspecific": os.path.join(DATA_DIR, "fr_unspecific_output.jsonl"),
        "fr_specific": os.path.join(DATA_DIR, "fr_specific_output.jsonl"),
        "en_unspecific": os.path.join(DATA_DIR, "en_unspecific_output.jsonl"),
        "en_specific": os.path.join(DATA_DIR, "en_specific_output.jsonl"),
    }

    results = run_multiple_analysis(paths)

    # 🔥 COMPARAISON FINALE (IMPORTANT POUR LE RAPPORT)
    print("\n\n===== FINAL COMPARISON =====")

    for name, metrics in results.items():
        print(f"\n{name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")