import os
import pandas as pd
from stats import compute_basic_stats
from embeddings import compute_similarity
from qualitative import extract_extreme_cases

def load_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_json(path, lines=True)

def run_analysis(path):
    df = load_jsonl(path)

    # Stats
    stats, df = compute_basic_stats(df)
    print("=== STATS ===")
    print(stats)

    # Similarité
    sim_matrix = compute_similarity(df)
    print("=== SIMILARITY MATRIX SHAPE ===")
    print(sim_matrix.shape)

    # Qualitatif
    qualitative = extract_extreme_cases(df, sim_matrix)
    print("=== QUALITATIVE THRESHOLDS ===")
    print(qualitative)

    print("\n=== SAMPLE ANSWERS ===")
    print(df["answer"].head(5))

if __name__ == "__main__":
    run_analysis("../../data/output/fr_unspecific_output.jsonl")