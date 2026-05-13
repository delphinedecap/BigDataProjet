import os
import pandas as pd

from stats import compute_basic_stats
from embeddings import compute_similarity
from qualitative import extract_extreme_cases


# Chemins de base du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "output")


def load_jsonl(path):
    # Vérifie que le fichier existe
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_json(path, lines=True)


def run_multiple_analysis(paths):
    results = {}

    for name, path in paths.items():
        print(f"\n===== ANALYSE: {name} =====")

        df = load_jsonl(path)

        # Vérifie la présence de la colonne attendue
        if "answer" not in df.columns:
            raise ValueError(f"Le fichier {name} ne contient pas de colonne 'answer'")

        df["answer"] = df["answer"].fillna("")

        # Statistiques de base
        stats, df = compute_basic_stats(df)

        # Calcul de similarité
        sim_matrix = compute_similarity(df)
        avg_similarity = sim_matrix.mean()

        # Seuils pour analyse qualitative
        qualitative = extract_extreme_cases(df, sim_matrix)

        print("\n--- Statistiques ---")
        print(stats)

        print("\n--- Similarité ---")
        print("Matrix shape:", sim_matrix.shape)
        print("Average similarity:", avg_similarity)

        print("\n--- Analyse qualitative ---")
        print(qualitative)

        print("\n--- Réponses ---")
        print(df["answer"].head(3))

        # Stockage pour comparaison finale
        results[name] = {
            "avg_word_count": stats["avg_word_count"],
            "avg_char_length": stats["avg_char_length"],
            "avg_similarity": avg_similarity
        }

    return results


if __name__ == "__main__":
    print("Current working dir:", os.getcwd())

    # Fichiers à analyser
    paths = {

        # ===== BASELINE =====
        "fr_unspecific": os.path.join(DATA_DIR, "fr_unspecific_output.jsonl"),
        "fr_specific": os.path.join(DATA_DIR, "fr_specific_output.jsonl"),

        "en_unspecific": os.path.join(DATA_DIR, "en_unspecific_output.jsonl"),
        "en_specific": os.path.join(DATA_DIR, "en_specific_output.jsonl"),

        "es_unspecific": os.path.join(DATA_DIR, "es_unspecific_output.jsonl"),
        "es_specific": os.path.join(DATA_DIR, "es_specific_output.jsonl"),

        "de_unspecific": os.path.join(DATA_DIR, "de_unspecific_output.jsonl"),
        "de_specific": os.path.join(DATA_DIR, "de_specific_output.jsonl"),

        "it_unspecific": os.path.join(DATA_DIR, "it_unspecific_output.jsonl"),
        "it_specific": os.path.join(DATA_DIR, "it_specific_output.jsonl"),

        # ===== VARIANT REWRITE =====
        "rewrite_fr_unspecific": os.path.join(DATA_DIR, "variant_rewrite_fr_unspecific_output.jsonl"),
        "rewrite_fr_specific": os.path.join(DATA_DIR, "variant_rewrite_fr_specific_output.jsonl"),

        "rewrite_en_unspecific": os.path.join(DATA_DIR, "variant_rewrite_en_unspecific_output.jsonl"),
        "rewrite_en_specific": os.path.join(DATA_DIR, "variant_rewrite_en_specific_output.jsonl"),

        "rewrite_es_unspecific": os.path.join(DATA_DIR, "variant_rewrite_es_unspecific_output.jsonl"),
        "rewrite_es_specific": os.path.join(DATA_DIR, "variant_rewrite_es_specific_output.jsonl"),

        "rewrite_de_unspecific": os.path.join(DATA_DIR, "variant_rewrite_de_unspecific_output.jsonl"),
        "rewrite_de_specific": os.path.join(DATA_DIR, "variant_rewrite_de_specific_output.jsonl"),

        "rewrite_it_unspecific": os.path.join(DATA_DIR, "variant_rewrite_it_unspecific_output.jsonl"),
        "rewrite_it_specific": os.path.join(DATA_DIR, "variant_rewrite_it_specific_output.jsonl"),

        # ===== VARIANT SYSTEM =====
        "system_fr_unspecific": os.path.join(DATA_DIR, "variant_system_prompt_fr_unspecific_output.jsonl"),
        "system_fr_specific": os.path.join(DATA_DIR, "variant_system_prompt_fr_specific_output.jsonl"),

        "system_en_unspecific": os.path.join(DATA_DIR, "variant_system_prompt_en_unspecific_output.jsonl"),
        "system_en_specific": os.path.join(DATA_DIR, "variant_system_prompt_en_specific_output.jsonl"),

        "system_es_unspecific": os.path.join(DATA_DIR, "variant_system_prompt_es_unspecific_output.jsonl"),
        "system_es_specific": os.path.join(DATA_DIR, "variant_system_prompt_es_specific_output.jsonl"),

        "system_de_unspecific": os.path.join(DATA_DIR, "variant_system_prompt_de_unspecific_output.jsonl"),
        "system_de_specific": os.path.join(DATA_DIR, "variant_system_prompt_de_specific_output.jsonl"),

        "system_it_unspecific": os.path.join(DATA_DIR, "variant_system_prompt_it_unspecific_output.jsonl"),
        "system_it_specific": os.path.join(DATA_DIR, "variant_system_prompt_it_specific_output.jsonl"),

        # ===== VARIANT NEUTRAL =====
        "neutral_fr_unspecific": os.path.join(DATA_DIR, "neutral_fr_unspecific_output.jsonl"),
        "neutral_fr_specific": os.path.join(DATA_DIR, "neutral_fr_specific_output.jsonl"),

        "neutral_en_unspecific": os.path.join(DATA_DIR, "neutral_en_unspecific_output.jsonl"),
        "neutral_en_specific": os.path.join(DATA_DIR, "neutral_en_specific_output.jsonl"),

        "neutral_es_unspecific": os.path.join(DATA_DIR, "neutral_es_unspecific_output.jsonl"),
        "neutral_es_specific": os.path.join(DATA_DIR, "neutral_es_specific_output.jsonl"),

        "neutral_de_unspecific": os.path.join(DATA_DIR, "neutral_de_unspecific_output.jsonl"),
        "neutral_de_specific": os.path.join(DATA_DIR, "neutral_de_specific_output.jsonl"),

        "neutral_it_unspecific": os.path.join(DATA_DIR, "neutral_it_unspecific_output.jsonl"),
        "neutral_it_specific": os.path.join(DATA_DIR, "neutral_it_specific_output.jsonl"),

        # ===== VARIANT CULTURAL =====
        "cultural_fr_unspecific": os.path.join(DATA_DIR, "cultural_fr_unspecific_output.jsonl"),
        "cultural_fr_specific": os.path.join(DATA_DIR, "cultural_fr_specific_output.jsonl"),

        "cultural_en_unspecific": os.path.join(DATA_DIR, "cultural_en_unspecific_output.jsonl"),
        "cultural_en_specific": os.path.join(DATA_DIR, "cultural_en_specific_output.jsonl"),

        "cultural_es_unspecific": os.path.join(DATA_DIR, "cultural_es_unspecific_output.jsonl"),
        "cultural_es_specific": os.path.join(DATA_DIR, "cultural_es_specific_output.jsonl"),

        "cultural_de_unspecific": os.path.join(DATA_DIR, "cultural_de_unspecific_output.jsonl"),
        "cultural_de_specific": os.path.join(DATA_DIR, "cultural_de_specific_output.jsonl"),

        "cultural_it_unspecific": os.path.join(DATA_DIR, "cultural_it_unspecific_output.jsonl"),
        "cultural_it_specific": os.path.join(DATA_DIR, "cultural_it_specific_output.jsonl"),
    }

    results = run_multiple_analysis(paths)

    # Comparaison globale
    print("\n\n===== Comparaison finale =====")

    for name, metrics in results.items():
        print(f"\n{name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")