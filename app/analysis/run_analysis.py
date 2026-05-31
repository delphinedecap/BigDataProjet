import json
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances

from stats import compute_basic_stats
from embeddings import compute_similarity
from qualitative import (
    extract_extreme_cases,
    get_most_similar_pairs,
    get_least_similar_pairs,
    detect_problematic_answers
)


# =========================
# CHEMINS DE BASE DU PROJET
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

DATA_DIR = os.path.join(BASE_DIR, "data", "output")
ANALYSIS_DIR = os.path.join(BASE_DIR, "data", "analysis")

REPORT_PATH = os.path.join(ANALYSIS_DIR, "analysis_report.txt")
SUMMARY_PATH = os.path.join(ANALYSIS_DIR, "analysis_summary.json")


# =========================
# OUTILS DE SAUVEGARDE
# =========================

class Tee:
    """
    Permet d'afficher dans le terminal ET d'écrire dans un fichier texte.
    """

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


def make_json_serializable(obj):
    """
    Convertit les objets pandas/numpy en objets compatibles JSON.
    """

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    if isinstance(obj, pd.Series):
        return obj.to_dict()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, tuple):
        return list(obj)

    if isinstance(obj, dict):
        return {
            key: make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_json_serializable(value)
            for value in obj
        ]

    return obj


def save_json_results(results, output_path=SUMMARY_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            make_json_serializable(results),
            file,
            ensure_ascii=False,
            indent=2
        )


# =========================
# CHARGEMENT JSONL
# =========================

def load_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    return pd.read_json(path, lines=True)


def prepare_dataframe(df, name):
    """
    Nettoie un dataframe de réponses :
    - suppression des doublons d'id
    - vérification de la colonne answer
    - remplacement des réponses manquantes
    """

    if "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset="id")
        after = len(df)

        if before != after:
            print(f"Doublons supprimés : {before - after}")

    if "answer" not in df.columns:
        raise ValueError(
            f"Le fichier {name} ne contient pas de colonne 'answer'"
        )

    df["answer"] = df["answer"].fillna("").astype(str)

    return df


# =========================
# ANALYSE D'UN RUN
# =========================

def run_single_analysis(name, path):
    print(f"\n===== ANALYSE : {name} =====")
    print(f"Fichier : {path}")

    df = load_jsonl(path)
    df = prepare_dataframe(df, name)

    # =========================
    # STATISTIQUES DE BASE
    # =========================

    stats, df = compute_basic_stats(df)

    # =========================
    # CALCUL DE SIMILARITÉ
    # =========================

    print(f"Nombre de réponses analysées : {len(df)}")
    print("Calcul TF-IDF + similarité...")

    sim_matrix = compute_similarity(df)

    print("Similarité terminée.")

    avg_similarity = float(sim_matrix.mean())

    # =========================
    # ANALYSE QUALITATIVE
    # =========================

    print("Extraction des cas extrêmes...")
    qualitative = extract_extreme_cases(df, sim_matrix)

    print("Recherche des paires les plus similaires...")
    most_similar = get_most_similar_pairs(df, sim_matrix, top_k=3)

    print("Recherche des paires les moins similaires...")
    least_similar = get_least_similar_pairs(df, sim_matrix, top_k=3)

    print("Détection des réponses problématiques...")
    problematic = detect_problematic_answers(df)

    # =========================
    # AFFICHAGE DES RÉSULTATS
    # =========================

    print("\n--- Statistiques ---")
    print(stats)

    print("\n--- Similarité ---")
    print("Matrix shape:", sim_matrix.shape)
    print("Average similarity:", avg_similarity)

    print("\n--- Analyse qualitative ---")
    print(qualitative)

    print("\n--- Réponses les PLUS similaires ---")

    for idx, pair in enumerate(most_similar):
        print(f"\nTOP {idx + 1}")
        print(f"Score : {pair['score']:.4f}")

        print("\nRéponse A :")
        print(pair["answer_a"])

        print("\nRéponse B :")
        print(pair["answer_b"])

    print("\n--- Réponses les MOINS similaires ---")

    for idx, pair in enumerate(least_similar):
        print(f"\nTOP {idx + 1}")
        print(f"Score : {pair['score']:.4f}")

        print("\nRéponse A :")
        print(pair["answer_a"])

        print("\nRéponse B :")
        print(pair["answer_b"])

    print("\n--- Réponses problématiques potentielles ---")

    if len(problematic) == 0:
        print("Aucune réponse problématique détectée.")
    else:
        for _, row in problematic.iterrows():
            print("\n----------------------")

            if "id" in row:
                print("ID :", row["id"])

            print("Réponse :")
            print(row["answer"])

    print("\n--- Exemples de réponses ---")
    examples = df["answer"].head(3).tolist()

    for idx, answer in enumerate(examples):
        print(f"\nExemple {idx + 1} :")
        print(answer)

    return {
        "stats": stats,
        "similarity": {
            "matrix_shape": sim_matrix.shape,
            "avg_similarity": avg_similarity
        },
        "qualitative": qualitative,
        "most_similar_pairs": most_similar,
        "least_similar_pairs": least_similar,
        "problematic_answers": problematic,
        "examples": examples
    }


# =========================
# ANALYSE MULTIPLE
# =========================

def run_multiple_analysis(paths):
    results = {}

    for name, path in paths.items():
        results[name] = run_single_analysis(name, path)

    return results


# =========================
# COMPARAISON BASELINE / VARIANT
# =========================

def compute_pairwise_answer_similarity(left_answers, right_answers):
    """
    Compare les réponses deux à deux :
    réponse baseline id=X vs réponse variant id=X.
    """

    left_answers = left_answers.fillna("").astype(str).tolist()
    right_answers = right_answers.fillna("").astype(str).tolist()

    if len(left_answers) == 0:
        return []

    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=1,
        max_df=1.0
    )

    all_answers = left_answers + right_answers
    tfidf = vectorizer.fit_transform(all_answers)

    n = len(left_answers)

    left_matrix = tfidf[:n]
    right_matrix = tfidf[n:]

    distances = paired_cosine_distances(left_matrix, right_matrix)
    similarities = 1 - distances

    return similarities.tolist()


def compare_two_runs_by_id(left_name, left_path, right_name, right_path, top_k=3):
    print(f"\n===== COMPARAISON PAR ID : {left_name} VS {right_name} =====")

    left_df = load_jsonl(left_path)
    right_df = load_jsonl(right_path)

    left_df = prepare_dataframe(left_df, left_name)
    right_df = prepare_dataframe(right_df, right_name)

    if "id" not in left_df.columns or "id" not in right_df.columns:
        print("Comparaison impossible : colonne 'id' absente.")
        return {
            "error": "Colonne id absente dans au moins un fichier."
        }

    merged = pd.merge(
        left_df[["id", "answer"]],
        right_df[["id", "answer"]],
        on="id",
        suffixes=("_left", "_right")
    )

    if len(merged) == 0:
        print("Aucun id commun trouvé.")
        return {
            "common_ids": 0,
            "error": "Aucun id commun trouvé."
        }

    merged["word_count_left"] = merged["answer_left"].astype(str).apply(
        lambda value: len(value.split())
    )

    merged["word_count_right"] = merged["answer_right"].astype(str).apply(
        lambda value: len(value.split())
    )

    merged["word_count_diff"] = (
        merged["word_count_right"] - merged["word_count_left"]
    )

    merged["abs_word_count_diff"] = merged["word_count_diff"].abs()

    merged["paired_similarity"] = compute_pairwise_answer_similarity(
        merged["answer_left"],
        merged["answer_right"]
    )

    avg_similarity = float(merged["paired_similarity"].mean())
    avg_word_count_diff = float(merged["word_count_diff"].mean())
    avg_abs_word_count_diff = float(merged["abs_word_count_diff"].mean())

    most_different_semantic = merged.sort_values(
        by="paired_similarity",
        ascending=True
    ).head(top_k)

    most_similar_semantic = merged.sort_values(
        by="paired_similarity",
        ascending=False
    ).head(top_k)

    biggest_length_diff = merged.sort_values(
        by="abs_word_count_diff",
        ascending=False
    ).head(top_k)

    print(f"Nombre d'IDs communs : {len(merged)}")
    print(f"Similarité moyenne baseline/variant : {avg_similarity:.4f}")
    print(f"Différence moyenne de mots : {avg_word_count_diff:.4f}")
    print(f"Différence absolue moyenne de mots : {avg_abs_word_count_diff:.4f}")

    print("\n--- Réponses baseline/variant les PLUS différentes lexicalement ---")

    for idx, row in most_different_semantic.iterrows():
        print("\n----------------------")
        print("ID :", row["id"])
        print(f"Similarité : {row['paired_similarity']:.4f}")

        print(f"\n{left_name} :")
        print(row["answer_left"])

        print(f"\n{right_name} :")
        print(row["answer_right"])

    print("\n--- Réponses baseline/variant les PLUS proches lexicalement ---")

    for idx, row in most_similar_semantic.iterrows():
        print("\n----------------------")
        print("ID :", row["id"])
        print(f"Similarité : {row['paired_similarity']:.4f}")

        print(f"\n{left_name} :")
        print(row["answer_left"])

        print(f"\n{right_name} :")
        print(row["answer_right"])

    print("\n--- Plus grandes différences de longueur ---")

    for idx, row in biggest_length_diff.iterrows():
        print("\n----------------------")
        print("ID :", row["id"])
        print(f"Différence de mots : {row['word_count_diff']}")
        print(f"Similarité : {row['paired_similarity']:.4f}")

        print(f"\n{left_name} :")
        print(row["answer_left"])

        print(f"\n{right_name} :")
        print(row["answer_right"])

    return {
        "common_ids": len(merged),
        "avg_paired_similarity": avg_similarity,
        "avg_word_count_diff": avg_word_count_diff,
        "avg_abs_word_count_diff": avg_abs_word_count_diff,
        "most_different_semantic": most_different_semantic.to_dict(orient="records"),
        "most_similar_semantic": most_similar_semantic.to_dict(orient="records"),
        "biggest_length_diff": biggest_length_diff.to_dict(orient="records")
    }


def run_pairwise_comparisons(paths):
    """
    Compare les variantes avec leur baseline correspondante,
    pour toutes les langues disponibles.
    """

    languages = ["fr", "en", "es", "de", "it"]
    specificities = ["unspecific", "specific"]
    variants = ["rewrite", "system", "neutral", "cultural"]

    comparison_results = {}

    for lang in languages:
        for specificity in specificities:
            baseline_name = f"{lang}_{specificity}"

            if baseline_name not in paths:
                continue

            for variant in variants:
                variant_name = f"{variant}_{lang}_{specificity}"

                if variant_name not in paths:
                    continue

                comparison_key = f"{baseline_name}_vs_{variant_name}"

                comparison_results[comparison_key] = compare_two_runs_by_id(
                    baseline_name,
                    paths[baseline_name],
                    variant_name,
                    paths[variant_name],
                    top_k=3
                )

    return comparison_results


# =========================
# FICHIERS À ANALYSER
# =========================

def get_paths():
    paths = {}

    languages = ["fr", "en", "es", "de", "it"]
    specificities = ["unspecific", "specific"]

    for lang in languages:
        for specificity in specificities:
            base_key = f"{lang}_{specificity}"

            paths[base_key] = os.path.join(
                DATA_DIR,
                f"{lang}_{specificity}_output.jsonl"
            )

            paths[f"rewrite_{base_key}"] = os.path.join(
                DATA_DIR,
                f"variant_rewrite_{lang}_{specificity}_output.jsonl"
            )

            paths[f"system_{base_key}"] = os.path.join(
                DATA_DIR,
                f"variant_system_prompt_{lang}_{specificity}_output.jsonl"
            )

            paths[f"neutral_{base_key}"] = os.path.join(
                DATA_DIR,
                f"{lang}_{specificity}_neutral_output.jsonl"
            )

            paths[f"cultural_{base_key}"] = os.path.join(
                DATA_DIR,
                f"{lang}_{specificity}_cultural_output.jsonl"
            )

    return paths


def filter_existing_paths(paths):
    existing_paths = {
        name: path
        for name, path in paths.items()
        if os.path.exists(path)
    }

    missing_paths = {
        name: path
        for name, path in paths.items()
        if not os.path.exists(path)
    }

    if missing_paths:
        print("\n===== FICHIERS IGNORÉS CAR INTROUVABLES =====")

        for name, path in missing_paths.items():
            print(f"{name} : {path}")

    return existing_paths


# =========================
# COMPARAISON FINALE
# =========================

def print_final_summary(results):
    print("\n\n===== COMPARAISON FINALE =====")

    for name, analysis in results.items():
        stats = analysis["stats"]
        similarity = analysis["similarity"]

        print(f"\n{name}")
        print(f"  avg_word_count : {stats.get('avg_word_count', 'N/A')}")
        print(f"  avg_char_length : {stats.get('avg_char_length', 'N/A')}")
        print(f"  avg_similarity : {similarity.get('avg_similarity', 'N/A')}")


def print_pairwise_summary(pairwise_results):
    print("\n\n===== SYNTHÈSE DES COMPARAISONS BASELINE / VARIANT =====")

    if not pairwise_results:
        print("Aucune comparaison baseline/variant disponible.")
        return

    for name, comparison in pairwise_results.items():
        print(f"\n{name}")
        print(f"  common_ids : {comparison.get('common_ids', 'N/A')}")
        print(
            "  avg_paired_similarity : "
            f"{comparison.get('avg_paired_similarity', 'N/A')}"
        )
        print(
            "  avg_word_count_diff : "
            f"{comparison.get('avg_word_count_diff', 'N/A')}"
        )
        print(
            "  avg_abs_word_count_diff : "
            f"{comparison.get('avg_abs_word_count_diff', 'N/A')}"
        )


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        tee = Tee(sys.stdout, report_file)

        with redirect_stdout(tee):
            print("Current working dir:", os.getcwd())
            print("Data dir:", DATA_DIR)
            print("Analysis dir:", ANALYSIS_DIR)

            paths = get_paths()
            existing_paths = filter_existing_paths(paths)

            results = run_multiple_analysis(existing_paths)

            pairwise_results = run_pairwise_comparisons(existing_paths)

            all_results = {
                "runs": results,
                "pairwise_comparisons": pairwise_results
            }

            print_final_summary(results)
            print_pairwise_summary(pairwise_results)

            save_json_results(all_results)

            print("\n===== SAUVEGARDE TERMINÉE =====")
            print(f"Rapport texte : {REPORT_PATH}")
            print(f"Résumé JSON : {SUMMARY_PATH}")