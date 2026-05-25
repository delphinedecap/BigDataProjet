import os
import pandas as pd

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "output")


# =========================
# CHARGEMENT JSONL
# =========================

def load_jsonl(path):

    # Vérifie que le fichier existe
    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Fichier introuvable: {path}"
        )

    return pd.read_json(path, lines=True)


# =========================
# ANALYSE MULTIPLE
# =========================

def run_multiple_analysis(paths):

    results = {}

    for name, path in paths.items():

        print(f"\n===== ANALYSE: {name} =====")

        df = load_jsonl(path)

        # =========================
        # SUPPRESSION DES DOUBLONS
        # =========================

        if "id" in df.columns:

            before = len(df)

            df = df.drop_duplicates(subset="id")

            after = len(df)

            if before != after:

                print(
                    f"Doublons supprimés : "
                    f"{before - after}"
                )

        # =========================
        # VALIDATION DES DONNÉES
        # =========================

        if "answer" not in df.columns:

            raise ValueError(
                f"Le fichier {name} "
                f"ne contient pas de colonne 'answer'"
            )

        # Gestion des valeurs manquantes
        df["answer"] = df["answer"].fillna("")

        # =========================
        # STATISTIQUES DE BASE
        # =========================

        stats, df = compute_basic_stats(df)

        # =========================
        # CALCUL DE SIMILARITÉ
        # =========================

        sim_matrix = compute_similarity(df)

        avg_similarity = sim_matrix.mean()

        # =========================
        # ANALYSE QUALITATIVE
        # =========================

        qualitative = extract_extreme_cases(
            df,
            sim_matrix
        )

        # Réponses les plus similaires
        most_similar = get_most_similar_pairs(
            df,
            sim_matrix,
            top_k=3
        )

        # Réponses les moins similaires
        least_similar = get_least_similar_pairs(
            df,
            sim_matrix,
            top_k=3
        )

        # Réponses problématiques
        problematic = detect_problematic_answers(df)

        # =========================
        # AFFICHAGE DES STATS
        # =========================

        print("\n--- Statistiques ---")
        print(stats)

        print("\n--- Similarité ---")
        print("Matrix shape:", sim_matrix.shape)

        print(
            "Average similarity:",
            avg_similarity
        )

        print("\n--- Analyse qualitative ---")
        print(qualitative)

        # =========================
        # RÉPONSES LES PLUS
        # SIMILAIRES
        # =========================

        print("\n--- Réponses les PLUS similaires ---")

        for idx, pair in enumerate(most_similar):

            print(f"\nTOP {idx + 1}")

            print(
                f"Score: "
                f"{pair['score']:.4f}"
            )

            print("\nRéponse A:")
            print(pair["answer_a"])

            print("\nRéponse B:")
            print(pair["answer_b"])

        # =========================
        # RÉPONSES LES MOINS
        # SIMILAIRES
        # =========================

        print("\n--- Réponses les MOINS similaires ---")

        for idx, pair in enumerate(least_similar):

            print(f"\nTOP {idx + 1}")

            print(
                f"Score: "
                f"{pair['score']:.4f}"
            )

            print("\nRéponse A:")
            print(pair["answer_a"])

            print("\nRéponse B:")
            print(pair["answer_b"])

        # =========================
        # RÉPONSES PROBLÉMATIQUES
        # =========================

        print(
            "\n--- Réponses problématiques "
            "potentielles ---"
        )

        if len(problematic) == 0:

            print(
                "Aucune réponse problématique détectée."
            )

        else:

            for _, row in problematic.iterrows():

                print("\n----------------------")

                if "id" in row:

                    print("ID:", row["id"])

                print("Réponse:")
                print(row["answer"])

        # =========================
        # EXEMPLES DE RÉPONSES
        # =========================

        print("\n--- Exemples de réponses ---")

        print(df["answer"].head(3))

        # =========================
        # STOCKAGE DES RÉSULTATS
        # =========================

        results[name] = {

            "avg_word_count":
                stats["avg_word_count"],

            "avg_char_length":
                stats["avg_char_length"],

            "avg_similarity":
                avg_similarity
        }

    return results


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print(
        "Current working dir:",
        os.getcwd()
    )

    # =========================
    # FICHIERS À ANALYSER
    # =========================

    paths = {

        # ===== BASELINE =====

        "fr_unspecific":
            os.path.join(
                DATA_DIR,
                "fr_unspecific_output.jsonl"
            ),

        "fr_specific":
            os.path.join(
                DATA_DIR,
                "fr_specific_output.jsonl"
            ),

        # Ajouter les autres fichiers ici...
    }

    # =========================
    # LANCEMENT ANALYSE
    # =========================

    results = run_multiple_analysis(paths)

    # =========================
    # COMPARAISON GLOBALE
    # =========================

    print("\n\n===== Comparaison finale =====")

    for name, metrics in results.items():

        print(f"\n{name}")

        for k, v in metrics.items():

            print(f"  {k}: {v:.4f}")