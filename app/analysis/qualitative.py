import numpy as np
import pandas as pd


# =========================
# EXTRACTION DES SEUILS
# =========================

def extract_extreme_cases(df, sim_matrix):

    # Récupère uniquement le triangle supérieur
    # pour éviter les doublons de comparaison
    upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

    return {

        # 5% les moins similaires
        "low_similarity": float(np.percentile(upper, 5)),

        # 5% les plus similaires
        "high_similarity": float(np.percentile(upper, 95))
    }


# =========================
# PAIRS LES PLUS SIMILAIRES
# =========================

def get_most_similar_pairs(df, sim_matrix, top_k=3):

    pairs = []

    n = len(df)

    # Compare chaque réponse avec les autres
    for i in range(n):

        for j in range(i + 1, n):

            pairs.append({

                "score": sim_matrix[i][j],

                "answer_a": df.iloc[i]["answer"],

                "answer_b": df.iloc[j]["answer"]
            })

    # Trie du plus similaire au moins similaire
    pairs = sorted(
        pairs,
        key=lambda x: x["score"],
        reverse=True
    )

    return pairs[:top_k]


# =========================
# PAIRS LES MOINS SIMILAIRES
# =========================

def get_least_similar_pairs(df, sim_matrix, top_k=3):

    pairs = []

    n = len(df)

    # Compare chaque réponse avec les autres
    for i in range(n):

        for j in range(i + 1, n):

            pairs.append({

                "score": sim_matrix[i][j],

                "answer_a": df.iloc[i]["answer"],

                "answer_b": df.iloc[j]["answer"]
            })

    # Trie du moins similaire au plus similaire
    pairs = sorted(
        pairs,
        key=lambda x: x["score"]
    )

    return pairs[:top_k]


# =========================
# DÉTECTION DE RÉPONSES
# POTENTIELLEMENT PROBLÉMATIQUES
# =========================

def detect_problematic_answers(df):

    # Expressions souvent associées
    # à des réponses faibles ou invalides
    keywords = [

        "i don't know",
        "unknown",
        "cannot answer",
        "je ne sais pas",
        "désolé",
        "sorry",
        "no information",
        "n/a"
    ]

    problematic_rows = []

    # Parcours des réponses
    for _, row in df.iterrows():

        answer = str(row["answer"]).lower()

        # Réponse trop courte
        if len(answer.strip()) < 10:

            problematic_rows.append(row)

            continue

        # Recherche de mots-clés problématiques
        for keyword in keywords:

            if keyword in answer:

                problematic_rows.append(row)

                break

    return pd.DataFrame(problematic_rows)