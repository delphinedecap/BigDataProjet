import numpy as np
import pandas as pd


def extract_extreme_cases(df, sim_matrix):
    n = len(df)

    if n < 2:
        return {
            "low_similarity": 0.0,
            "high_similarity": 0.0
        }

    upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

    return {
        "low_similarity": float(np.percentile(upper, 5)),
        "high_similarity": float(np.percentile(upper, 95))
    }


def _get_similarity_pairs(df, sim_matrix, top_k=3, most=True):
    n = len(df)

    if n < 2:
        return []

    rows, cols = np.triu_indices(n, k=1)
    scores = sim_matrix[rows, cols]

    top_k = min(top_k, len(scores))

    if most:
        selected_idx = np.argpartition(scores, -top_k)[-top_k:]
        selected_idx = selected_idx[np.argsort(scores[selected_idx])[::-1]]
    else:
        selected_idx = np.argpartition(scores, top_k - 1)[:top_k]
        selected_idx = selected_idx[np.argsort(scores[selected_idx])]

    pairs = []

    for idx in selected_idx:
        i = rows[idx]
        j = cols[idx]

        pairs.append({
            "score": float(scores[idx]),
            "answer_a": df.iloc[i]["answer"],
            "answer_b": df.iloc[j]["answer"]
        })

    return pairs


def get_most_similar_pairs(df, sim_matrix, top_k=3):
    return _get_similarity_pairs(df, sim_matrix, top_k=top_k, most=True)


def get_least_similar_pairs(df, sim_matrix, top_k=3):
    return _get_similarity_pairs(df, sim_matrix, top_k=top_k, most=False)


def detect_problematic_answers(df):
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

    for _, row in df.iterrows():
        answer = str(row["answer"]).lower()

        if len(answer.strip()) < 10:
            problematic_rows.append(row)
            continue

        for keyword in keywords:
            if keyword in answer:
                problematic_rows.append(row)
                break

    return pd.DataFrame(problematic_rows)