import numpy as np

def extract_extreme_cases(df, sim_matrix):
    # Met toutes les similarités à plat (matrice -> liste)
    similarities = sim_matrix.flatten()

    # Seuils bas et haut (10% et 90%)
    low_threshold = np.percentile(similarities, 10)
    high_threshold = np.percentile(similarities, 90)

    # Permet d’identifier les cas très différents / très similaires
    return {
        "low_similarity": low_threshold,
        "high_similarity": high_threshold
    }