import numpy as np

def extract_extreme_cases(df, sim_matrix):
    similarities = sim_matrix.flatten()

    low_threshold = np.percentile(similarities, 10)
    high_threshold = np.percentile(similarities, 90)

    return {
        "low_similarity": low_threshold,
        "high_similarity": high_threshold
    }