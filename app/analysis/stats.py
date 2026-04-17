import pandas as pd

def compute_basic_stats(df):
    df["char_length"] = df["answer"].str.len()
    df["word_count"] = df["answer"].str.split().str.len()

    stats = {
        "avg_char_length": df["char_length"].mean(),
        "avg_word_count": df["word_count"].mean(),
        "empty_responses": df["answer"].isna().sum()
    }

    return stats, df