from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(df):
    texts = df["answer"].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    sim_matrix = cosine_similarity(tfidf_matrix)

    return sim_matrix