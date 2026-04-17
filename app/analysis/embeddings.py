from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(df):
    texts = df["answer"].fillna("").tolist()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    sim_matrix = cosine_similarity(tfidf_matrix)

    return sim_matrix