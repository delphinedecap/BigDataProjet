from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(df):
    # Récupère les réponses (remplace les valeurs manquantes)
    texts = df["answer"].fillna("").tolist()

    # Transformation des textes en vecteurs TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calcul de la similarité cosinus entre toutes les réponses
    sim_matrix = cosine_similarity(tfidf_matrix)

    return sim_matrix