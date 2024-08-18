from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

def extract_topic_features(reviews, n_topics):
    reviews = [str(review) for review in reviews]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_distribution = lda.fit_transform(X)
    return topic_distribution

def construct_preference_matrix(user_topic_features, product_topic_features):
    preference_matrix = np.dot(user_topic_features, product_topic_features.T)
    return preference_matrix