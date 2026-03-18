"""Simple TF-IDF based clustering for themes."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer



def assign_clusters(comments_df: pd.DataFrame, topic_count: int = 6) -> pd.DataFrame:
    if comments_df.empty:
        comments_df["cluster_id"] = []
        return comments_df
    working = comments_df.copy()
    texts = working["cleaned_text"].fillna("").tolist()
    non_empty = [index for index, text in enumerate(texts) if text.strip()]
    working["cluster_id"] = -1
    if len(non_empty) < 2:
        return working
    vectorizer = TfidfVectorizer(max_features=300)
    matrix = vectorizer.fit_transform([texts[index] for index in non_empty])
    n_clusters = min(topic_count, len(non_empty))
    if n_clusters < 2:
        return working
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(matrix)
    for source_index, label in zip(non_empty, labels):
        working.loc[working.index[source_index], "cluster_id"] = int(label)
    return working
