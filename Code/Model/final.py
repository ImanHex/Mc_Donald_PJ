import pickle
from pathlib import Path
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

stored_folder = Path(os.path.abspath('')).parent.parent / "data" / "processed" / "cleaned_df.pkl"
input_file = open(stored_folder, "rb")
cleaned_data = pickle.load(input_file)


def compute_cluster_groups(shrunk_norm_matrix, review=cleaned_data['processed_review']):
    cluster_model = KMeans(n_clusters=10, n_init=10, random_state=42)
    clusters = cluster_model.fit_predict(shrunk_norm_matrix)
    df = pd.DataFrame({'Index': range(clusters.size), 'Cluster': clusters, 'Review': review})
    return [df_cluster for _, df_cluster in df.groupby('Cluster')]


def compute_cluster_models(tfidf_list):
    cluster_models = []
    for tfidf_matrix in tfidf_list:
        if tfidf_matrix.ndim > 2:
            tfidf_matrix = tfidf_matrix.squeeze()
        normalized_matrix = normalize(tfidf_matrix)
        cluster_model = KMeans(n_clusters=10, n_init=10, random_state=42)
        cluster_model.fit(normalized_matrix)
        cluster_models.append(cluster_model)
    return cluster_models


if __name__ == "__main__":

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_list = []
    tfidf_vocab = {}

    for rating in sorted(cleaned_data['rating'].unique()):
        tfidf_list.append(tfidf_vectorizer.fit_transform(cleaned_data[cleaned_data['rating'] == rating].processed_review))
        tfidf_vocab[rating] = tfidf_vectorizer.get_feature_names_out()


    df_rank = pd.DataFrame(
        {'Words': tfidf_vocab.get(2), 'Summed TFIDF': tfidf_list[1].toarray().sum(axis=0)}).sort_values('Summed TFIDF',
                                                                                                        ascending=False)

    shrunk_norm_matrix_list = []
    cluster_models = []
    for tfidf in tfidf_list:
        shrunk_norm_matrix_list.append(normalize(TruncatedSVD(n_components=100, random_state=42).fit_transform(tfidf)))

    cluster_models = compute_cluster_models(tfidf_list)

    cluster_groups = []
    true_labels = cleaned_data['rating']
    for index, rating in enumerate(sorted(cleaned_data.rating.unique())):
        cluster_groups.append(
            compute_cluster_groups(shrunk_norm_matrix_list[index], review=cleaned_data[cleaned_data.rating == rating].processed_review))

    predicted_labels = [cluster_df['Cluster'].tolist() for cluster_group in cluster_groups for cluster_df in
                        cluster_group]
    predicted_labels = [item for sublist in predicted_labels for item in sublist]

    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    print(f"ARI: {ari_score}")

    output_dir = Path(os.path.abspath('')).parent.parent / "data" / "modeling"
    with open(str(output_dir) + '/cluster_models.pkl', 'wb') as f:
        pickle.dump(cluster_groups, f)
