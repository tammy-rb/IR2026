import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from s_03_clustering_utils import (
    VECTORS_LEMMAS_DIR,
    CLUSTERS_ROOT_DIR,
    load_bm25_and_metadata,
    save_clustering_result,
    ensure_dir,
)


# ============================
# K-Means clustering
# ============================

def run_kmeans_clustering(bm25_matrix, n_clusters=2, random_state=42,
                          batch_size=256, max_iter=100):
    """
    Run MiniBatchKMeans on the given BM25 matrix.

    Args:
        bm25_matrix: sparse BM25 matrix (n_docs x n_terms)
        n_clusters: number of clusters (here = 2)
        random_state: random seed for reproducibility
        batch_size: MiniBatchKMeans batch size
        max_iter: max iterations

    Returns:
        labels: 1D array of cluster assignments (length = n_docs)
    """
    print(f"Running K-Means with k={n_clusters}...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init="auto",
    )
    labels = kmeans.fit_predict(bm25_matrix)
    print("K-Means clustering completed.")
    return labels, kmeans


def main():
    print("=== K-Means clustering on BM25 (lemmas, UK+US) ===")

    # 1. Load BM25 and metadata for lemmas
    bm25_matrix, feature_names, filenames, true_labels = load_bm25_and_metadata(
        VECTORS_LEMMAS_DIR
    )

    # 2. Run K-Means with k=2
    n_clusters = 2
    labels, kmeans_model = run_kmeans_clustering(
        bm25_matrix,
        n_clusters=n_clusters,
        random_state=42,
        batch_size=256,
        max_iter=100,
    )

    # Simple sanity print: how many docs per cluster
    unique, counts = np.unique(labels, return_counts=True)
    print("Cluster sizes:", dict(zip(unique, counts)))

    # 3. Save clustering result under clusters/kmeans
    kmeans_output_dir = os.path.join(CLUSTERS_ROOT_DIR, "kmeans")
    params = {
        "n_clusters": n_clusters,
        "random_state": 42,
        "batch_size": 256,
        "max_iter": 100,
    }
    save_clustering_result(kmeans_output_dir, labels, "MiniBatchKMeans", params)

    print("\n✅ K-Means clustering finished and saved.")


if __name__ == "__main__":
    main()
