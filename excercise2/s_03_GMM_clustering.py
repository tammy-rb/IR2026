import os
import numpy as np

from sklearn.mixture import GaussianMixture
import umap

from s_03_clustering_utils import (
    VECTORS_LEMMAS_DIR,
    CLUSTERS_ROOT_DIR,
    NOISE_LABEL,
    load_bm25_and_metadata,
    save_clustering_result,
    assign_noise_to_nearest_cluster,
    get_cluster_size_dict,
    count_noise_points,
)

# ======================================
# Settings
# ======================================

POSTPROCESS_ASSIGN_NOISE = True
NOISE_MAX_DISTANCE = None      # If None → assign every point to the nearest cluster

UMAP_N_COMPONENTS = 10        # ← as requested
UMAP_METRIC = "cosine"


# ======================================
# GMM on UMAP-reduced space
# ======================================

def run_umap_reduction(matrix, n_components=10, metric="cosine"):
    """
    Reduce BM25 vectors to low-dimensional space using UMAP.

    Args:
        matrix: BM25 sparse matrix (n_docs x n_terms)
        n_components: number of UMAP output dimensions
        metric: distance metric ("cosine" recommended for sparse text vectors)

    Returns:
        matrix_10d: ndarray shape (n_docs, n_components)
        reducer: fitted UMAP reducer
    """
    print(
        f"Running UMAP dimensionality reduction → "
        f"{n_components} components (metric={metric}) ..."
    )

    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        random_state=42,
        low_memory=True,
        n_neighbors=15,
        min_dist=0.0,
    )

    matrix_10d = reducer.fit_transform(matrix)
    print("UMAP reduction completed.")

    return matrix_10d, reducer


def run_gmm_clustering(X, n_components=2, covariance_type="tied", max_iter=100):
    """
    Run Gaussian Mixture Model clustering on a dense matrix.

    Args:
        X: dense ndarray of shape (n_docs, n_features)
        n_components: #clusters (2 = UK vs US)
        covariance_type: "tied" recommended for high-dimensional data
        max_iter: max iterations of EM algorithm

    Returns:
        labels: hard cluster assignments
        gmm_model: fitted GaussianMixture model
    """

    print(
        f"Running GMM with n_components={n_components}, "
        f"covariance_type={covariance_type}, max_iter={max_iter} ..."
    )

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=42,
    )

    gmm.fit(X)
    labels = gmm.predict(X)

    print("GMM clustering completed.")
    return labels, gmm


# ======================================
# Main Pipeline
# ======================================

def main():
    print("=== GMM clustering on UMAP-reduced BM25 (lemmas, UK+US) ===")

    # 1. Load BM25 matrix and metadata
    bm25_matrix, feature_names, filenames, true_labels = load_bm25_and_metadata(
        VECTORS_LEMMAS_DIR
    )

    n_docs = bm25_matrix.shape[0]
    print(f"Number of documents: {n_docs}")

    # -------------------------
    # 2. UMAP dimensionality reduction
    # -------------------------
    matrix_10d, umap_reducer = run_umap_reduction(
        bm25_matrix,
        n_components=UMAP_N_COMPONENTS,
        metric=UMAP_METRIC,
    )

    # -------------------------
    # 3. GMM on the UMAP 10D space
    # -------------------------
    labels, gmm_model = run_gmm_clustering(
        matrix_10d,
        n_components=2,
        covariance_type="tied",
        max_iter=100,
    )

    # GMM never outputs -1 on its own
    n_original_noise = count_noise_points(labels, noise_label=NOISE_LABEL)

    # -------------------------
    # 4. Optional noise reassignment
    # -------------------------
    if POSTPROCESS_ASSIGN_NOISE:
        labels = assign_noise_to_nearest_cluster(
            matrix_10d,                # note: use the reduced space!
            labels,
            metric="euclidean",        # UMAP output is in Euclidean space
            noise_label=NOISE_LABEL,
            max_distance=NOISE_MAX_DISTANCE,
            normalize_centroids=True,
            inplace=False,
            verbose=True,
        )

    # -------------------------
    # 5. Summary statistics
    # -------------------------
    cluster_sizes = get_cluster_size_dict(labels)

    n_noise = int(cluster_sizes.get(NOISE_LABEL, 0))
    n_clusters = len([c for c in cluster_sizes.keys() if c != NOISE_LABEL])

    print("Cluster sizes (label → count):", cluster_sizes)
    print(f"Number of clusters (excluding noise): {n_clusters}")
    print(f"Number of noise points: {n_noise}")

    # -------------------------
    # 6. Save results
    # -------------------------
    gmm_output_dir = os.path.join(CLUSTERS_ROOT_DIR, "gmm")

    params = {
        "umap": {
            "enabled": True,
            "n_components": UMAP_N_COMPONENTS,
            "metric": UMAP_METRIC,
        },
        "gmm": {
            "n_components": 2,
            "covariance_type": "tied",
            "max_iter": 100,
        },
        "n_documents": int(n_docs),
        "n_clusters_excl_noise": n_clusters,
        "n_noise_points": n_noise,
        "cluster_sizes": cluster_sizes,
        "noise_reassignment": {
            "enabled": POSTPROCESS_ASSIGN_NOISE,
            "max_distance": NOISE_MAX_DISTANCE,
            "original_noise_count": int(n_original_noise),
        },
        "description": (
            "GMM clustering applied on UMAP-reduced BM25 (lemmas). "
            "UMAP reduces dimensionality to 10D with cosine metric. "
            "GMM then performs probabilistic clustering on the reduced space. "
            "Includes optional noise reassignment for completeness."
        ),
    }

    save_clustering_result(gmm_output_dir, labels, "GMM", params)

    print("\n✅ GMM + UMAP clustering finished and saved.\n")


if __name__ == "__main__":
    main()
