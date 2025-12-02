import os

import numpy as np
from sklearn.mixture import GaussianMixture

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

# Toggle post-processing of noise assignment
POSTPROCESS_ASSIGN_NOISE = True
NOISE_MAX_DISTANCE = None  # If None: assign every point to nearest cluster


# ============================
# GMM clustering
# ============================

def run_gmm_clustering(
    bm25_matrix,
    n_components=2,
    covariance_type="tied",
    max_iter=100,
    random_state=42,
):
    """
    Run Gaussian Mixture Model clustering on the BM25 vectors.
    GMM performs soft clustering but returns hard labels via predict().

    Args:
        bm25_matrix: sparse or dense matrix (n_docs x n_terms)
        n_components: number of clusters (2 for UK/US)
        covariance_type: "tied" (recommended for high-dimensional sparse data)
        max_iter: maximum EM iterations

    Returns:
        labels: hard cluster assignments (0,1)
        gmm_model: fitted GaussianMixture object
    """
    print(
        f"Running GMM with n_components={n_components}, "
        f"covariance_type={covariance_type}, max_iter={max_iter}"
    )

    # Convert sparse to dense for GMM (it requires dense matrix)
    if hasattr(bm25_matrix, "toarray"):
        X = bm25_matrix.toarray()
    else:
        X = bm25_matrix

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=random_state,
    )

    gmm.fit(X)
    labels = gmm.predict(X)

    print("GMM clustering completed.")
    return labels, gmm


def main():
    print("=== GMM clustering on BM25 (lemmas, UK+US) ===")

    # 1. Load BM25 matrix and metadata
    bm25_matrix, feature_names, filenames, true_labels = load_bm25_and_metadata(
        VECTORS_LEMMAS_DIR
    )
    n_docs = bm25_matrix.shape[0]
    print(f"Number of documents: {n_docs}")

    # -------------------------
    # Hyper-parameters
    # -------------------------
    n_components = 2
    covariance_type = "tied"
    max_iter = 100

    # 2. Run GMM
    labels, gmm_model = run_gmm_clustering(
        bm25_matrix,
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=42,
    )

    # Count noise points BEFORE reassignment (GMM never produces noise by itself)
    n_original_noise = count_noise_points(labels, noise_label=NOISE_LABEL)

    # ==========================================
    # Optional: reassign noise to nearest cluster
    # (Probably zero noise, but kept for consistency)
    # ==========================================
    if POSTPROCESS_ASSIGN_NOISE:
        labels = assign_noise_to_nearest_cluster(
            bm25_matrix,
            labels,
            metric="cosine",
            noise_label=NOISE_LABEL,
            max_distance=NOISE_MAX_DISTANCE,
            normalize_centroids=True,
            inplace=False,
            verbose=True,
        )

    # ===========================
    # Summary statistics
    # ===========================
    cluster_sizes = get_cluster_size_dict(labels)

    n_noise = int(cluster_sizes.get(NOISE_LABEL, 0))
    n_clusters = len([c for c in cluster_sizes.keys() if c != NOISE_LABEL])

    print("Cluster sizes (label -> count):", cluster_sizes)
    print(f"Number of clusters (excluding noise): {n_clusters}")
    print(f"Number of noise points: {n_noise}")

    # 3. Save results to clusters/gmm
    gmm_output_dir = os.path.join(CLUSTERS_ROOT_DIR, "gmm")

    params = {
        "n_components": n_components,
        "covariance_type": covariance_type,
        "max_iter": max_iter,
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
            "GMM clustering on BM25 (lemmas). "
            "Uses Gaussian Mixture Model with tied covariance "
            "for high-dimensional sparse data. "
            "Produces soft-clustering but outputs hard labels."
        ),
    }

    save_clustering_result(gmm_output_dir, labels, "GMM", params)

    print("\n✅ GMM clustering finished and saved.")


if __name__ == "__main__":
    main()
