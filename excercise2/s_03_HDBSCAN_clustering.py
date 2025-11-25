import os

import numpy as np
import hdbscan  

from s_03_clustering_utils import (
    VECTORS_LEMMAS_DIR,
    CLUSTERS_ROOT_DIR,
    NOISE_LABEL,
    load_bm25_and_metadata,
    save_clustering_result,
    assign_noise_to_nearest_cluster,
    get_cluster_size_dict,
)


# Toggle post-processing of noise points
POSTPROCESS_ASSIGN_NOISE = True
# If None: assign every noise point to the nearest cluster
# Else: only assign noise points whose nearest distance <= NOISE_MAX_DISTANCE
NOISE_MAX_DISTANCE = None


# ============================
# HDBSCAN clustering
# ============================

def run_hdbscan_clustering(
    bm25_matrix,
    min_cluster_size=30,
    min_samples=8,
    metric="cosine",
    cluster_selection_method="eom",
):
    """
    Run HDBSCAN on the given BM25 matrix using cosine distance.

    Args:
        bm25_matrix: sparse BM25 matrix (n_docs x n_terms)
        min_cluster_size (int): minimal size of a cluster to be considered valid.
        min_samples (int): controls how conservative the clustering is.
                           Higher = more points become noise, clusters are denser.
        metric (str): distance metric to use.
        cluster_selection_method (str): "eom" (Excess of Mass) or "leaf".

    Returns:
        labels: 1D array of cluster assignments (length = n_docs)
                noise points are labeled as NOISE_LABEL.
        clusterer: fitted HDBSCAN object.
    """
    print(
        f"Running HDBSCAN with "
        f"min_cluster_size={min_cluster_size}, "
        f"min_samples={min_samples}, "
        f"metric={metric}, "
        f"cluster_selection_method={cluster_selection_method}..."
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=-1,  # parallel core distance computation if possible
    )

    labels = clusterer.fit_predict(bm25_matrix)
    print("HDBSCAN clustering completed.")
    return labels, clusterer


def main():
    print("=== HDBSCAN clustering on BM25 (lemmas, UK+US) ===")

    # 1. Load BM25 and metadata for lemmas (joint UK+US matrix)
    bm25_matrix, feature_names, filenames, true_labels = load_bm25_and_metadata(
        VECTORS_LEMMAS_DIR
    )

    n_docs = bm25_matrix.shape[0]
    print(f"Number of documents: {n_docs}")

    # -------------------------
    # Hyper-parameter choice
    # -------------------------
    # Choose min_cluster_size as ~3–5% of dataset size (~600 docs)
    # Choose min_samples to require reasonable local density without being overly strict
    min_cluster_size = 30
    min_samples = 8

    # 2. Run HDBSCAN
    labels, clusterer = run_hdbscan_clustering(
        bm25_matrix,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="cosine",
        cluster_selection_method="eom",
    )

    # ==========================================
    # Optional: reassign noise to nearest cluster
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

    # Number of noise points
    n_noise = int(cluster_sizes.get(NOISE_LABEL, 0))

    # Number of real clusters (excluding noise)
    n_clusters = len([c for c in cluster_sizes.keys() if c != NOISE_LABEL])

    print("Cluster sizes (label -> count):", cluster_sizes)
    print(f"Number of clusters (excluding noise): {n_clusters}")
    print(f"Number of noise points: {n_noise}")

    # 3. Save clustering result under clusters/hdbscan
    hdbscan_output_dir = os.path.join(CLUSTERS_ROOT_DIR, "hdbscan")

    params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": "cosine",
        "cluster_selection_method": "eom",
        "n_documents": int(n_docs),
        "n_clusters_excl_noise": n_clusters,
        "n_noise_points": n_noise,
        "cluster_sizes": cluster_sizes,
        "noise_reassignment": {
            "enabled": POSTPROCESS_ASSIGN_NOISE,
            "max_distance": NOISE_MAX_DISTANCE,
        },
        "description": (
            "HDBSCAN run on joint BM25 (lemmas) for UK+US documents. "
            "min_cluster_size chosen as ~3–5% of dataset size (~600 docs). "
            "min_samples set to 8 to require a reasonable local density "
            "without being overly strict, balancing noise and cluster formation."
        ),
    }

    save_clustering_result(hdbscan_output_dir, labels, "HDBSCAN", params)

    print("\n✅ HDBSCAN clustering finished and saved.")


if __name__ == "__main__":
    main()
