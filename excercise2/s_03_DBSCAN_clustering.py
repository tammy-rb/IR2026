import os

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

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


# Toggle post-processing of noise points
POSTPROCESS_ASSIGN_NOISE = True
NOISE_MAX_DISTANCE = None  # If None: assign every noise point to the nearest cluster


# ============================
# Heuristic for DBSCAN params
# ============================

def estimate_eps_with_kdist(matrix, min_samples=5, quantile=0.70):
    """
    Estimate a reasonable eps for DBSCAN using the k-distance heuristic.

    Steps:
      1. Compute cosine distances between all document pairs.
      2. For each document, sort distances to neighbors.
      3. Take the distance to the k-th neighbor (k = min_samples) -> k-dist value.
      4. Take a high quantile (e.g. 80%) of these k-dist values as eps.

    Intuition:
      - Points in dense regions have smaller k-dist.
      - Points in sparser regions / near cluster borders have larger k-dist.
      - A high quantile of k-dist approximates the "elbow" in the k-distance plot
        and serves as a threshold for density (eps).

    Args:
        matrix: BM25 matrix (n_docs x n_terms), sparse or dense.
        min_samples: DBSCAN's min_samples parameter.
        quantile: which quantile of k-distances to use for eps (0..1).

    Returns:
        eps (float), k_distances (1D array of length n_docs)
    """
    print(f"  → Computing pairwise cosine distances for k-distance heuristic...")
    # pairwise_distances returns a dense matrix (n_docs x n_docs)
    dist_matrix = pairwise_distances(matrix, metric="cosine")

    # Sort distances in each row; first element is 0 (self-distance)
    sorted_dists = np.sort(dist_matrix, axis=1)

    # k-distance: distance to the k-th neighbor (index min_samples, since 0 is self)
    k_distances = sorted_dists[:, min_samples]

    eps = float(np.quantile(k_distances, quantile))
    print(
        f"  → Estimated eps = {eps:.4f} "
        f"(from {quantile*100:.0f}th percentile of {min_samples}-nearest neighbor distances)"
    )
    return eps, k_distances


# ============================
# DBSCAN clustering
# ============================

def run_dbscan_clustering(bm25_matrix, eps, min_samples=5):
    """
    Run DBSCAN on the given BM25 matrix using cosine distance.

    Args:
        bm25_matrix: sparse BM25 matrix (n_docs x n_terms)
        eps (float): neighborhood radius (in cosine distance)
        min_samples (int): minimum number of points to form a dense region

    Returns:
        labels: 1D array of cluster assignments (length = n_docs)
                noise points are labeled as NOISE_LABEL.
    """
    print(f"  → Running DBSCAN (eps={eps:.4f}, min_samples={min_samples}, metric=cosine)...")
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
        n_jobs=-1,  # use all cores if available
    )
    labels = dbscan.fit_predict(bm25_matrix)
    print(f"  → DBSCAN completed")
    return labels, dbscan


def main():
    print("\n" + "=" * 70)
    print("DBSCAN CLUSTERING ON BM25 VECTORS (Lemmas, UK + US)")
    print("=" * 70)

    # 1. Load BM25 and metadata for lemmas
    print("\n[1] Loading BM25 matrix and metadata...")
    bm25_matrix, feature_names, filenames, true_labels = load_bm25_and_metadata(
        VECTORS_LEMMAS_DIR
    )
    n_docs = bm25_matrix.shape[0]
    n_features = bm25_matrix.shape[1]
    print(f"  → Documents: {n_docs}")
    print(f"  → Features: {n_features}")

    # -------------------------
    # Hyper-parameter choice
    # -------------------------
    print("\n[2] Estimating DBSCAN parameters...")
    
    # We choose min_samples = 5 (a common rule of thumb: small value in high dimension)
    min_samples = 5
    print(f"  → min_samples: {min_samples}")

    # Use k-distance heuristic to estimate eps:
    # - Compute cosine distance
    # - Look at the 5-nearest-neighbor distance (k = min_samples)
    # - Take the 80th percentile as eps (captures most dense regions, ignores extreme outliers)
    eps, k_distances = estimate_eps_with_kdist(
        bm25_matrix,
        min_samples=min_samples,
        quantile=0.75,
    )

    # 2. Run DBSCAN with the estimated eps
    print("\n[3] Running DBSCAN clustering...")
    labels, dbscan_model = run_dbscan_clustering(
        bm25_matrix,
        eps=eps,
        min_samples=min_samples,
    )

    # Count original noise points before reassignment
    n_original_noise = count_noise_points(labels, noise_label=NOISE_LABEL)

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

    # Simple summary: cluster counts (including noise = NOISE_LABEL)
    print("\n[4] Clustering Results:")
    cluster_sizes = get_cluster_size_dict(labels)
    
    # Number of noise points (label == NOISE_LABEL)
    n_noise = cluster_sizes.get(NOISE_LABEL, 0)
    
    # Number of real clusters (excluding noise label)
    n_clusters = len([c for c in cluster_sizes.keys() if c != NOISE_LABEL])
    
    print(f"  → Total clusters found: {n_clusters}")
    print(f"  → Noise points: {n_noise} ({n_noise/n_docs*100:.1f}%)")
    print(f"  → Clustered points: {n_docs - n_noise} ({(n_docs - n_noise)/n_docs*100:.1f}%)")
    print(f"\n  Cluster distribution:")
    for label in sorted(cluster_sizes.keys()):
        size = cluster_sizes[label]
        label_str = "Noise" if label == NOISE_LABEL else f"Cluster {label}"
        print(f"    {label_str:>12}: {size:>4} documents ({size/n_docs*100:>5.1f}%)")

    # 3. Save clustering result under clusters/dbscan
    print("\n[5] Saving results...")
    dbscan_output_dir = os.path.join(CLUSTERS_ROOT_DIR, "dbscan")
    params = {
        "eps": eps,
        "min_samples": min_samples,
        "metric": "cosine",
        "n_jobs": -1,
        "n_documents": int(n_docs),
        "n_clusters": n_clusters,
        "n_noise_points": int(n_noise),
        "cluster_sizes": cluster_sizes,
        "heuristic": {
            "type": "k-distance percentile",
            "neighbor_k": min_samples,
            "quantile": 0.80,
            "description": (
                "eps chosen as 80th percentile of k-distances (k = min_samples) "
                "computed using cosine distance between BM25 vectors."
            ),
        },
        "noise_reassignment": {
            "enabled": POSTPROCESS_ASSIGN_NOISE,
            "max_distance": NOISE_MAX_DISTANCE,
            "original_noise_count": int(n_original_noise),
        },
    }
    save_clustering_result(dbscan_output_dir, labels, "DBSCAN", params)

    print("\n" + "=" * 70)
    print("✅ DBSCAN clustering completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
