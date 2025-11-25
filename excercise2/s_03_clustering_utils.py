import os
import json

import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances


# ============================
# Paths
# ============================

# BM25 vector directories
VECTORS_LEMMAS_DIR = "vectors/BM25_lemmas"
VECTORS_CLEAN_DIR = "vectors/BM25_clean"

# Root folder for clustering results (subfolders per algorithm)
CLUSTERS_ROOT_DIR = "clusters"

# Label used to mark noise points in clustering outputs
NOISE_LABEL = -1


# ============================
# Helpers
# ============================

def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _as_csr(matrix):
    """
    Ensure matrix is CSR (for efficient row slicing) without changing dense input.
    """
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return matrix


# ============================
# Load BM25 + metadata
# ============================

def load_bm25_and_metadata(vectors_dir):
    """
    Load BM25 matrix and associated metadata from the given directory.

    Expected files:
      - bm25_okapi.npz   : sparse BM25 matrix
      - vocabulary.json  : mapping term -> column index
      - filenames.json   : list of document filenames (prefixed with corpus)
      - labels.json      : list of "UK"/"US" labels, one per document

    Returns:
        bm25_matrix: sparse CSR matrix
        feature_names: list where feature_names[col] = term
        filenames: list of filenames (same order as rows)
        labels: list of labels (same order as rows)
    """
    bm25_path = os.path.join(vectors_dir, "bm25_okapi.npz")
    vocab_path = os.path.join(vectors_dir, "vocabulary.json")
    filenames_path = os.path.join(vectors_dir, "filenames.json")
    labels_path = os.path.join(vectors_dir, "labels.json")

    if not os.path.exists(bm25_path):
        raise FileNotFoundError(f"BM25 matrix not found: {bm25_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    if not os.path.exists(filenames_path):
        raise FileNotFoundError(f"Filenames file not found: {filenames_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Load BM25 matrix
    bm25_matrix = sparse.load_npz(bm25_path)

    # Load vocabulary (term -> column index)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Convert vocab dict to a list sorted by index: feature_names[col] = term
    feature_names = [None] * len(vocab)
    for term, idx in vocab.items():
        feature_names[idx] = term

    # Load filenames
    with open(filenames_path, "r", encoding="utf-8") as f:
        filenames = json.load(f)

    # Load labels ("UK"/"US")
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    print(f"Loaded BM25 matrix from {vectors_dir} with shape {bm25_matrix.shape}")
    return bm25_matrix, feature_names, filenames, labels


# ============================
# Cluster utilities
# ============================

def compute_cluster_centroids(matrix, labels, noise_label=NOISE_LABEL, normalize=True):
    """
    Compute cluster centroids (mean vectors) for each non-noise cluster.

    Args:
        matrix: BM25 matrix (n_docs x n_terms), sparse or dense.
        labels: 1D array of cluster labels, length = n_docs.
        noise_label: label used for noise (default: -1).
        normalize: if True, L2-normalize each centroid (good for cosine).

    Returns:
        centroids: 2D numpy array of shape (n_clusters, n_features).
        cluster_ids: list of cluster labels in the same order as centroids.
    """
    matrix = _as_csr(matrix)
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)
    cluster_ids = sorted(int(c) for c in unique_labels if c != noise_label)

    if not cluster_ids:
        raise ValueError("No non-noise clusters found – cannot compute centroids.")

    centroids = []

    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue

        cluster_matrix = matrix[idx]
        centroid = cluster_matrix.mean(axis=0)

        # Convert from matrix to 1D ndarray
        if sparse.issparse(centroid):
            centroid = centroid.A1
        else:
            centroid = np.asarray(centroid).ravel()

        if normalize:
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

        centroids.append(centroid)

    centroids = np.vstack(centroids)
    return centroids, cluster_ids


def get_cluster_size_dict(labels):
    """
    Return a plain-Python dict {label: count} for cluster sizes.
    """
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def relabel_clusters_consecutively(labels, noise_label=NOISE_LABEL, start_from=0):
    """
    Relabel clusters to start_from..start_from+K-1, keeping `noise_label` unchanged.

    Example:
        labels = [-1, 10, 10, 3] -> new_labels = [-1, 0, 0, 1],
        mapping = {3: 1, 10: 0}

    Returns:
        new_labels: 1D numpy array of relabeled clusters.
        mapping: dict {old_label: new_label}
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    cluster_ids = sorted(int(c) for c in unique_labels if c != noise_label)
    mapping = {old: (start_from + i) for i, old in enumerate(cluster_ids)}

    new_labels = labels.copy()
    for old, new in mapping.items():
        new_labels[labels == old] = new

    return new_labels, mapping


def assign_noise_to_nearest_cluster(
    matrix,
    labels,
    metric="cosine",
    noise_label=NOISE_LABEL,
    max_distance=None,
    normalize_centroids=True,
    inplace=False,
    verbose=True,
):
    """
    Reassign noise points (label == noise_label) to the nearest non-noise cluster.

    Steps:
      1. Compute centroids for all non-noise clusters.
      2. For each noise doc, compute distance to each centroid.
      3. Assign it to the nearest centroid (optionally only if distance <= max_distance).

    Args:
        matrix: BM25 matrix (n_docs x n_terms), sparse or dense.
        labels: 1D array of cluster labels (length = n_docs).
        metric: distance metric for pairwise_distances (default: "cosine").
        noise_label: label used for noise.
        max_distance: if not None, only reassign noise points whose nearest
                      distance <= max_distance; others remain noise.
        normalize_centroids: if True, L2-normalize centroids (recommended for cosine).
        inplace: if True, modify `labels` in-place; else work on a copy.
        verbose: if True, print a short summary.

    Returns:
        new_labels: 1D array of updated labels (same type as input if inplace=False).
    """
    matrix = _as_csr(matrix)
    labels_arr = labels if inplace else np.array(labels, copy=True)

    labels_arr = np.asarray(labels_arr)
    noise_mask = labels_arr == noise_label
    n_noise = int(np.sum(noise_mask))

    if n_noise == 0:
        if verbose:
            print("No noise points found – nothing to reassign.")
        return labels_arr

    # 1. Compute centroids of non-noise clusters
    centroids, cluster_ids = compute_cluster_centroids(
        matrix, labels_arr, noise_label=noise_label, normalize=normalize_centroids
    )

    # 2. Distances from each noise point to each centroid
    noise_indices = np.where(noise_mask)[0]
    X_noise = matrix[noise_indices]

    if verbose:
        print(
            f"Reassigning {n_noise} noise points "
            f"to {len(cluster_ids)} non-noise clusters using {metric} distance..."
        )

    dist_matrix = pairwise_distances(X_noise, centroids, metric=metric)
    best_idx = np.argmin(dist_matrix, axis=1)
    best_dist = dist_matrix[np.arange(best_idx.size), best_idx]
    best_cluster_ids = np.array(cluster_ids)[best_idx]

    # 3. Optional distance threshold
    if max_distance is not None:
        reassign_mask = best_dist <= max_distance
    else:
        reassign_mask = np.ones_like(best_dist, dtype=bool)

    n_reassigned = int(np.sum(reassign_mask))
    labels_arr[noise_indices[reassign_mask]] = best_cluster_ids[reassign_mask]

    if verbose:
        if max_distance is None:
            print(f"  → Reassigned ALL {n_reassigned}/{n_noise} noise points.")
        else:
            print(
                f"  → Reassigned {n_reassigned}/{n_noise} noise points "
                f"(max_distance={max_distance})."
            )

    return labels_arr


# ============================
# Save clustering result
# ============================

def save_clustering_result(output_dir, labels_array, algorithm_name, params=None):
    """
    Save clustering labels (and optionally algorithm parameters) to disk.

    Files written in output_dir:
      - cluster_labels.npy
      - clustering_meta.json
    """
    ensure_dir(output_dir)

    labels_path = os.path.join(output_dir, "cluster_labels.npy")
    np.save(labels_path, labels_array)

    meta = {
        "algorithm": algorithm_name,
        "n_documents": int(len(labels_array)),
    }
    if params is not None:
        meta["params"] = params

    meta_path = os.path.join(output_dir, "clustering_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved clustering labels to {labels_path}")
    print(f"Saved clustering metadata to {meta_path}")
