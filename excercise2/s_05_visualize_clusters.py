"""
s_05_visualize_clusters.py

Create 2D visualizations (UMAP) of:
    - True labels (UK/US)
    - Cluster assignments for each method (KMeans, DBSCAN, HDBSCAN, GMM).

Assumes:
- BM25 + labels were built by BM25 script and can be loaded via
  `load_bm25_and_metadata(VECTORS_LEMMAS_DIR)`.
- Each clustering method saved results under:
    clusters/
        kmeans/
            labels.npy
        dbscan/
            labels.npy
        hdbscan/
            labels.npy
        gmm/
            labels.npy

Requires:
    pip install umap-learn matplotlib seaborn
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from s_03_clustering_utils import (
    VECTORS_LEMMAS_DIR,
    CLUSTERS_ROOT_DIR,
    load_bm25_and_metadata,
)


# ============================
# Helpers for loading data
# ============================

def load_bm25_and_true_labels():
    """
    Load BM25 matrix and true labels (UK/US) via the BM25 metadata.
    Returns:
        bm25_matrix: sparse matrix (n_docs x n_terms)
        true_labels: np.ndarray of shape (n_docs,)
    """
    bm25_matrix, feature_names, filenames, true_labels = load_bm25_and_metadata(
        VECTORS_LEMMAS_DIR
    )
    true_labels = np.array(true_labels)
    return bm25_matrix, true_labels


def load_cluster_labels(method_name):
    """
    Load cluster labels for a given method.

    Expects:
        clusters/<method_name>/cluster_labels.npy

    Returns:
        labels: np.ndarray of shape (n_docs,)
    """
    method_dir = os.path.join(CLUSTERS_ROOT_DIR, method_name)
    labels_path = os.path.join(method_dir, "cluster_labels.npy")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels file for {method_name}: {labels_path}")

    labels = np.load(labels_path)
    return labels


# ============================
# UMAP visualization
# ============================

def compute_umap_embedding(bm25_matrix, n_components=2, random_state=42, metric="cosine"):
    """
    Compute a 2D UMAP embedding from the BM25 matrix.
    n_components: number of UMAP dimensions (2 for visualization)
    metric: distance metric for UMAP (e.g. "cosine", "euclidean
    random_state: for reproducibility

    returns:
        embedding: np.ndarray of shape (n_docs, n_components) (2D coordinates)
    """
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for UMAP visualization. "
            "Install it via: pip install umap-learn"
        ) from e

    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(bm25_matrix)
    return embedding


def plot_umap_scatter(
    embedding,
    labels,
    title,
    palette="tab10",
    save_path=None,
    noise_label=-1,
):
    """
    Create a 2D scatter plot of the UMAP embedding colored by 'labels'.

    If noise_label is provided (e.g. -1 for DBSCAN/HDBSCAN),
    noise points will be plotted with a distinct style.
    """
    x = embedding[:, 0]
    y = embedding[:, 1]

    plt.figure(figsize=(8, 6)) # size of figure
    unique_labels = np.unique(labels)

    # Separate noise if needed
    if (noise_label is not None) and (noise_label in unique_labels):
        # Non-noise
        mask_core = labels != noise_label
        sns.scatterplot(
            x=x[mask_core],
            y=y[mask_core],
            hue=labels[mask_core].astype(str), # color by cluster
            palette=palette, 
            s=30, # size of points
            alpha=0.8, # transparency
            legend="full", # 
        )

        # Noise as gray crosses
        mask_noise = labels == noise_label
        if np.any(mask_noise):
            plt.scatter(
                x[mask_noise],
                y[mask_noise],
                c="lightgray",
                s=20,
                alpha=0.6,
                marker="x",
                label="noise",
            )
        plt.legend()
    else:
        sns.scatterplot(
            x=x,
            y=y,
            hue=labels.astype(str),
            palette=palette,
            s=30,
            alpha=0.8,
            legend="full",
        )

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"  → Saved plot: {save_path}")

    plt.close()


def visualize_all_methods_umap(bm25_matrix, true_labels, methods):
    """
    Create UMAP 2D visualizations for:
        - True labels (UK/US)
        - Each clustering method's labels

    Args:
        bm25_matrix: sparse BM25 matrix
        true_labels: array of UK/US labels
        methods: list of method names to visualize (e.g. ["kmeans","dbscan","hdbscan","gmm"])
    """
    print("\n[UMAP] Computing 2D embedding from BM25 (this may take a bit)...")
    embedding = compute_umap_embedding(bm25_matrix, n_components=2, metric="cosine")
    print("  → UMAP embedding computed.")

    vis_dir = os.path.join(CLUSTERS_ROOT_DIR, "visualizations")

    # 1) Plot true labels
    plot_umap_scatter(
        embedding,
        labels=true_labels,
        title="UMAP - True Labels (UK vs US)",
        save_path=os.path.join(vis_dir, "umap_true_labels.png"),
        noise_label=None,
    )

    # 2) Plot each method's clustering
    for method in methods:
        method_dir = os.path.join(CLUSTERS_ROOT_DIR, method)
        labels_path = os.path.join(method_dir, "cluster_labels.npy")
        if not os.path.exists(labels_path):
            print(f"[UMAP] Skipping '{method}' (no cluster_labels.npy found).")
            continue

        labels = load_cluster_labels(method)

        has_noise = np.any(labels == -1)
        noise_label = -1 if has_noise else None

        title = f"UMAP - Clusters ({method.upper()})"
        save_path = os.path.join(vis_dir, f"umap_{method.lower()}.png")

        plot_umap_scatter(
            embedding,
            labels=labels,
            title=title,
            save_path=save_path,
            noise_label=noise_label,
        )


# ============================
# Main
# ============================

def main():
    print("=" * 70)
    print("UMAP VISUALIZATION OF CLUSTERS (BM25 Lemmas)")
    print("=" * 70)

    # 1. Load BM25 + true labels
    print("\n[1] Loading BM25 matrix and true labels...")
    bm25_matrix, true_labels = load_bm25_and_true_labels()
    print(f"  → Loaded {len(true_labels)} labels.")
    print(f"  → Classes: {np.unique(true_labels)}")

    # 2. Visualize methods (only those that exist)
    methods = ["kmeans", "dbscan", "hdbscan", "gmm"]
    visualize_all_methods_umap(bm25_matrix, true_labels, methods)


if __name__ == "__main__":
    main()
