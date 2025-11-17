import os
import json

import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans


# ============================
# Folder paths
# ============================

LEMMA_DIR = "lemmas"
VECTORS_LEMMAS_DIR = "vectors/BM25_lemmas"

CLEAN_DIR = "clean_docs"
VECTORS_CLEAN_DIR = "vectors/BM25_words"

# File name for saving cluster labels inside each vectors directory
CLUSTER_LABELS_FILENAME = "cluster_labels_k5.npy"


# ============================
# Load BM25 matrix and vocabulary
# ============================

def load_bm25_and_vocab(vectors_dir):
    """
    Loads the BM25 matrix, vocabulary, and filenames from the given directory.

    Expected files in the directory:
      - bm25_okapi.npz           (sparse BM25 matrix)
      - vocabulary.json          (mapping: term -> column index)
      - filenames.json           (list of document filenames)
    """
    bm25_path = os.path.join(vectors_dir, "bm25_okapi.npz")
    vocab_path = os.path.join(vectors_dir, "vocabulary.json")
    filenames_path = os.path.join(vectors_dir, "filenames.json")

    if not os.path.exists(bm25_path):
        raise FileNotFoundError(f"BM25 matrix not found in {bm25_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found in {vocab_path}")
    if not os.path.exists(filenames_path):
        raise FileNotFoundError(f"Filenames file not found in {filenames_path}")

    # Load sparse BM25 matrix
    bm25_matrix = sparse.load_npz(bm25_path)

    # Load vocabulary (term -> index)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Load filenames
    with open(filenames_path, "r", encoding="utf-8") as f:
        filenames = json.load(f)

    # Convert vocabulary dict to list sorted by index: feature_names[col] = term
    feature_names = [None] * len(vocab)
    for term, idx in vocab.items():
        feature_names[idx] = term

    print(f"Loaded BM25 matrix from {vectors_dir} with shape {bm25_matrix.shape}")
    return bm25_matrix, feature_names, filenames


# ============================
# Create pseudo-labels using clustering
# ============================

def build_pseudo_labels_by_clustering(bm25_matrix, n_clusters=5, random_state=42):
    """
    Creates pseudo-labels for documents using MiniBatchKMeans clustering.

    bm25_matrix: sparse BM25 matrix (documents × terms)

    Returns:
        y: cluster assignment for each document (array of length n_docs)
    """
    print(f"Clustering documents into {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=256,
        max_iter=100
    )
    y = kmeans.fit_predict(bm25_matrix)
    print("Clustering done.")
    return y


def save_cluster_labels(vectors_dir, y, filename=CLUSTER_LABELS_FILENAME):
    """
    Save cluster labels (one label per document) into the vectors directory.
    """
    out_path = os.path.join(vectors_dir, filename)
    np.save(out_path, y)
    print(f"Saved cluster labels to {out_path}")


def process_dataset_for_clustering(vectors_dir, dataset_label, n_clusters=5):
    """
    For a given dataset (lemmas / words):
      1. Loads BM25 matrix + vocab (for shape info & consistency)
      2. Runs clustering to create pseudo-labels
      3. Saves the cluster labels to disk
    """
    print(f"\n=== Clustering dataset: {dataset_label} ===")
    bm25_matrix, feature_names, filenames = load_bm25_and_vocab(vectors_dir)
    y = build_pseudo_labels_by_clustering(bm25_matrix, n_clusters=n_clusters)
    save_cluster_labels(vectors_dir, y)


def main():
    # 1) Lemmas (BM25 on lemmatized documents)
    process_dataset_for_clustering(
        VECTORS_LEMMAS_DIR,
        dataset_label="TFIDF_Lemm",
        n_clusters=5
    )

    # 2) Words (BM25 on cleaned documents)
    process_dataset_for_clustering(
        VECTORS_CLEAN_DIR,
        dataset_label="TFIDF_Word",
        n_clusters=5
    )

    print("\n✅ Done! Cluster labels saved for all datasets.")


if __name__ == "__main__":
    main()
