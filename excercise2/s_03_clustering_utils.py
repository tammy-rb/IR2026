import os
import json

import numpy as np
from scipy import sparse


# ============================
# Paths
# ============================

# BM25 vector directories
VECTORS_LEMMAS_DIR = "vectors/BM25_lemmas"
VECTORS_CLEAN_DIR = "vectors/BM25_clean"

# Root folder for clustering results (subfolders per algorithm)
CLUSTERS_ROOT_DIR = "clusters"


# ============================
# Helpers
# ============================

def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


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
