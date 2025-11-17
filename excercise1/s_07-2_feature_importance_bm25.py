import os
import json

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.feature_selection import mutual_info_classif, chi2


# ============================
# Folder paths
# ============================

LEMMA_DIR = "lemmas"
VECTORS_LEMMAS_DIR = "vectors/BM25_lemmas"

CLEAN_DIR = "clean_docs"
VECTORS_CLEAN_DIR = "vectors/BM25_words"

OUTPUT_EXCEL = "feature_importance_bm25.xlsx"
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


def load_cluster_labels(vectors_dir, filename=CLUSTER_LABELS_FILENAME):
    """
    Loads precomputed cluster labels (one label per document) from the vectors directory.
    """
    path = os.path.join(vectors_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cluster labels not found in {path}. "
            f"Run the clustering script first."
        )
    y = np.load(path)
    print(f"Loaded cluster labels from {path} with shape {y.shape}")
    return y


# ============================
# Compute Information Gain + Chi-Squared
# ============================

def compute_information_gain_and_chi2(bm25_matrix, y, feature_names):
    """
    Computes two feature-importance metrics for each column (term):

      - Information Gain (implemented via mutual_info_classif)
      - Chi-Squared statistic (chi2)

    bm25_matrix: sparse BM25 document-term matrix
    y: cluster labels
    feature_names: list of terms (ordered by column index)
    """
    # Convert BM25 to binary presence/absence (important for IG and chi2)
    bm25_binary = bm25_matrix.copy().tocsr()
    bm25_binary.data = np.ones_like(bm25_binary.data)

    print("Computing Information Gain (mutual information)...")
    ig_scores = mutual_info_classif(
        bm25_binary, y,
        discrete_features=True,
        random_state=42
    )

    print("Computing Chi-Squared scores...")
    chi2_scores, p_values = chi2(bm25_binary, y) 

    # Create sorted dataframes
    df_ig = pd.DataFrame({
        "feature": feature_names,
        "information_gain": ig_scores
    }).sort_values("information_gain", ascending=False)

    df_chi2 = pd.DataFrame({
        "feature": feature_names,
        "chi2_score": chi2_scores
    }).sort_values("chi2_score", ascending=False)

    return df_ig, df_chi2


# ============================
# Run full process and produce Excel output
# ============================

def process_dataset(vectors_dir, dataset_label):
    """
    Processes one dataset (e.g., lemmas or clean words):

      1. Loads BM25 + vocabulary
      2. Loads precomputed cluster labels
      3. Computes Information Gain + Chi-Squared

    Returns:
        df_ig: dataframe of Information Gain scores
        df_chi2: dataframe of Chi-Squared scores
    """
    print(f"\n=== Processing dataset: {dataset_label} ===")
    bm25_matrix, feature_names, filenames = load_bm25_and_vocab(vectors_dir)
    y = load_cluster_labels(vectors_dir)
    df_ig, df_chi2 = compute_information_gain_and_chi2(bm25_matrix, y, feature_names)
    return df_ig, df_chi2

def save_feature_tables_to_excel(output_path, df_ig, df_chi2):
    """
    Saves IG and Chi-Squared DataFrames into a 2-sheet Excel file.

    output_path : str
        Full path to the output .xlsx file
    df_ig : DataFrame
        Information Gain table
    df_chi2 : DataFrame
        Chi-Squared score table
    """
    print(f"Saving results to: {output_path}")
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df_ig.to_excel(writer, sheet_name="InformationGain", index=False)
        df_chi2.to_excel(writer, sheet_name="ChiSquared", index=False)


def main():

    # ========== 1) Lemmas ==========
    print("\n=== Processing Lemmas ===")
    df_ig_lemmas, df_chi2_lemmas = process_dataset(
        VECTORS_LEMMAS_DIR,
        dataset_label="TFIDF_Lemm"
    )

    lemmas_excel_path = os.path.join(VECTORS_LEMMAS_DIR, "feature_importance_lemmas.xlsx")
    save_feature_tables_to_excel(lemmas_excel_path, df_ig_lemmas, df_chi2_lemmas)

    # ========== 2) Words (clean) ==========
    print("\n=== Processing Words ===")
    df_ig_words, df_chi2_words = process_dataset(
        VECTORS_CLEAN_DIR,
        dataset_label="TFIDF_Word"
    )

    words_excel_path = os.path.join(VECTORS_CLEAN_DIR, "feature_importance_words.xlsx")
    save_feature_tables_to_excel(words_excel_path, df_ig_words, df_chi2_words)

    print("\n✅ Done! Each dataset saved into its own Excel file.")


if __name__ == "__main__":
    main()
