import os
import json

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import mutual_info_classif, chi2


# ============================
# Folder paths
# ============================

LEMMA_DIR = "lemmas"
VECTORS_LEMMAS_DIR = "vectors/BM25_lemmas"

CLEAN_DIR = "clean_docs"
VECTORS_CLEAN_DIR = "vectors/BM25_words"

OUTPUT_EXCEL = "feature_importance_bm25.xlsx"


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

    # Load sparse matrix
    X = sparse.load_npz(bm25_path)

    # Load vocabulary (term -> index)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Load filenames
    with open(filenames_path, "r", encoding="utf-8") as f:
        filenames = json.load(f)

    # Convert vocabulary dict to list sorted by index
    feature_names = [None] * len(vocab)
    for term, idx in vocab.items():
        feature_names[idx] = term

    print(f"Loaded BM25 matrix from {vectors_dir} with shape {X.shape}")
    return X, feature_names, filenames


# ============================
# Create pseudo-labels using clustering
# ============================

def build_pseudo_labels_by_clustering(X, n_clusters=5, random_state=42):
    """
    Creates pseudo-labels for documents using MiniBatchKMeans clustering.

    X: sparse BM25 matrix (documents × terms)

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
    y = kmeans.fit_predict(X)
    print("Clustering done.")
    return y


# ============================
# Compute Information Gain + Chi-Squared
# ============================

def compute_information_gain_and_chi2(X, y, feature_names):
    """
    Computes two feature-importance metrics for each column (term):

      - Information Gain (implemented via mutual_info_classif)
      - Chi-Squared statistic (chi2)

    X: sparse BM25 document-term matrix
    y: cluster labels
    feature_names: list of terms (ordered by column index)
    """
    # Convert X to binary presence/absence (important for IG and chi2)
    X_bin = X.copy().tocsr()
    X_bin.data = np.ones_like(X_bin.data)

    print("Computing Information Gain (mutual information)...")
    ig_scores = mutual_info_classif(
        X_bin, y,
        discrete_features=True,
        random_state=42
    )

    print("Computing Chi-Squared scores...")
    chi2_scores, chi2_pvalues = chi2(X_bin, y)

    # Create sorted dataframes
    df_ig = pd.DataFrame({
        "feature": feature_names,
        "information_gain": ig_scores
    }).sort_values("information_gain", ascending=False)

    df_chi2 = pd.DataFrame({
        "feature": feature_names,
        "chi2_score": chi2_scores,
        "p_value": chi2_pvalues
    }).sort_values("chi2_score", ascending=False)

    return df_ig, df_chi2


# ============================
# Run full process and produce Excel output
# ============================

def process_dataset(vectors_dir, dataset_label, n_clusters=5):
    """
    Processes one dataset (e.g., lemmas or clean words):

      1. Loads BM25 + vocabulary
      2. Creates cluster-based pseudo-labels
      3. Computes Information Gain + Chi-Squared

    Returns:
        df_ig: dataframe of Information Gain scores
        df_chi2: dataframe of Chi-Squared scores
    """
    print(f"\n=== Processing dataset: {dataset_label} ===")
    X, feature_names, filenames = load_bm25_and_vocab(vectors_dir)
    y = build_pseudo_labels_by_clustering(X, n_clusters=n_clusters)
    df_ig, df_chi2 = compute_information_gain_and_chi2(X, y, feature_names)
    return df_ig, df_chi2


def main():
    results = {}

    # 1) Lemmas (BM25 on lemmatized documents)
    df_ig_lemmas, df_chi2_lemmas = process_dataset(
        VECTORS_LEMMAS_DIR,
        dataset_label="TFIDF_Lemm",
        n_clusters=5
    )
    results["TFIDF_Lemm_InformationGain"] = df_ig_lemmas
    results["TFIDF_Lemm_ChiSquared"] = df_chi2_lemmas

    # 2) Words (BM25 on cleaned documents)
    df_ig_words, df_chi2_words = process_dataset(
        VECTORS_CLEAN_DIR,
        dataset_label="TFIDF_Word",
        n_clusters=5
    )
    results["TFIDF_Word_InformationGain"] = df_ig_words
    results["TFIDF_Word_ChiSquared"] = df_chi2_words

    # Save everything to a single Excel file (4 sheets)
    print(f"\nSaving all results to Excel: {OUTPUT_EXCEL}")
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="xlsxwriter") as writer:
        for sheet_name, df in results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("✅ Done! Feature importance tables saved to", OUTPUT_EXCEL)


if __name__ == "__main__":
    main()
