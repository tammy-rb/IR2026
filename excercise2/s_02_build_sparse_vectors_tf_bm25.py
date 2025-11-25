import os
import glob
import json

from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import numpy as np

# =========================
# Configuration
# =========================

LEMMA_DIR = "lemmas"
CLEAN_DIR = "clean_docs"

BRITISH = "british_parliament_debates"
US = "US_congress_debates"

# Output folders for both lemmas and clean documents
VECTORS_LEMMAS_DIR = "vectors/BM25_lemmas"
VECTORS_CLEAN_DIR = "vectors/BM25_clean"


# =========================
# Loaders
# =========================

def load_documents(doc_dir, file_pattern):
    """
    Load all documents from the specified directory matching the pattern.

    Returns:
        docs: list of text strings
        filenames: list of basenames
    """
    docs = []
    filenames = []

    pattern = os.path.join(doc_dir, file_pattern)
    for path in sorted(glob.glob(pattern)):
        filenames.append(os.path.basename(path))
        with open(path, encoding="utf-8") as f:
            docs.append(f.read())

    print(f"Loaded {len(docs)} documents from {doc_dir}")
    return docs, filenames


def load_lemmatized_documents(corpus_name, lemma_root=LEMMA_DIR):
    """
    Load all lemmatized documents for a given corpus.
    Expects files under: lemma_root/corpus_name/*.lemma.txt
    """
    lemma_dir = os.path.join(lemma_root, corpus_name)
    return load_documents(lemma_dir, "*.lemma.txt")


def load_clean_documents(corpus_name, clean_root=CLEAN_DIR):
    """
    Load all cleaned documents for a given corpus.
    Expects files under: clean_root/corpus_name/*.clean.txt
    """
    clean_dir = os.path.join(clean_root, corpus_name)
    return load_documents(clean_dir, "*.clean.txt")


def _load_all_documents_generic(single_corpus_loader):
    """
    Helper: load ALL documents (UK + US) using a given loader function
    (either load_lemmatized_documents or load_clean_documents).

    Returns:
        docs_all:       list of document texts
        filenames_all:  list of "corpus/filename" strings
        labels_all:     list of "UK"/"US" labels
    """
    docs_all = []
    filenames_all = []
    labels_all = []

    corpora = [
        (BRITISH, "UK"),
        (US, "US"),
    ]

    for corpus_name, label in corpora:
        docs, filenames = single_corpus_loader(corpus_name)
        docs_all.extend(docs)
        filenames_all.extend([f"{corpus_name}/{fn}" for fn in filenames])
        labels_all.extend([label] * len(docs))

    print(f"Total documents loaded (UK + US): {len(docs_all)}")
    return docs_all, filenames_all, labels_all


def load_all_lemmatized_documents():
    """Load all lemmatized documents from both corpora (UK + US)."""
    return _load_all_documents_generic(load_lemmatized_documents)


def load_all_clean_documents():
    """Load all cleaned documents from both corpora (UK + US)."""
    return _load_all_documents_generic(load_clean_documents)


# =========================
# TF + BM25
# =========================

def build_tf_matrix(docs, min_df=3, max_df=0.8):
    """
    Build a TF (Term Frequency) matrix from documents using CountVectorizer.

    Args:
        docs (list): List of document texts
        min_df (int): Minimum document frequency for terms
        max_df (float): Maximum document frequency for terms (as ratio)

    Returns:
        tf_matrix: sparse TF matrix
        vectorizer: fitted CountVectorizer
    """
    vectorizer = CountVectorizer(
        input="content",
        analyzer="word",
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
    )

    tf_matrix = vectorizer.fit_transform(docs)

    print(f"Built TF matrix with shape: {tf_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    return tf_matrix, vectorizer


def save_tf_results(tf_matrix, vectorizer, filenames, labels, vectors_dir):
    """
    Save TF matrix, filenames, vocabulary, and labels to files.

    Files written:
      - sparse_TF_matrix.npz
      - filenames.json
      - vocabulary.json
      - labels.json
    """
    os.makedirs(vectors_dir, exist_ok=True)

    # Save sparse TF matrix
    sparse.save_npz(os.path.join(vectors_dir, "sparse_TF_matrix.npz"), tf_matrix)

    # Save filenames (with corpus prefix)
    with open(os.path.join(vectors_dir, "filenames.json"), "w", encoding="utf-8") as f:
        json.dump(filenames, f)

    # Save vocabulary term -> column index
    with open(os.path.join(vectors_dir, "vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump(vectorizer.vocabulary_, f)

    # Save labels (for evaluation later)
    with open(os.path.join(vectors_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f)

    print(f"Saved TF matrix, vocabulary, filenames and labels to {vectors_dir}")


def bm25_weight(tf_csr, k1=1.6, b=0.75):
    """
    Calculate BM25 weights from a TF matrix using the Okapi BM25 formula.

    Args:
        tf_csr: sparse TF matrix in CSR format

    Returns:
        bm25_matrix: sparse BM25-weighted matrix
    """
    tf_csr = tf_csr.tocsr().astype(float)
    N, n_terms = tf_csr.shape

    # Document frequency for each term
    df = np.bincount(tf_csr.indices, minlength=n_terms)

    # IDF for each term
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

    # Document length and average document length
    dl = np.asarray(tf_csr.sum(axis=1)).ravel()
    avgdl = dl.mean()

    rows, cols = tf_csr.nonzero()
    data = tf_csr.data
    new_data = np.empty_like(data)

    for i in range(len(data)):
        doc = rows[i]
        term = cols[i]
        tf = data[i]

        denom = tf + k1 * (1 - b + b * dl[doc] / avgdl)
        new_data[i] = idf[term] * (tf * (k1 + 1)) / denom

    bm25_matrix = sparse.csr_matrix((new_data, (rows, cols)), shape=tf_csr.shape)

    print(f"Built BM25 matrix with shape: {bm25_matrix.shape}")
    print(f"BM25 parameters: k1={k1}, b={b}")
    return bm25_matrix


def save_bm25_matrix(bm25_matrix, vectors_dir):
    """
    Save BM25 matrix to file as bm25_okapi.npz.
    """
    os.makedirs(vectors_dir, exist_ok=True)
    sparse.save_npz(os.path.join(vectors_dir, "bm25_okapi.npz"), bm25_matrix)
    print(f"Saved BM25 matrix to {vectors_dir}")


# =========================
# Main builder
# =========================

def build_joint_bm25(doc_loader, vectors_dir, dataset_name, source_type):
    """
    Build ONE joint BM25 matrix for all documents (UK + US).

    Args:
        doc_loader: function that returns (docs, filenames, labels)
        vectors_dir (str): Output directory for vectors
        dataset_name (str): Name for logging (e.g., "Lemmas" or "Clean")
        source_type (str): Description for logging ("lemmatized"/"clean")
    """
    print(f"\n=== Building joint BM25 for {dataset_name} (source: {source_type}) ===")

    docs, filenames, labels = doc_loader()

    if not docs:
        print(f"No documents found for {dataset_name}! Aborting.")
        return

    tf_matrix, vectorizer = build_tf_matrix(docs)
    save_tf_results(tf_matrix, vectorizer, filenames, labels, vectors_dir)

    bm25_matrix = bm25_weight(tf_matrix)
    save_bm25_matrix(bm25_matrix, vectors_dir)

    print(f"\n✅ Joint BM25 matrix for {dataset_name} built and saved to: {vectors_dir}")


def main():
    print("Starting BM25 construction for UK + US (lemmas and clean)...")

    # Build BM25 for lemmatized documents
    build_joint_bm25(
        doc_loader=load_all_lemmatized_documents,
        vectors_dir=VECTORS_LEMMAS_DIR,
        dataset_name="Lemmas",
        source_type="lemmatized",
    )

    # Build BM25 for clean documents
    build_joint_bm25(
        doc_loader=load_all_clean_documents,
        vectors_dir=VECTORS_CLEAN_DIR,
        dataset_name="Clean",
        source_type="clean",
    )

    print("\n🎉 Done! BM25 matrices built for both lemmas and clean documents!")


if __name__ == "__main__":
    main()
