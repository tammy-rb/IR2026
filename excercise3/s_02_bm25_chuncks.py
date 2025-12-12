# s_02_bm25_chunks.py
from __future__ import annotations

import os
import json
import glob
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Where chunking.py writes its outputs:
CHUNKS_DIR = os.path.join(BASE_DIR, "outputs", "chunks")

# Output directory for BM25 artifacts
BM25_OUT_DIR = os.path.join(BASE_DIR, "outputs", "bm25")

# Input JSONL files
CHUNKS_FIXED_JSONL = os.path.join(CHUNKS_DIR, "chunks_fixed.jsonl")
CHUNKS_SEM_JSONL = os.path.join(CHUNKS_DIR, "chunks_semantic.jsonl")

# BM25 params (Okapi)
BM25_K1 = 1.6
BM25_B = 0.75

# Vectorizer params (tune if needed)
STOP_WORDS = "english"
MIN_DF = 2
MAX_DF = 0.85


# ============================================================
# Helpers: IO
# ============================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries, where each dictionary
    represents a single chunk/document and contains its text and metadata
    (e.g., document ID, corpus label, offsets, and word count).
    """
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# BM25 implementation (from TF matrix)
# ============================================================

def build_tf_matrix(docs: List[str]) -> Tuple[sparse.csr_matrix, CountVectorizer]:
    """
    Build a sparse TF matrix over the chunk texts using CountVectorizer.
    """
    vectorizer = CountVectorizer(
        input="content",
        analyzer="word",
        stop_words=STOP_WORDS,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )
    tf = vectorizer.fit_transform(docs)
    tf = tf.tocsr().astype(float)
    return tf, vectorizer


def bm25_weight(tf_csr: sparse.csr_matrix, k1: float = BM25_K1, b: float = BM25_B) -> sparse.csr_matrix:
    """
    Okapi BM25 weighting from a TF matrix (CSR).
    """
    tf_csr = tf_csr.tocsr().astype(float)
    N, n_terms = tf_csr.shape

    # df: document frequency per term
    df = np.bincount(tf_csr.indices, minlength=n_terms)

    # idf
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

    # doc lengths
    dl = np.asarray(tf_csr.sum(axis=1)).ravel()
    avgdl = dl.mean() if dl.size else 0.0

    rows, cols = tf_csr.nonzero()
    data = tf_csr.data
    new_data = np.empty_like(data)

    for i in range(len(data)):
        doc = rows[i]
        term = cols[i]
        tf = data[i]
        denom = tf + k1 * (1.0 - b + b * (dl[doc] / (avgdl + 1e-12)))
        new_data[i] = idf[term] * (tf * (k1 + 1.0)) / (denom + 1e-12)

    return sparse.csr_matrix((new_data, (rows, cols)), shape=tf_csr.shape)


def bm25_query_vector(query: str, vectorizer: CountVectorizer) -> sparse.csr_matrix:
    """
    Build a TF vector for query in the same vocab space.
    """
    q_tf = vectorizer.transform([query]).tocsr().astype(float)
    return q_tf


def bm25_scores(bm25_matrix: sparse.csr_matrix, q_tf: sparse.csr_matrix) -> np.ndarray:
    """
    Score all docs: score(d) = BM25(d) dot TF(query).
    This is a common simple approach for Okapi-weighted document matrix.
    """
    # (N x V) dot (V x 1) => (N x 1)
    scores = bm25_matrix.dot(q_tf.T).toarray().ravel()
    return scores


# ============================================================
# Index builder
# ============================================================

def build_and_save_bm25(chunks_jsonl: str, out_subdir: str) -> None:
    ensure_dir(out_subdir)

    chunks = read_jsonl(chunks_jsonl)
    if not chunks:
        raise ValueError(f"No chunks found in: {chunks_jsonl}")

    # texts + metadata
    docs: List[str] = []
    meta: List[Dict[str, Any]] = []

    for c in chunks:
        docs.append(c["text"])
        meta.append({
            "doc_id": c.get("doc_id"),
            "source_path": c.get("source_path"),
            "corpus": c.get("corpus"),
            "chunking_method": c.get("chunking_method"),
            "chunk_index": c.get("chunk_index"),
            "start_char": c.get("start_char"),
            "end_char": c.get("end_char"),
            "num_words": c.get("num_words"),
        })

    print(f"Loaded {len(docs)} chunks from: {os.path.basename(chunks_jsonl)}")

    tf, vectorizer = build_tf_matrix(docs)
    print(f"TF matrix: {tf.shape} | vocab={len(vectorizer.vocabulary_)}")

    bm25 = bm25_weight(tf)
    print(f"BM25 matrix: {bm25.shape}")

    # Save artifacts
    sparse.save_npz(os.path.join(out_subdir, "bm25_okapi.npz"), bm25)

    with open(os.path.join(out_subdir, "vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump(vectorizer.vocabulary_, f, ensure_ascii=False)

    with open(os.path.join(out_subdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    # Optional: store the vectorizer settings for reproducibility
    with open(os.path.join(out_subdir, "vectorizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "stop_words": STOP_WORDS,
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved BM25 artifacts to: {out_subdir}")


# ============================================================
# (Optional) tiny smoke-test
# ============================================================

def smoke_test_query(index_dir: str, query: str, top_k: int = 5) -> None:
    """
    Quick sanity test: load BM25 artifacts and retrieve top_k chunks for a query.
    """
    bm25 = sparse.load_npz(os.path.join(index_dir, "bm25_okapi.npz"))

    with open(os.path.join(index_dir, "vocabulary.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Recreate vectorizer with the same vocab
    vectorizer = CountVectorizer(
        input="content",
        analyzer="word",
        stop_words=STOP_WORDS,
        vocabulary=vocab,
    )

    with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    q_tf = bm25_query_vector(query, vectorizer)
    scores = bm25_scores(bm25, q_tf)
    top_idx = np.argsort(-scores)[:top_k]

    print("\n--- BM25 Smoke Test ---")
    print(f"Query: {query}")
    for rank, i in enumerate(top_idx, 1):
        m = meta[i]
        print(f"{rank}. score={scores[i]:.4f} | file={os.path.basename(m['source_path'])} "
              f"| chunk={m['chunk_index']} | offsets=({m['start_char']},{m['end_char']})")


def main() -> None:
    ensure_dir(BM25_OUT_DIR)

    fixed_out = os.path.join(BM25_OUT_DIR, "fixed")
    sem_out = os.path.join(BM25_OUT_DIR, "semantic")

    build_and_save_bm25(CHUNKS_FIXED_JSONL, fixed_out)
    build_and_save_bm25(CHUNKS_SEM_JSONL, sem_out)

    # Optional quick retrieval test (edit query as you like)
    smoke_test_query(sem_out, query="foreign policy and security cooperation", top_k=5)


if __name__ == "__main__":
    main()
