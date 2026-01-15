"""
embedders/bm25_embedder.py

BM25 (Okapi) sparse embedding strategy.

Builds BM25-weighted sparse retrieval indexes over chunk corpora
while preserving temporal metadata (doc_date_iso + doc_timestamp).

Outputs per strategy:
- bm25_okapi.npz          (sparse BM25 matrix)
- vocabulary.json         (term -> index)
- meta.json               (chunk metadata including timestamps)
- row_uids.json           (row index -> chunk_uid alignment)
- vectorizer_config.json  (reproducibility config)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from embedders.base import BaseEmbedder

# BM25 params (Okapi)
BM25_K1 = 1.6
BM25_B = 0.75

# Vectorizer params
STOP_WORDS = "english"
MIN_DF = 2
MAX_DF = 0.85


class BM25Embedder(BaseEmbedder):
    """
    BM25 (Okapi) sparse embedding implementation.
    
    Converts chunks into BM25-weighted sparse vectors for retrieval.
    """
    
    def __init__(
        self,
        output_dir: Path,
        k1: float = BM25_K1,
        b: float = BM25_B,
        stop_words: str = STOP_WORDS,
        min_df: int = MIN_DF,
        max_df: float = MAX_DF,
    ):
        """
        Initialize BM25 embedder.
        
        Args:
            output_dir: Directory to save artifacts.
            k1: BM25 k1 parameter.
            b: BM25 b parameter.
            stop_words: Stop words setting for vectorizer.
            min_df: Minimum document frequency.
            max_df: Maximum document frequency.
        """
        super().__init__(output_dir)
        self.k1 = k1
        self.b = b
        self.stop_words = stop_words
        self.min_df = min_df
        self.max_df = max_df
    
    @property
    def name(self) -> str:
        return "bm25"
    
    def build_tf_matrix(self, docs: List[str]) -> Tuple[sparse.csr_matrix, CountVectorizer]:
        """
        Build a sparse TF (term frequency) matrix over chunk texts.
        
        Args:
            docs: List of chunk texts.
        
        Returns:
            (tf_csr_matrix, fitted_vectorizer)
        """
        vectorizer = CountVectorizer(
            input="content",
            analyzer="word",
            stop_words=self.stop_words,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        tf = vectorizer.fit_transform(docs)
        return tf.tocsr().astype(float), vectorizer
    
    def bm25_weight(self, tf_csr: sparse.csr_matrix) -> sparse.csr_matrix:
        """
        Convert a TF matrix into an Okapi BM25-weighted sparse matrix.
        
        Args:
            tf_csr: Term-frequency matrix (CSR).
        
        Returns:
            BM25-weighted matrix (CSR) with the same shape as tf_csr.
        """
        tf_csr = tf_csr.tocsr().astype(float)
        N, n_terms = tf_csr.shape
        
        # Document frequency per term
        df = np.bincount(tf_csr.indices, minlength=n_terms)
        
        # IDF
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
        
        # Document lengths
        dl = np.asarray(tf_csr.sum(axis=1)).ravel()
        avgdl = dl.mean() if dl.size else 0.0
        
        rows, cols = tf_csr.nonzero()
        data = tf_csr.data
        new_data = np.empty_like(data)
        
        for i in range(len(data)):
            doc = rows[i]
            term = cols[i]
            tf = data[i]
            denom = tf + self.k1 * (1.0 - self.b + self.b * (dl[doc] / (avgdl + 1e-12)))
            new_data[i] = idf[term] * (tf * (self.k1 + 1.0)) / (denom + 1e-12)
        
        return sparse.csr_matrix((new_data, (rows, cols)), shape=tf_csr.shape)
    
    def embed_chunks(self, chunks_jsonl: Path) -> None:
        """
        Build and persist BM25 artifacts for a chunk JSONL file.
        
        Args:
            chunks_jsonl: Path to chunks JSONL file.
        """
        chunks = self.read_chunks(chunks_jsonl)
        if not chunks:
            raise ValueError(f"No chunks found in: {chunks_jsonl}")
        
        docs: List[str] = []
        meta: List[Dict[str, Any]] = []
        
        for c in chunks:
            docs.append(c.text)
            meta.append(self.extract_metadata(c))
        
        print(f"Building BM25 index for {len(docs)} chunks...")
        
        # Build TF matrix
        tf, vectorizer = self.build_tf_matrix(docs)
        
        # Convert to BM25
        bm25 = self.bm25_weight(tf)
        
        # Save artifacts
        sparse.save_npz(self.output_dir / "bm25_okapi.npz", bm25)
        print(f"  ✓ Saved BM25 matrix: {self.output_dir / 'bm25_okapi.npz'}")
        
        with (self.output_dir / "vocabulary.json").open("w", encoding="utf-8") as f:
            json.dump(vectorizer.vocabulary_, f, ensure_ascii=False)
        print(f"  ✓ Saved vocabulary")
        
        self.save_metadata(meta)
        print(f"  ✓ Saved metadata")
        
         # Save row -> chunk_uid alignment
        row_uids = [c.chunk_uid for c in chunks]
        with (self.output_dir / "row_uids.json").open("w", encoding="utf-8") as f:
            json.dump(row_uids, f, ensure_ascii=False)
        print("  ✓ Saved row_uids.json")
        
        # Save vectorizer config
        with (self.output_dir / "vectorizer_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "stop_words": self.stop_words,
                    "min_df": self.min_df,
                    "max_df": self.max_df,
                    "bm25_k1": self.k1,
                    "bm25_b": self.b,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  ✓ Saved vectorizer config")
        
        print(f"✅ BM25 artifacts saved to: {self.output_dir}")
