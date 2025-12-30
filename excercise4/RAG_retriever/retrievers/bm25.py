from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from models.chunk import Chunk
from .base import Retriever
from ..utils import RetrievedChunk, assert_exists, read_chunks_jsonl


@dataclass(frozen=True)
class BM25Index:
    """
    In-memory BM25 artifacts aligned by row index.

    Attributes:
        matrix: Sparse document(term) weight matrix, one row per chunk.
        vectorizer: CountVectorizer configured with the same fixed vocabulary.
        chunks: Chunk objects aligned with `matrix` row order.
    """
    matrix: sparse.csr_matrix
    vectorizer: CountVectorizer
    chunks: List[Chunk]


class BM25Retriever(Retriever):
    """
    Sparse lexical retriever using precomputed BM25 artifacts.

    Loads:
        - `bm25_okapi.npz` sparse matrix
        - `vocabulary.json` used to build a query vectorizer
        - chunk metadata/text from the provided JSONL file

    Notes:
        - Score semantics: higher is better.
        - Supports oversampling for time-aware post-filtering / reranking.
    """

    def __init__(self, index_dir: Path, chunks_jsonl: Path, *, stop_words: str = "english") -> None:
        self._index = self._load(index_dir=index_dir, chunks_jsonl=chunks_jsonl, stop_words=stop_words)

    @staticmethod
    def _load(*, index_dir: Path, chunks_jsonl: Path, stop_words: str) -> BM25Index:
        assert_exists([index_dir / "bm25_okapi.npz", index_dir / "vocabulary.json"])

        matrix = sparse.load_npz(index_dir / "bm25_okapi.npz")
        with (index_dir / "vocabulary.json").open("r", encoding="utf-8") as f:
            vocab = json.load(f)

        vectorizer = CountVectorizer(analyzer="word", stop_words=stop_words, vocabulary=vocab)

        chunks = read_chunks_jsonl(chunks_jsonl)
        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"BM25 row/chunk mismatch for {index_dir}: "
                f"matrix rows={matrix.shape[0]} vs chunks={len(chunks)} from {chunks_jsonl.name}"
            )

        return BM25Index(matrix.tocsr(), vectorizer, chunks)

    def search(self, query: str, k: int) -> List[RetrievedChunk]:
        """
        Return top-k results (no oversampling).
        """
        return self.search_candidates(query, k, oversample=0)

    def search_candidates(self, query: str, k: int, *, oversample: int = 0) -> List[RetrievedChunk]:
        """
        Return top-(k + oversample) results to enable:
          - hard filtering after retrieval
          - soft-decay reranking
        """
        k_total = max(1, int(k) + int(oversample))

        q_vec = self._index.vectorizer.transform([query]).astype(float)
        scores = self._index.matrix.dot(q_vec.T).toarray().ravel()

        top = np.argsort(-scores)[:k_total]
        return [(self._index.chunks[int(i)], float(scores[int(i)])) for i in top]
