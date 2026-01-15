from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from models.chunk import Chunk
from .base import Retriever, RetrievedChunk
from ..utils import assert_exists, read_chunks_jsonl
from ..prefilter.chuncks_selector import CandidateSelector, ChunkFilter


@dataclass(frozen=True)
class BM25Index:
    """
    In-memory BM25 artifacts aligned by row index.
    """
    matrix: sparse.csr_matrix
    vectorizer: CountVectorizer
    chunks: List[Chunk]


class BM25Retriever(Retriever):
    """
    Sparse lexical retriever using precomputed BM25 artifacts.

    Score semantics: higher is better.
    Supports oversampling and metadata prefiltering.
    """

    def __init__(self, index_dir: Path, chunks_jsonl: Path, *, stop_words: str = "english") -> None:
        self._index = self._load(index_dir=index_dir, chunks_jsonl=chunks_jsonl, stop_words=stop_words)
        self._selector = CandidateSelector.from_chunks(self._index.chunks)

    @staticmethod
    def _load(*, index_dir: Path, chunks_jsonl: Path, stop_words: str) -> BM25Index:
        assert_exists([index_dir / "bm25_okapi.npz", index_dir / "vocabulary.json"])

        matrix = sparse.load_npz(index_dir / "bm25_okapi.npz").tocsr()

        with (index_dir / "vocabulary.json").open("r", encoding="utf-8") as f:
            vocab = json.load(f)

        vectorizer = CountVectorizer(analyzer="word", stop_words=stop_words, vocabulary=vocab)

        chunks = read_chunks_jsonl(chunks_jsonl)
        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"BM25 row/chunk mismatch for {index_dir}: "
                f"matrix rows={matrix.shape[0]} vs chunks={len(chunks)} from {chunks_jsonl.name}"
            )

        return BM25Index(matrix=matrix, vectorizer=vectorizer, chunks=chunks)

    def search(self, query: str, k: int) -> List[RetrievedChunk]:
        return self.search_candidates(query, k, oversample=0)

    def search_candidates(self, query: str, k: int, *, oversample: int = 0) -> List[RetrievedChunk]:
        k_total = max(1, int(k) + int(oversample))

        q_vec = self._index.vectorizer.transform([query]).astype(float)

        # Efficient 1D scores
        scores = (self._index.matrix @ q_vec.T).A1  # shape: (num_chunks,)

        top = np.argsort(-scores)[:k_total]
        return [(self._index.chunks[int(i)], float(scores[int(i)])) for i in top]

    def search_candidates_prefiltered(
        self,
        query: str,
        k: int,
        *,
        flt: ChunkFilter,
        oversample: int = 0,
    ) -> List[RetrievedChunk]:
        """
        Metadata-prefiltered BM25 retrieval using CandidateSelector row slicing.
        """
        selected = self._selector.select(flt)
        row_ids = selected.row_ids
        if row_ids.size == 0:
            return []

        k_total = max(1, int(k) + int(oversample))

        q_vec = self._index.vectorizer.transform([query]).astype(float)

        sub_matrix = self._index.matrix[row_ids]
        scores_subset = (sub_matrix @ q_vec.T).A1  # shape: (len(row_ids),)

        top_local = np.argsort(-scores_subset)[:k_total]
        top_global = row_ids[top_local]

        results: List[RetrievedChunk] = []
        for local_i, global_i in zip(top_local.tolist(), top_global.tolist()):
            results.append((self._index.chunks[int(global_i)], float(scores_subset[int(local_i)])))

        return results
