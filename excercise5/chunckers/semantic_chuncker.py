"""
chunkers/semantic_chunker.py

Implements semantic chunking for Exercise 4 Stage 2.

Approach:
- Split document into sentences (done in BaseChunker).
- Embed sentences using Sentence-Transformers.
- Compute cosine similarity between adjacent sentences.
- Mark a "semantic boundary" when similarity drops below sim_threshold.
- Build chunks while enforcing:
  - max_words (default 660): forced boundary if exceeded.
  - min_sentences_per_chunk (default 4): avoids tiny chunks.
  - optional overlap_sentences (default 0).

This chunker returns sentence-index + character-offset ranges.
Chunk materialization (Chunk objects + temporal metadata) is handled by BaseChunker.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from chunckers.base import BaseChunker, Range
from utils.text_utils import count_words


class SemanticChunker(BaseChunker):
    """Embedding-based semantic chunker using Sentence-Transformers."""

    method_name = "semantic"

    def __init__(
        self,
        max_words: int = 660,
        sim_threshold: float = 0.62,
        min_sentences_per_chunk: int = 4,
        overlap_sentences: int = 0,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Args:
            max_words: Maximum words per chunk (forced boundary beyond this).
            sim_threshold: Boundary threshold for adjacent-sentence cosine similarity.
            min_sentences_per_chunk: Prevents very small chunks.
            overlap_sentences: Optional overlap between chunks (usually 0 for semantic).
            embedding_model: Sentence-Transformers model name.
        """
        self.max_words = max_words
        self.sim_threshold = sim_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.embedding_model = embedding_model
        self._model = None

    def _load_model(self):
        """Lazy-load Sentence-Transformers model (loaded once per run)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    def _embed(self, sents: List[str]):
        """
        Encode sentences into normalized embeddings.

        normalize_embeddings=True ensures vectors are unit-length (L2),
        making cosine similarity equivalent to dot product.
        """
        model = self._load_model()
        return model.encode(sents, normalize_embeddings=True, show_progress_bar=False)

    @staticmethod
    def _cos(a, b) -> float:
        """Cosine similarity between two vectors."""
        a = np.asarray(a).reshape(1, -1)
        b = np.asarray(b).reshape(1, -1)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float((a @ b.T)[0, 0] / denom)

    def _make_ranges(self, sentence_spans: List[Tuple[str, int, int]]) -> List[Range]:
        """
        Build semantic chunks using adjacent-sentence similarity boundaries.

        Args:
            sentence_spans: List of (sentence_text, start_char, end_char)

        Returns:
            List of chunk ranges: (start_sent_idx, end_sent_idx, start_char, end_char)
        """
        n = len(sentence_spans)
        if n == 0:
            return []

        sent_texts = [t for (t, _, _) in sentence_spans]
        E = self._embed(sent_texts)

        sims = [self._cos(E[i], E[i + 1]) for i in range(n - 1)]
        boundary_after = {i for i, s in enumerate(sims) if s < self.sim_threshold}

        chunks: List[Range] = []
        start_idx = 0
        i = 0
        cur_words = 0
        cur_sent_count = 0

        def flush(end_idx_inclusive: int) -> None:
            """Finalize the current chunk and reset counters (with optional overlap)."""
            nonlocal start_idx, cur_words, cur_sent_count

            start_char = sentence_spans[start_idx][1]
            end_char = sentence_spans[end_idx_inclusive][2]
            chunks.append((start_idx, end_idx_inclusive, start_char, end_char))

            next_start = end_idx_inclusive + 1 - self.overlap_sentences
            start_idx = max(0, next_start)
            cur_words = 0
            cur_sent_count = 0

        while i < n:
            sent_text, _, _ = sentence_spans[i]
            sent_words = count_words(sent_text)

            # Forced boundary if adding this sentence exceeds max_words.
            if cur_sent_count > 0 and (cur_words + sent_words) > self.max_words:
                flush(i - 1)
                i = start_idx
                continue

            cur_words += sent_words
            cur_sent_count += 1

            is_last = (i == n - 1)

            # Semantic boundary (only if chunk is big enough).
            if (
                not is_last
                and (i in boundary_after)
                and (cur_sent_count >= self.min_sentences_per_chunk)
            ):
                flush(i)
                i = start_idx
                continue

            if is_last:
                flush(i)
                break

            i += 1

        return chunks
