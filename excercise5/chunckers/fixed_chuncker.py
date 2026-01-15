"""
chunkers/fixed_chunker.py

Implements fixed-size chunking for Exercise 4 Stage 2.

Rules (per assignment):
- Chunks contain whole sentences only (no sentence splitting).
- Each chunk contains up to max_words words (default 660),
  unless a single sentence exceeds the limit (then it forms its own chunk).
- Consecutive chunks overlap by overlap_sentences sentences (default 3).

This chunker returns sentence-index + character-offset ranges.
Chunk materialization (Chunk objects + temporal metadata) is handled by BaseChunker.
"""

from __future__ import annotations

from typing import List, Tuple

from chunckers.base import BaseChunker, Range
from utils.text_utils import count_words


class FixedChunker(BaseChunker):
    """Fixed-size sentence-based chunker with configurable overlap."""

    method_name = "fixed"

    def __init__(self, max_words: int = 660, overlap_sentences: int = 3):
        """
        Args:
            max_words: Maximum words per chunk.
            overlap_sentences: Number of sentences to overlap between consecutive chunks.
        """
        self.max_words = max_words
        self.overlap_sentences = overlap_sentences

    def _make_ranges(self, sentence_spans: List[Tuple[str, int, int]]) -> List[Range]:
        """
        Build fixed chunks by accumulating complete sentences up to max_words.

        Args:
            sentence_spans: List of (sentence_text, start_char, end_char)

        Returns:
            List of chunk ranges: (start_sent_idx, end_sent_idx, start_char, end_char)
        """
        n = len(sentence_spans)
        chunks: List[Range] = []

        i = 0
        while i < n:
            start_idx = i
            cur_words = 0
            end_idx = i - 1

            # Grow current chunk by adding sentences until we hit the word limit
            while i < n:
                sent_text, _, _ = sentence_spans[i]
                sent_words = count_words(sent_text)

                # If adding this sentence would exceed the limit, stop (unless chunk is empty)
                if cur_words > 0 and (cur_words + sent_words) > self.max_words:
                    break

                cur_words += sent_words
                end_idx = i
                i += 1

                # If we reached/exceeded the limit, stop
                if cur_words >= self.max_words:
                    break

            # Convert sentence indices to character offsets
            start_char = sentence_spans[start_idx][1]
            end_char = sentence_spans[end_idx][2]
            chunks.append((start_idx, end_idx, start_char, end_char))

            # If no more sentences, we are done
            if i >= n:
                break

            # Move back to create overlap for the next chunk
            i = max(0, (end_idx + 1) - self.overlap_sentences)

            # Safety: if overlap prevents progress, force forward movement
            if i <= start_idx and end_idx >= start_idx:
                i = end_idx + 1

        return chunks
