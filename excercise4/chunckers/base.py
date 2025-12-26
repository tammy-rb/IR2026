"""
chunkers/base.py

Defines the BaseChunker abstraction used by Exercise 4 Stage 2.

Responsibilities:
- Read a raw debate text file.
- Split it into sentences while preserving character offsets.
- Extract temporal metadata (ISO date + Unix timestamp) from the filename/path.
- Delegate the actual segmentation logic to a concrete chunker
  (FixedChunker / SemanticChunker) via _make_ranges().
- Materialize Chunk objects containing:
  text, offsets, chunking metadata, and temporal metadata.

This file is intentionally "strategy-agnostic":
it provides the common pipeline and enforces uniform chunk schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from models import Chunk
from utils.time_utils import timestamp_from_path
from utils.text_utils import (
    extract_text,
    split_to_sentences_with_spans,
    detect_corpus_label,
    doc_id_from_path,
    count_words,
)

Range = Tuple[int, int, int, int]  # (start_sent_idx, end_sent_idx, start_char, end_char)


class BaseChunker(ABC):
    """
    Abstract base class for chunking strategies.

    Subclasses must define:
    - method_name: str  (e.g., "fixed" or "semantic")
    - _make_ranges(...): returns a list of ranges over sentence spans

    Public API:
    - build_chunks_for_file(file_path): returns a list of Chunk objects.
    """

    method_name: str  # "fixed" / "semantic"

    def build_chunks_for_file(self, file_path: str) -> List[Chunk]:
        """
        Build chunks for a single text file.

        Steps:
        1) Load full file text.
        2) Split into sentences with stable char spans (offsets).
        3) Extract temporal metadata from the file name/path.
        4) Compute chunk ranges using the concrete strategy (_make_ranges).
        5) Materialize Chunk objects, including offsets + text + timestamp metadata.

        Args:
            file_path: Path to a .txt debate file.

        Returns:
            List[Chunk] for the given file, ordered by chunk_index.
        """
        text = extract_text(file_path)
        sentence_spans = split_to_sentences_with_spans(text)

        doc_date_iso, doc_timestamp = timestamp_from_path(file_path)
        corpus = detect_corpus_label(file_path)
        doc_id = doc_id_from_path(file_path)

        ranges = self._make_ranges(sentence_spans)

        chunks: List[Chunk] = []
        for idx, (_s_i, _e_i, start_char, end_char) in enumerate(ranges):
            chunk_text = text[start_char:end_char]
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    source_path=file_path,
                    corpus=corpus,
                    chunking_method=self.method_name,
                    chunk_index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    text=chunk_text,
                    num_words=count_words(chunk_text),
                    doc_date_iso=doc_date_iso,
                    doc_timestamp=doc_timestamp,
                )
            )
        return chunks

    @abstractmethod
    def _make_ranges(self, sentence_spans: List[Tuple[str, int, int]]) -> List[Range]:
        """
        Compute chunk boundaries as ranges over sentence spans.

        Args:
            sentence_spans: List of (sentence_text, start_char, end_char)

        Returns:
            List[Range]: each Range is (start_sent_idx, end_sent_idx, start_char, end_char)
        """
        raise NotImplementedError
