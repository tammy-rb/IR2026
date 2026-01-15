from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import numpy as np

from models.chunk import Chunk
from ..utils import read_chunks_jsonl


# ------------------------------------------------------------------
# Filter definition
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkFilter:
    """
    Metadata filter for selecting a subset of chunks.

    Time filtering uses Unix timestamps (UTC):
      - time_min_ts: inclusive lower bound
      - time_max_ts: exclusive upper bound

    If require_timestamp=True, chunks with missing doc_timestamp are excluded.

    Optional categorical filters further restrict the candidate set:
      - corpora: allowed corpus labels (e.g., {"us", "british"})
      - chunking_methods: allowed chunking methods (e.g., {"fixed", "semantic"})
      - doc_ids: allowed document identifiers
    """
    time_min_ts: Optional[int] = None
    time_max_ts: Optional[int] = None
    require_timestamp: bool = True

    corpora: Optional[Set[str]] = None
    chunking_methods: Optional[Set[str]] = None
    doc_ids: Optional[Set[str]] = None


# ------------------------------------------------------------------
# Selection result
# ------------------------------------------------------------------

@dataclass(frozen=True)
class Selection:
    """
    Result of applying a ChunkFilter.

    Attributes:
      - row_ids:
          Global row indices of selected chunks (int32).
          The meaning of "row" is defined by the caller and typically
          corresponds to the position of the chunk in a shared chunk list.
      - chunk_uids:
          Stable chunk identifiers aligned with row_ids (same order).
    """
    row_ids: np.ndarray
    chunk_uids: List[str]


# ------------------------------------------------------------------
# Candidate selector (retriever-agnostic)
# ------------------------------------------------------------------

class CandidateSelector:
    """
    Fast, retriever-agnostic metadata prefilter over chunks.

    The selector is built once from a list of Chunk objects and reused
    across many queries or analysis windows.

    It does NOT perform retrieval or scoring.
    It only selects candidate chunk rows based on metadata.

    Alignment invariant:
      row_id == index of the chunk in the provided chunks list.

    This invariant allows the selector to be reused by:
      - sparse retrievers (e.g., BM25)
      - dense retrievers (FAISS, Qdrant, brute-force)
      - hybrid pipelines
    """

    def __init__(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        n = len(chunks)

        # Aligned metadata arrays (one entry per row_id)
        ts = np.full(n, -1, dtype=np.int64)   # -1 indicates missing timestamp
        corpus = np.empty(n, dtype=object)
        method = np.empty(n, dtype=object)
        doc_id = np.empty(n, dtype=object)
        chunk_uid = np.empty(n, dtype=object)

        for i, ch in enumerate(chunks):
            ts[i] = int(ch.doc_timestamp) if ch.doc_timestamp is not None else -1
            corpus[i] = ch.corpus
            method[i] = ch.chunking_method
            doc_id[i] = ch.doc_id
            chunk_uid[i] = ch.chunk_uid

        self._ts = ts
        self._corpus = corpus
        self._method = method
        self._doc_id = doc_id
        self._chunk_uid = chunk_uid

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(cls, chunks_jsonl: Path) -> "CandidateSelector":
        """Construct selector by reading chunks from a JSONL file."""
        chunks = read_chunks_jsonl(chunks_jsonl)
        return cls(chunks)

    @classmethod
    def from_chunks(cls, chunks: List[Chunk]) -> "CandidateSelector":
        """Construct selector from an already-loaded list of chunks."""
        return cls(chunks)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, flt: ChunkFilter) -> Selection:
        """
        Select candidate chunks that satisfy the given filter.

        This method applies metadata constraints only (no scoring).
        The result can be reused by any retrieval backend.

        Returns:
          Selection containing:
            - row_ids: indices into the original chunks list
            - chunk_uids: stable identifiers aligned with those indices
        """
        n = len(self._chunks)
        mask = np.ones(n, dtype=bool)

        # Timestamp requirement
        if flt.require_timestamp:
            mask &= (self._ts >= 0)

        # Time range
        if flt.time_min_ts is not None:
            mask &= (self._ts >= int(flt.time_min_ts))
        if flt.time_max_ts is not None:
            mask &= (self._ts < int(flt.time_max_ts))

        # Corpus constraint
        if flt.corpora:
            mask &= np.isin(self._corpus, list(flt.corpora))

        # Chunking method constraint
        if flt.chunking_methods:
            mask &= np.isin(self._method, list(flt.chunking_methods))

        # Document constraint
        if flt.doc_ids:
            mask &= np.isin(self._doc_id, list(flt.doc_ids))

        row_ids = np.where(mask)[0].astype(np.int32)
        chunk_uids = [str(self._chunk_uid[int(i)]) for i in row_ids.tolist()]
        return Selection(row_ids=row_ids, chunk_uids=chunk_uids)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def chunks_for_rows(self, row_ids: np.ndarray) -> List[Chunk]:
        """
        Convert selected row indices back into Chunk objects.

        Intended for debugging, inspection, or analysis.
        Retrieval backends should typically operate directly on row_ids.
        """
        return [self._chunks[int(i)] for i in row_ids.tolist()]