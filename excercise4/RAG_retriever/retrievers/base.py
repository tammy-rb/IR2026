from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..utils import RetrievedChunk


class Retriever(ABC):
    """
    Retriever interface.

    Implementations encapsulate loading/index resources and expose a common
    `search(query, k)` method returning (Chunk, score) pairs.

    IMPORTANT:
    - For all retrievers, score semantics must be "higher is better".
      (Dense retrievers must convert distances to similarity.)
    """

    @abstractmethod
    def search(self, query: str, k: int) -> List[RetrievedChunk]:
        """Return top-k retrieved chunks for the given query."""
        raise NotImplementedError

    def search_candidates(self, query: str, k: int, *, oversample: int = 0) -> List[RetrievedChunk]:
        """
        Return more than k results to allow:
        - hard filtering after retrieval
        - soft-decay reranking

        Default implementation falls back to search() only.
        Backends should override for efficient oversampling.
        """
        k_total = max(1, int(k) + int(oversample))
        return self.search(query, k_total)
