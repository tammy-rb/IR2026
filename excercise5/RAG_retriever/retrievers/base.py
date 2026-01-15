from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from models.chunk import Chunk
from ..prefilter.chuncks_selector import ChunkFilter


# Retrieved item: validated Chunk + score
# For BM25: higher scores = more relevant
# For dense retrievers: score semantics depend on vector store (often distance)
RetrievedChunk = Tuple[Chunk, float]


class Retriever(ABC):
    """
    Retriever interface.

    Implementations encapsulate loading/index resources and expose common methods.

    IMPORTANT:
    - For all retrievers, score semantics must be "higher is better".
      (Dense retrievers must convert distances to similarity.)
    """

    @abstractmethod
    def search(self, query: str, k: int) -> List[RetrievedChunk]:
        """Return top-k retrieved chunks for the given query (no filtering)."""
        raise NotImplementedError

    def search_candidates(self, query: str, k: int, *, oversample: int = 0) -> List[RetrievedChunk]:
        """
        Return top-(k + oversample) results to allow:
        - hard filtering after retrieval
        - soft-decay reranking

        Default implementation falls back to search() only.
        """
        k_total = max(1, int(k) + int(oversample))
        return self.search(query, k_total)

    def search_candidates_prefiltered(
        self,
        query: str,
        k: int,
        *,
        flt: ChunkFilter,
        oversample: int = 0,
    ) -> List[RetrievedChunk]:
        """
        Metadata-prefiltered retrieval.

        This method MUST be overridden by retrievers that support
        efficient prefiltering.

        Examples:
        - BM25: CandidateSelector row slicing before scoring
        - QdrantDense: translate ChunkFilter -> Qdrant filter

        Default behavior: not supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support metadata prefiltering"
        )
