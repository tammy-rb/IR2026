from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, TypeVar, Set

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from models.chunk import Chunk
from .base import Retriever
from ..utils import RetrievedChunk
from ..prefilter.chuncks_selector import ChunkFilter

T = TypeVar("T")


@dataclass(frozen=True)
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "chunks_openai"
    model: str = "text-embedding-3-large"


def chunkfilter_to_qdrant_filter(flt: ChunkFilter) -> qmodels.Filter:
    """
    Convert your project ChunkFilter -> Qdrant Filter.

    ChunkFilter semantics:
      - time_min_ts: inclusive
      - time_max_ts: exclusive
      - require_timestamp: if True, exclude chunks with missing doc_timestamp
      - corpora: optional set[str]
      - chunking_methods: optional set[str]
      - doc_ids: optional set[str]

    Qdrant notes:
      - For "time_max_ts exclusive", we implement: doc_timestamp < time_max_ts
        using Range(lt=...).
      - For require_timestamp, we implement: doc_timestamp >= 0
        (since you store unix timestamps >= 0; missing timestamps are None and won't match range).
    """
    must: List[qmodels.Condition] = []

    # require_timestamp
    if flt.require_timestamp:
        must.append(
            qmodels.FieldCondition(
                key="doc_timestamp",
                range=qmodels.Range(gte=0),
            )
        )

    # time range
    if flt.time_min_ts is not None or flt.time_max_ts is not None:
        must.append(
            qmodels.FieldCondition(
                key="doc_timestamp",
                range=qmodels.Range(
                    gte=int(flt.time_min_ts) if flt.time_min_ts is not None else None,
                    lt=int(flt.time_max_ts) if flt.time_max_ts is not None else None,
                ),
            )
        )

    # corpus constraint
    if flt.corpora:
        must.append(_match_any_keyword("corpus", flt.corpora))

    # doc_id constraint
    if flt.doc_ids:
        must.append(_match_any_keyword("doc_id", flt.doc_ids))

    return qmodels.Filter(must=must)


def _match_any_keyword(field: str, values: Set[str]) -> qmodels.Condition:
    """
    Match field in {values}.

    Qdrant has a MatchAny in newer versions. In some versions, the object name differs.
    This helper tries MatchAny if available; otherwise it falls back to OR-ing conditions.
    """
    vals = [str(v) for v in values if v is not None]
    if not vals:
        # No constraint
        return qmodels.Filter(must=[])

    # Prefer MatchAny if your qdrant_client supports it
    match_any = getattr(qmodels, "MatchAny", None)
    if match_any is not None:
        return qmodels.FieldCondition(key=field, match=match_any(any=vals))

    # Fallback: should=[cond1, cond2, ...] means OR
    should_conds = [
        qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=v)) for v in vals
    ]
    return qmodels.Filter(should=should_conds)


class QdrantDenseRetriever(Retriever):
    """
    Dense retriever backed by Qdrant + OpenAI embeddings.

    - Embeds the query using OpenAIEmbeddings(model=...)
    - Uses Qdrant vector search to retrieve points
    - Converts point payload -> Chunk (Chunk.from_dict)
    - Returns (Chunk, score) where score is "higher is better"
    """

    def __init__(
        self,
        *,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "chunks_openai",
        model: str = "text-embedding-3-large",
        timeout_s: Optional[float] = 60.0,
    ) -> None:
        self.cfg = QdrantConfig(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name,
            model=model,
        )
        self._embeddings = OpenAIEmbeddings(model=self.cfg.model)
        self._client = QdrantClient(host=self.cfg.host, port=self.cfg.port, timeout=timeout_s)
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        cols = self._with_retry(lambda: self._client.get_collections()).collections
        if not any(c.name == self.cfg.collection_name for c in cols):
            raise ValueError(
                f"Qdrant collection '{self.cfg.collection_name}' not found on {self.cfg.host}:{self.cfg.port}. "
                f"Did you run the embedder first?"
            )

    # ----------------------------
    # Retriever API
    # ----------------------------

    def search(self, query: str, k: int) -> List[RetrievedChunk]:
        return self.search_candidates(query, k, oversample=0)

    def search_candidates(self, query: str, k: int, *, oversample: int = 0) -> List[RetrievedChunk]:
        k_total = max(1, int(k) + int(oversample))
        qvec = self._with_retry(lambda: self._embeddings.embed_query(query))

        hits = self._with_retry(
            lambda: self._client.search(
                collection_name=self.cfg.collection_name,
                query_vector=qvec,
                limit=k_total,
                with_payload=True,
                with_vectors=False,
            )
        )
        return self._hits_to_retrieved_chunks(hits)

    def search_candidates_prefiltered(
        self,
        query: str,
        k: int,
        *,
        flt: ChunkFilter,
        oversample: int = 0,
    ) -> List[RetrievedChunk]:
        """
        Dense retrieval with metadata filter pushed down into Qdrant.
        """
        k_total = max(1, int(k) + int(oversample))
        qvec = self._with_retry(lambda: self._embeddings.embed_query(query))

        q_filter = chunkfilter_to_qdrant_filter(flt)

        hits = self._with_retry(
            lambda: self._client.search(
                collection_name=self.cfg.collection_name,
                query_vector=qvec,
                query_filter=q_filter,
                limit=k_total,
                with_payload=True,
                with_vectors=False,
            )
        )
        return self._hits_to_retrieved_chunks(hits)

    # ----------------------------
    # Helpers
    # ----------------------------

    def _hits_to_retrieved_chunks(self, hits: List[qmodels.ScoredPoint]) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        for h in hits:
            payload = h.payload or {}
            try:
                chunk = Chunk.from_dict(payload)
            except Exception:
                raise ValueError(
                    "Qdrant payload could not be parsed into Chunk. "
                    f"payload_keys={sorted(list(payload.keys()))}"
                )
            out.append((chunk, float(h.score)))
        return out

    # ----------------------------
    # Retry helper
    # ----------------------------

    def _with_retry(self, fn: Callable[[], T], *, max_retries: int = 6, base_sleep_s: float = 2.0) -> T:
        last_err: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_err = e
                if attempt >= max_retries:
                    raise
                sleep_s = min(base_sleep_s * (2**attempt), 60.0)
                print(f"  ⚠️  call failed ({e.__class__.__name__}: {e}). Retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)

        assert last_err is not None
        raise last_err
