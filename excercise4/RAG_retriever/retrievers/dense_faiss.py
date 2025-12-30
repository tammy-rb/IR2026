from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from models.chunk import Chunk
from .base import Retriever
from ..utils import RetrievedChunk, detect_corpus_label

load_dotenv()


def _distance_to_similarity(d: float) -> float:
    """
    Convert a distance-like score to similarity where higher is better.
    Common for FAISS: lower distance = better match.
    We map: sim = 1 / (1 + max(d, 0)).
    """
    try:
        dd = float(d)
    except Exception:
        dd = 0.0
    if dd < 0:
        dd = abs(dd)
    return 1.0 / (1.0 + dd)


class DenseFAISSRetriever(Retriever):
    """
    Dense semantic retriever using FAISS + OpenAI embeddings.

    IMPORTANT:
        Returns score semantics as similarity (higher is better),
        even if underlying FAISS returns distance.
    """

    def __init__(self, index_dir: Path, *, embed_model: str) -> None:
        self._vs = self._load(index_dir=index_dir, embed_model=embed_model)

    @staticmethod
    def _load(*, index_dir: Path, embed_model: str) -> FAISS:
        if not index_dir.is_dir():
            raise FileNotFoundError(str(index_dir))
        embeddings = OpenAIEmbeddings(model=embed_model)
        return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)

    def search(self, query: str, k: int) -> List[RetrievedChunk]:
        return self.search_candidates(query, k, oversample=0)

    def search_candidates(self, query: str, k: int, *, oversample: int = 0) -> List[RetrievedChunk]:
        k_total = max(1, int(k) + int(oversample))

        pairs = self._vs.similarity_search_with_score(query, k=k_total)
        results: List[RetrievedChunk] = []

        for doc, raw_score in pairs:
            payload: Dict[str, Any] = dict(doc.metadata or {})
            payload["text"] = doc.page_content

            if not payload.get("corpus"):
                payload["corpus"] = detect_corpus_label(payload.get("source_path", ""))

            if payload.get("num_words") is None:
                payload["num_words"] = len([w for w in str(payload["text"]).split() if w])

            chunk = Chunk.from_dict(payload)

            # Convert distance-like score to similarity (higher is better)
            sim = _distance_to_similarity(float(raw_score))
            results.append((chunk, float(sim)))

        return results
