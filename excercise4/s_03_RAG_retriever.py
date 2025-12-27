"""
s_03_RAG_retriever.py

Central retrieval engine for a Retrieval-Augmented Generation (RAG) system.

This module encapsulates all retrieval logic and resources, including:
- Sparse lexical retrieval using BM25
- Dense semantic retrieval using FAISS with OpenAI embeddings
- Support for multiple chunking strategies (fixed / semantic)

All indices are loaded once and cached in memory, allowing efficient
repeated top-K retrieval calls from an external LLM orchestration layer.

This file intentionally contains NO LLM logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from models.chunk import Chunk
from paths import CHUNKS_DIR, BM25_DIR, OPENAI_DIR


# ============================================================
# Configuration
# ============================================================

BM25_FIXED_DIR = BM25_DIR / "fixed"
BM25_SEM_DIR = BM25_DIR / "semantic"

FAISS_FIXED_DIR = OPENAI_DIR / "fixed_faiss"
FAISS_SEM_DIR = OPENAI_DIR / "semantic_faiss"

CHUNKS_FIXED_JSONL = CHUNKS_DIR / "chunks_fixed.jsonl"
CHUNKS_SEM_JSONL = CHUNKS_DIR / "chunks_semantic.jsonl"

OPENAI_EMBED_MODEL = "text-embedding-3-large"
STOP_WORDS = "english"

BRITISH = "british_parliament_debates"
US = "US_congress_debates"

"""Internal retrieved item: validated Chunk + score."""
RetrievedChunk = Tuple[Chunk, float]


# ============================================================
# Utilities
# ============================================================

def read_chunks_jsonl(path: Path) -> List[Chunk]:
    """Load and validate Chunk records from a JSONL file.

    Raises:
        FileNotFoundError: if the file is missing.
        ValueError: if any line cannot be parsed into a Chunk.
    """
    if not path.is_file():
        raise FileNotFoundError(str(path))

    return Chunk.read_jsonl(path)

def detect_corpus_label(path: str) -> str:
    """
    Determine which corpus the file belongs to (british/us) using folder name.
    Falls back to the file's parent folder name if neither token is found.
    """
    low = (path or "").lower()
    if BRITISH.lower() in low:
        return "british"
    if US.lower() in low:
        return "us"
    return os.path.basename(os.path.dirname(path or "")) or "unknown"

def short_source_id(chunk: Chunk) -> str:
    """
    Create a compact citation identifier for a chunk, including corpus label.

    Returns:
        String identifier in the form: corpus:filename [start,end]
        Example: us:debate_12.txt [123,456]
    """
    source_path = chunk.source_path or ""
    corpus = chunk.corpus or detect_corpus_label(source_path)

    base = os.path.basename(source_path)
    return f"{corpus}:{base} [{chunk.start_char},{chunk.end_char}]"


def build_context_block(chunks: List[RetrievedChunk]) -> str:
    """
    Convert retrieved chunks into a single context block for an LLM.

    Each chunk is prefixed with a citation marker.

    Args:
        chunks: Retrieved chunks (metadata, score, text).

    Returns:
        Formatted context string.
    """
    lines: List[str] = []
    for chunk, _score in chunks:
        lines.append(f"[{short_source_id(chunk)}]")
        lines.append(chunk.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def assert_exists(paths: List[Path]) -> None:
    """
    Ensure required files or directories exist.

    Args:
        paths: List of filesystem paths.

    Raises:
        FileNotFoundError: If any path is missing.
    """
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


# ============================================================
# BM25 Retrieval
# ============================================================

@dataclass
class BM25Index:
    """
    Container for BM25 retrieval artifacts.
    Args:
        matrix: Sparse document(chunck)-term matrix.
        vectorizer: CountVectorizer instance.
        chunks: List of Chunk objects aligned with BM25 matrix row order.
    """
    matrix: sparse.csr_matrix
    vectorizer: CountVectorizer
    chunks: List[Chunk]


def load_bm25(index_dir: Path, chunks_jsonl: Path) -> BM25Index:
    """
    Load BM25 artifacts from disk.

    Args:
        index_dir: Directory containing BM25 artifacts.

    Returns:
        Loaded BM25Index instance.
    """
    assert_exists([
        index_dir / "bm25_okapi.npz",
        index_dir / "vocabulary.json",
    ])

    matrix = sparse.load_npz(index_dir / "bm25_okapi.npz")

    with (index_dir / "vocabulary.json").open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    vectorizer = CountVectorizer(
        analyzer="word",
        stop_words=STOP_WORDS,
        vocabulary=vocab,
    )

    chunks = read_chunks_jsonl(chunks_jsonl)
    if matrix.shape[0] != len(chunks):
        raise ValueError(
            f"BM25 row/chunk mismatch for {index_dir}: "
            f"matrix rows={matrix.shape[0]} vs chunks={len(chunks)} from {chunks_jsonl.name}"
        )

    return BM25Index(matrix.tocsr(), vectorizer, chunks)


def bm25_search(index: BM25Index, query: str, k: int) -> List[RetrievedChunk]:
    """
    Perform BM25 top-K retrieval.

    Args:
        index: Loaded BM25 index.
        query: Query string.
        k: Number of results to return.

    Returns:
        Top-K retrieved chunks with scores.
    """
    # convert the query to vector using the same vectorizer (vocabulary)
    q_vec = index.vectorizer.transform([query]).astype(float)
    scores = index.matrix.dot(q_vec.T).toarray().ravel()

    top = np.argsort(-scores)[:k]
    results: List[RetrievedChunk] = []

    for i in top:
        chunk = index.chunks[int(i)]
        results.append((chunk, float(scores[int(i)])))

    return results


# ============================================================
# Dense Retrieval (FAISS)
# ============================================================

def load_faiss(index_dir: Path) -> FAISS:
    """
    Load a FAISS vector store from disk.

    Args:
        index_dir: Directory containing FAISS index files.

    Returns:
        Loaded FAISS vector store.
    """
    if not index_dir.is_dir():
        raise FileNotFoundError(str(index_dir))

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)


def dense_search(vs: FAISS, query: str, k: int) -> List[RetrievedChunk]:
    """
    Perform dense semantic retrieval using FAISS.

    Args:
        vs: Loaded FAISS vector store.
        query: Query string.
        k: Number of results to return.

    Returns:
        Top-K retrieved chunks with similarity scores.
    """
    pairs = vs.similarity_search_with_score(query, k=k)
    results: List[RetrievedChunk] = []

    for doc, score in pairs:
        payload: Dict[str, Any] = dict(doc.metadata or {})
        payload["text"] = doc.page_content

        # Be forgiving about corpus; default using path.
        if not payload.get("corpus"):
            payload["corpus"] = detect_corpus_label(payload.get("source_path", ""))

        # Ensure num_words exists (older indexes might not have it).
        if payload.get("num_words") is None:
            payload["num_words"] = len([w for w in str(payload["text"]).split() if w])

        chunk = Chunk.from_dict(payload)
        results.append((chunk, float(score)))

    return results


# ============================================================
# Retriever Interface
# ============================================================

@dataclass
class Pipeline:
    """
    Retrieval pipeline configuration.
    """
    chunking: str         # "fixed" or "semantic"
    representation: str   # "bm25" or "dense"
    bm25: Optional[BM25Index]
    faiss: Optional[FAISS]


class RAGRetriever:
    """
    Unified retrieval engine for a RAG system.

    This class loads and caches all retrieval pipelines and exposes
    a single `get_topk` method used by the LLM layer.
    """

    def __init__(self) -> None:
        self._pipelines: Dict[Tuple[str, str], Pipeline] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all supported retrieval pipelines into memory."""
        for chunking in ("fixed", "semantic"):
            for repr_ in ("bm25", "dense"):
                self._pipelines[(chunking, repr_)] = self._load_pipeline(chunking, repr_)

    def _load_pipeline(self, chunking: str, repr_: str) -> Pipeline:
        """
        Load a single retrieval pipeline.

        Args:
            chunking: Chunking strategy ("fixed" or "semantic").
            repr_: Representation type ("bm25" or "dense").

        Returns:
            Initialized Pipeline instance.
        """
        bm25 = faiss = None

        if repr_ == "bm25":
            bm25 = load_bm25(
                BM25_FIXED_DIR if chunking == "fixed" else BM25_SEM_DIR,
                CHUNKS_FIXED_JSONL if chunking == "fixed" else CHUNKS_SEM_JSONL,
            )

        if repr_ == "dense":
            faiss = load_faiss(FAISS_FIXED_DIR if chunking == "fixed" else FAISS_SEM_DIR)

        return Pipeline(chunking, repr_, bm25, faiss)

    def get_topk(
        self,
        query: str,
        chunking: str,
        representation: str,
        k: int,
    ) -> Dict[str, Any]:
        """
        Retrieve top-K chunks for a given query and pipeline.

        Args:
            query: Query string.
            chunking: Chunking strategy.
            representation: Retrieval representation.
            k: Number of chunks to retrieve.

        Returns:
            Dictionary containing retrieved chunks, formatted context,
            and reference metadata.
        """
        pipe = self._pipelines[(chunking, representation)]

        if representation == "bm25":
            retrieved = bm25_search(pipe.bm25, query, k)
        else:
            retrieved = dense_search(pipe.faiss, query, k)

        return {
            # Keep a JSON-serializable representation for downstream scripts.
            "retrieved": [
                {
                    "chunk": asdict(c),
                    "score": float(s),
                    "text": c.text,
                }
                for (c, s) in retrieved
            ],
            "context": build_context_block(retrieved),
            "refs": [
                {
                    "corpus": (c.corpus or detect_corpus_label(c.source_path)),
                    "file_name": os.path.basename(c.source_path or ""),
                    "source_path": c.source_path,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "chunk_index": c.chunk_index,
                    "doc_date_iso": c.doc_date_iso,
                    "doc_timestamp": c.doc_timestamp,
                }
                for (c, _s) in retrieved
            ],
        }
