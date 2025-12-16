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

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

BM25_DIR = os.path.join(BASE_DIR, "outputs", "bm25")
DENSE_DIR = os.path.join(BASE_DIR, "outputs", "embeddings_openai")

BM25_FIXED_DIR = os.path.join(BM25_DIR, "fixed")
BM25_SEM_DIR = os.path.join(BM25_DIR, "semantic")

FAISS_FIXED_DIR = os.path.join(DENSE_DIR, "fixed_faiss")
FAISS_SEM_DIR = os.path.join(DENSE_DIR, "semantic_faiss")

OPENAI_EMBED_MODEL = "text-embedding-3-large"
STOP_WORDS = "english"

BRITISH = "british_parliament_debates"
US = "US_congress_debates"

'''
Dict - metadata
float - score
str - chunk text
'''
RetrievedChunk = Tuple[Dict[str, Any], float, str]


# ============================================================
# Utilities
# ============================================================

def extract_text(path: str) -> str:
    """
    Read a UTF-8 encoded text file from disk.

    Args:
        path: Path to the source document.

    Returns:
        Full document text as a string.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def materialize_from_offsets(path: str, start: int, end: int) -> str:
    """
    Reconstruct chunk text using character offsets.

    Args:
        path: Path to original document.
        start: Inclusive start character offset.
        end: Exclusive end character offset.

    Returns:
        Substring corresponding to the chunk.
    """
    return extract_text(path)[start:end]

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

def short_source_id(meta: Dict[str, Any]) -> str:
    """
    Create a compact citation identifier for a chunk, including corpus label.

    Returns:
        String identifier in the form: corpus:filename [start,end]
        Example: us:debate_12.txt [123,456]
    """
    source_path = meta.get("source_path", "")
    corpus = meta.get("corpus") or detect_corpus_label(source_path)

    base = os.path.basename(source_path)
    return f"{corpus}:{base} [{meta.get('start_char')},{meta.get('end_char')}]"


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
    for meta, _score, text in chunks:
        lines.append(f"[{short_source_id(meta)}]")
        lines.append(text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def assert_exists(paths: List[str]) -> None:
    """
    Ensure required files or directories exist.

    Args:
        paths: List of filesystem paths.

    Raises:
        FileNotFoundError: If any path is missing.
    """
    missing = [p for p in paths if not os.path.exists(p)]
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
        meta: List of chunk metadata dictionaries.
    """
    matrix: sparse.csr_matrix
    vectorizer: CountVectorizer
    meta: List[Dict[str, Any]]


def load_bm25(index_dir: str) -> BM25Index:
    """
    Load BM25 artifacts from disk.

    Args:
        index_dir: Directory containing BM25 artifacts.

    Returns:
        Loaded BM25Index instance.
    """
    assert_exists([
        os.path.join(index_dir, "bm25_okapi.npz"),
        os.path.join(index_dir, "vocabulary.json"),
        os.path.join(index_dir, "meta.json"),
    ])

    matrix = sparse.load_npz(os.path.join(index_dir, "bm25_okapi.npz"))

    with open(os.path.join(index_dir, "vocabulary.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vectorizer = CountVectorizer(
        analyzer="word",
        stop_words=STOP_WORDS,
        vocabulary=vocab,
    )

    with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    return BM25Index(matrix.tocsr(), vectorizer, meta)


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
        meta = index.meta[int(i)]
        text = materialize_from_offsets(
            meta["source_path"],
            meta["start_char"],
            meta["end_char"],
        )
        results.append((meta, float(scores[int(i)]), text))

    return results


# ============================================================
# Dense Retrieval (FAISS)
# ============================================================

def load_faiss(index_dir: str) -> FAISS:
    """
    Load a FAISS vector store from disk.

    Args:
        index_dir: Directory containing FAISS index files.

    Returns:
        Loaded FAISS vector store.
    """
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(index_dir)

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


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
        meta = dict(doc.metadata)
        if all(x in meta for x in ("source_path", "start_char", "end_char")):
            text = materialize_from_offsets(
                meta["source_path"],
                int(meta["start_char"]),
                int(meta["end_char"]),
            )
        else:
            text = doc.page_content

        results.append((meta, float(score), text))

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
            bm25 = load_bm25(BM25_FIXED_DIR if chunking == "fixed" else BM25_SEM_DIR)

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
            "retrieved": retrieved,
            "context": build_context_block(retrieved), 
            "refs": [
            {
                "corpus": (m.get("corpus") or detect_corpus_label(m.get("source_path", ""))),
                "file_name": os.path.basename(m.get("source_path", "")),
                "source_path": m.get("source_path"),
                "start_char": m.get("start_char"),
                "end_char": m.get("end_char"),
                "chunk_index": m.get("chunk_index"),
            }
            for (m, _, _) in retrieved
            ],
        }
