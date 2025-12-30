"""
Utility helpers for the RAG retriever package.

This module contains small, reusable, mostly pure functions used across the
retrieval codebase, such as:
- loading validated Chunk records from JSONL
- corpus label detection from file paths
- compact citation formatting for retrieved chunks
- building an LLM-ready context block with citations
- lightweight filesystem existence checks

These helpers intentionally do not load models or indices.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from models.chunk import Chunk

BRITISH = "british_parliament_debates"
US = "US_congress_debates"

"""Internal retrieved item: validated Chunk + score.

Notes:
    - For BM25, larger scores mean "more relevant".
    - For dense/FAISS, the score semantics depend on the vector store (often distance);
      treat it as an ordering signal rather than an absolute value.
"""
RetrievedChunk = Tuple[Chunk, float]


def read_chunks_jsonl(path: Path) -> List[Chunk]:
    """
    Load and validate Chunk records from a JSONL file.

    Raises:
        FileNotFoundError: If the JSONL file is missing.
        ValueError: If any JSONL line cannot be parsed into a Chunk.
    """
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return Chunk.read_jsonl(path)


def detect_corpus_label(path: str) -> str:
    """
    Infer a corpus label from a file path.

    Rules:
        - if the path contains BRITISH token -> "british"
        - if the path contains US token -> "us"
        - otherwise -> use the parent folder name, or "unknown"

    Note:
        This is a best-effort heuristic used when `Chunk.corpus` is missing.
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
        A string like: "us:debate_12.txt [123,456]".
    """
    source_path = chunk.source_path or ""
    corpus = chunk.corpus or detect_corpus_label(source_path)
    base = os.path.basename(source_path)
    return f"{corpus}:{base} [{chunk.start_char},{chunk.end_char}]"


def build_context_block(chunks: List[RetrievedChunk]) -> str:
    """
    Build a single context string from retrieved chunks.

    Format:
        Each chunk is prefixed with a bracketed citation id, followed by the text.

    Intended use:
        Provide this as the "context" input to an LLM prompt template.
    """
    lines: List[str] = []
    for chunk, _score in chunks:
        lines.append(f"[{short_source_id(chunk)}]")
        lines.append(chunk.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def assert_exists(paths: List[Path]) -> None:
    """
    Ensure required files/directories exist.

    Raises:
        FileNotFoundError: If any provided path does not exist.
    """
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def format_ref_id(ref: dict) -> str:
    """
    Format a compact citation id from a retriever `refs` item.

    Expected keys:
        corpus, file_name, start_char, end_char
    """
    corpus = ref.get("corpus") or "unknown"
    file_name = ref.get("file_name") or ""
    start_char = ref.get("start_char")
    end_char = ref.get("end_char")
    return f"{corpus}:{file_name} [{start_char},{end_char}]"

# Public API of this module (used to keep exports intentional and stable).
__all__ = [
    "BRITISH",
    "US",
    "RetrievedChunk",
    "read_chunks_jsonl",
    "detect_corpus_label",
    "short_source_id",
    "build_context_block",
    "assert_exists",
    "format_ref_id",
]
