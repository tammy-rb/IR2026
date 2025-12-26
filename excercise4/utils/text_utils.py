"""
utils/text_utils.py

Text and preprocessing helpers for Exercise 4 chunking.

This module provides:
- File loading (UTF-8)
- Recursive discovery of .txt debate files
- Document id and corpus label derivation from paths
- Sentence splitting with stable character offsets
- Simple whitespace-based word counting (used for chunk size constraints)

Design note:
These utilities are intentionally stateless and do not depend on project paths.
Path construction is centralized in paths.py and used by the stage scripts.
"""

from __future__ import annotations

import os
import re
import glob
from typing import List, Tuple

BRITISH = "british_parliament_debates"
US = "US_congress_debates"

# Sentence boundary heuristic:
# split on whitespace that follows a sentence-ending punctuation mark (. ! ?).
_SENT_BOUNDARY_RE = re.compile(r"(?<=[\.\!\?])\s+")


def extract_text(file_path: str) -> str:
    """
    Read a UTF-8 text file and return its content.

    Args:
        file_path: Path to a .txt file.

    Returns:
        Full file content as a string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def iter_text_files(root_dir: str) -> List[str]:
    """
    Recursively collect all .txt files under root_dir.

    Args:
        root_dir: Root directory to scan.

    Returns:
        Sorted list of file paths (strings).
    """
    pattern = os.path.join(root_dir, "**", "*.txt")
    return sorted(glob.glob(pattern, recursive=True))


def doc_id_from_path(path: str) -> str:
    """
    Create a stable document id from the filename without extension.

    Example:
        "/a/b/debates2023-07-10.txt" -> "debates2023-07-10"

    Args:
        path: File path.

    Returns:
        Filename without extension.
    """
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def detect_corpus_label(path: str) -> str:
    """
    Determine corpus label based on folder name in the path.

    Returns:
        "british" if path contains "british_parliament_debates"
        "us" if path contains "US_congress_debates"
        otherwise: the immediate parent folder name.

    Args:
        path: File path.

    Returns:
        Corpus label string.
    """
    low = path.lower()
    if BRITISH.lower() in low:
        return "british"
    if US.lower() in low:
        return "us"
    return os.path.basename(os.path.dirname(path))


def split_to_sentences_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Split a document into sentences while preserving original character offsets.

    Returns:
        List of tuples: (sentence_text, start_char, end_char_exclusive)

    Notes:
        - We avoid whitespace normalization that changes string length, to keep offsets valid.
        - NBSP (U+00A0) is replaced with a normal space (1 char -> 1 char),
          so offsets remain stable while tokenization improves.
    """
    if not text:
        return []

    text = text.replace("\u00a0", " ")

    spans: List[Tuple[str, int, int]] = []
    start = 0

    for m in _SENT_BOUNDARY_RE.finditer(text):
        end = m.start()
        if end > start:
            seg = text[start:end]
            if seg.strip():
                spans.append((seg, start, end))
        start = m.end()

    if start < len(text):
        tail = text[start:]
        if tail.strip():
            spans.append((tail, start, len(text)))

    return spans


def count_words(s: str) -> int:
    """
    Count words using simple whitespace splitting.

    This is sufficient for enforcing the assignment constraint:
    "max 660 words per chunk".

    Args:
        s: Input text.

    Returns:
        Number of non-empty whitespace-delimited tokens.
    """
    return len([w for w in s.split() if w])
