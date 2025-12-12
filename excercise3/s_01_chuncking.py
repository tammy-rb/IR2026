# chunking.py
from __future__ import annotations

import os
import re
import json
import glob
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BRITISH = "british_parliament_debates"
US = "US_congress_debates"
DATA_DIR = os.path.join(BASE_DIR, "..", "excercise2", "data")
OUTPUT_CHUNCKS_DIR = os.path.join(BASE_DIR, "outputs", "chunks")

# Fixed chunking requirements (assignment):
# - chunks contain full sentences only
# - max 660 words per chunk (unless a single sentence exceeds 660)
# - overlap of 3 sentences between consecutive chunks
FIXED_MAX_WORDS = 660
FIXED_OVERLAP_SENTENCES = 3

# Semantic chunking requirements/knobs:
# - detect semantic boundaries between adjacent sentences using embeddings
# - enforce max 660 words per chunk (unless a single sentence exceeds 660)
# - avoid very small chunks (minimum number of sentences)
SEM_MAX_WORDS = 660
SEM_SIM_THRESHOLD = 0.62
SEM_MIN_SENTENCES_PER_CHUNK = 4
SEM_OVERLAP_SENTENCES = 0  # semantic overlap is optional; assignment does not require it

# Embedding model (Sentence-Transformers / SBERT-style)
SEM_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# Data structures
# ============================================================

@dataclass
class Chunk:
    """
    Represents one chunk extracted from a document.

    We store both:
      - pointers to the original document (start_char/end_char)
      - the actual text (for convenience + debugging + downstream indexing)
    """
    doc_id: str
    source_path: str
    corpus: str                 # "british" / "us" / fallback folder name
    chunking_method: str        # "fixed" / "semantic"
    chunk_index: int

    # Character offsets in the ORIGINAL file text:
    start_char: int             # inclusive
    end_char: int               # exclusive

    # Convenience: duplicated content (can be removed later if needed)
    text: str

    # Cached token/word count (for analysis/debug)
    num_words: int


# ============================================================
# File utilities
# ============================================================

def extract_text(file_path: str) -> str:
    """Read a UTF-8 text file and return its contents."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def iter_text_files(root_dir: str) -> List[str]:
    """
    Recursively collect all .txt files under root_dir.

    glob pattern:
      root_dir/**/*.txt
    where ** means "any subfolder depth" (recursive=True).
    """
    pattern = os.path.join(root_dir, "**", "*.txt")
    return sorted(glob.glob(pattern, recursive=True))


def doc_id_from_path(path: str) -> str:
    """
    Build a stable document id from the filename (without extension).
    Example: '/a/b/c/debate_12.txt' -> 'debate_12'
    """
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def detect_corpus_label(path: str) -> str:
    """
    Determine which corpus the file belongs to (british/us) using folder name.
    Falls back to the file's parent folder name if neither token is found.
    """
    low = path.lower()
    if BRITISH.lower() in low:
        return "british"
    if US.lower() in low:
        return "us"
    return os.path.basename(os.path.dirname(path))


# ============================================================
# Sentence splitting with character spans (offsets)
# ============================================================

# Sentence boundary heuristic:
# - split at whitespace that follows a sentence-ending punctuation mark (. ! ?)
# - we match the whitespace itself (so we know exactly where the boundary is)
# - the punctuation remains part of the previous sentence
_SENT_BOUNDARY_RE = re.compile(r"(?<=[\.\!\?])\s+")


def split_to_sentences_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into sentences and return their character spans in the ORIGINAL text.

    Returns:
        List of tuples: (sentence_text, start_char, end_char_exclusive)

    Notes:
    - We MUST preserve character offsets. Therefore, we avoid normalizing whitespace
      in a way that changes string length.
    - We do replace NBSP (U+00A0) with a normal space because it is a 1->1 replacement,
      preserving offsets while improving tokenization consistency.
    """
    if not text:
        return []

    # Replace NBSP with space (1 char -> 1 char) to keep offsets stable.
    text = text.replace("\u00a0", " ")

    spans: List[Tuple[str, int, int]] = []
    start = 0

    # Each match corresponds to the whitespace after a sentence-ending punctuation.
    # - m.start() is the index where the whitespace begins (end of sentence)
    # - m.end() is the index where whitespace ends (start of next sentence)
    for m in _SENT_BOUNDARY_RE.finditer(text):
        end = m.start()
        if end > start:
            seg = text[start:end]
            if seg.strip():
                spans.append((seg, start, end))
        start = m.end()

    # Add trailing text after the last boundary (the final sentence).
    if start < len(text):
        tail = text[start:]
        if tail.strip():
            spans.append((tail, start, len(text)))

    return spans


def count_words(s: str) -> int:
    """Simple whitespace-based word count (sufficient for the assignment constraints)."""
    return len([w for w in s.split() if w])


# ============================================================
# Embeddings (Sentence-Transformers is mandatory)
# ============================================================

def load_sentence_transformer(model_name: str = SEM_EMBEDDING_MODEL):
    """
    Load a Sentence-Transformers model.

    We intentionally do NOT provide a TF-IDF fallback because semantic chunking
    is required to be embedding-based.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for semantic chunking. "
            "Install it with: pip install sentence-transformers"
        ) from e

    return SentenceTransformer(model_name)


# Global cache so the model is loaded once per run.
_ST_MODEL = None


def embed_sentences(sentences: List[str]):
    """
    Encode sentences into normalized embeddings using Sentence-Transformers.

    normalize_embeddings=True means each vector is L2-normalized (||v|| = 1),
    which makes cosine similarity equivalent to dot product.
    """
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = load_sentence_transformer(SEM_EMBEDDING_MODEL)

    return _ST_MODEL.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def cosine_sim(a, b) -> float:
    """
    Cosine similarity between two vectors.

    Even though embeddings are normalized, we compute the full cosine formula
    to remain robust if normalization settings change later.
    """
    import numpy as np

    a = np.asarray(a).reshape(1, -1)
    b = np.asarray(b).reshape(1, -1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float((a @ b.T)[0, 0] / denom)


# ============================================================
# Fixed chunking (size-based) with sentence overlap
# ============================================================

def fixed_chunk_sentences_with_spans(
    sentence_spans: List[Tuple[str, int, int]],
    max_words: int = FIXED_MAX_WORDS,
    overlap_sentences: int = FIXED_OVERLAP_SENTENCES,
) -> List[Tuple[int, int, int, int]]:
    """
    Create fixed-size chunks by accumulating full sentences until max_words.

    Returns:
        List of (start_sentence_idx, end_sentence_idx_inclusive, start_char, end_char_exclusive)

    Rules:
    - Never split a sentence.
    - Target max_words per chunk; if a single sentence exceeds max_words, it forms a chunk alone.
    - Overlap is defined in number of sentences (previous chunk's last N sentences).
    """
    n = len(sentence_spans)
    chunks: List[Tuple[int, int, int, int]] = []

    i = 0  # next sentence index to consider
    while i < n:
        start_idx = i
        cur_words = 0
        end_idx = i - 1  # will be updated as we include sentences

        # Grow current chunk by adding sentences until we hit the word limit.
        while i < n:
            sent_text, _, _ = sentence_spans[i]
            sent_words = count_words(sent_text)

            # If adding this sentence would exceed the limit, stop (unless chunk is empty).
            if cur_words > 0 and (cur_words + sent_words) > max_words:
                break

            cur_words += sent_words
            end_idx = i
            i += 1

            # If we reached/exceeded the limit, stop.
            if cur_words >= max_words:
                break

        # Convert sentence indices to character offsets.
        start_char = sentence_spans[start_idx][1]
        end_char = sentence_spans[end_idx][2]
        chunks.append((start_idx, end_idx, start_char, end_char))

        # If no more sentences, we are done.
        if i >= n:
            break

        # Move back to create overlap for the next chunk.
        # Next chunk begins overlap_sentences before the next sentence after end_idx.
        i = max(0, (end_idx + 1) - overlap_sentences)

        # Safety: if overlap prevents progress, force forward movement.
        if i <= start_idx and end_idx >= start_idx:
            i = end_idx + 1

    return chunks


# ============================================================
# Semantic chunking (embedding-based segmentation)
# ============================================================

def semantic_chunk_sentences_with_spans(
    sentence_spans: List[Tuple[str, int, int]],
    max_words: int = SEM_MAX_WORDS,
    sim_threshold: float = SEM_SIM_THRESHOLD,
    min_sentences_per_chunk: int = SEM_MIN_SENTENCES_PER_CHUNK,
    overlap_sentences: int = SEM_OVERLAP_SENTENCES,
) -> List[Tuple[int, int, int, int]]:
    """
    Semantic chunking:
    1) Split document into sentences (already done: sentence_spans)
    2) Compute sentence embeddings
    3) Compute similarity between adjacent sentences
    4) Mark a semantic boundary after sentence i if sim(i, i+1) < sim_threshold
    5) Build chunks while enforcing:
       - max_words limit (forced boundary)
       - min_sentences_per_chunk (avoid tiny chunks)

    Returns:
        List of (start_sentence_idx, end_sentence_idx_inclusive, start_char, end_char_exclusive)
    """
    n = len(sentence_spans)
    if n == 0:
        return []

    # Extract sentence texts and compute embeddings.
    sent_texts = [t for (t, _, _) in sentence_spans]
    E = embed_sentences(sent_texts)

    # Similarity between sentence i and i+1 (adjacent pairs).
    sims: List[float] = []
    for i in range(n - 1):
        sims.append(cosine_sim(E[i], E[i + 1]))

    # boundary_after contains indices i such that the boundary is after i (between i and i+1).
    boundary_after = {i for i, s in enumerate(sims) if s < sim_threshold}

    chunks: List[Tuple[int, int, int, int]] = []

    # Current chunk window state
    start_idx = 0
    i = 0
    cur_words = 0
    cur_sent_count = 0

    def flush(end_idx_inclusive: int) -> None:
        """
        Finalize the current chunk from start_idx to end_idx_inclusive.
        Converts sentence indices to character offsets and appends to chunks.
        Resets counters and sets start_idx for the next chunk (with optional overlap).
        """
        nonlocal start_idx, cur_words, cur_sent_count, chunks

        start_char = sentence_spans[start_idx][1]
        end_char = sentence_spans[end_idx_inclusive][2]
        chunks.append((start_idx, end_idx_inclusive, start_char, end_char))

        # Next chunk start (overlap is optional; usually 0 here).
        next_start = end_idx_inclusive + 1 - overlap_sentences
        start_idx = max(0, next_start)

        # Reset stats for the next chunk.
        cur_words = 0
        cur_sent_count = 0

    while i < n:
        sent_text, _, _ = sentence_spans[i]
        sent_words = count_words(sent_text)

        # Forced boundary: adding sentence i would exceed max_words.
        # In that case, flush the chunk up to i-1 and re-process sentence i in a new chunk.
        if cur_sent_count > 0 and (cur_words + sent_words) > max_words:
            flush(i - 1)
            i = start_idx  # restart from the new chunk start (matters if overlap > 0)
            continue

        # Add the current sentence to the chunk.
        cur_words += sent_words
        cur_sent_count += 1

        is_last = (i == n - 1)

        # Semantic boundary: if there's a meaning shift after i AND the chunk is not too small,
        # close the chunk at i.
        if not is_last:
            if (i in boundary_after) and (cur_sent_count >= min_sentences_per_chunk):
                flush(i)
                i = start_idx
                continue

        # End of document: always flush the final chunk.
        if is_last:
            flush(i)
            break

        # Otherwise, continue to next sentence.
        i += 1

        # Extra safety: in rare edge cases (e.g., overlap/backtracking logic),
        # ensure we do not get stuck without making progress.
        # If i == start_idx and start_idx does not move forward relative to the last chunk,
        # reset counters so we can proceed.
        if len(chunks) > 0 and start_idx <= chunks[-1][0] and start_idx == i:
            cur_words = 0
            cur_sent_count = 0

    return chunks


# ============================================================
# Build + save chunks for all files
# ============================================================

def build_chunks_for_file(file_path: str) -> Dict[str, List[Chunk]]:
    """
    Build both fixed and semantic chunks for a single file.
    We store offsets AND also materialize the chunk text for convenience.
    """
    text = extract_text(file_path)
    sentence_spans = split_to_sentences_with_spans(text)

    corpus = detect_corpus_label(file_path)
    doc_id = doc_id_from_path(file_path)

    fixed_ranges = fixed_chunk_sentences_with_spans(sentence_spans)
    sem_ranges = semantic_chunk_sentences_with_spans(sentence_spans)

    fixed_chunks: List[Chunk] = []
    for idx, (_s_i, _e_i, start_char, end_char) in enumerate(fixed_ranges):
        chunk_text = text[start_char:end_char]
        fixed_chunks.append(
            Chunk(
                doc_id=doc_id,
                source_path=file_path,
                corpus=corpus,
                chunking_method="fixed",
                chunk_index=idx,
                start_char=start_char,
                end_char=end_char,
                text=chunk_text,
                num_words=count_words(chunk_text),
            )
        )

    sem_chunks: List[Chunk] = []
    for idx, (_s_i, _e_i, start_char, end_char) in enumerate(sem_ranges):
        chunk_text = text[start_char:end_char]
        sem_chunks.append(
            Chunk(
                doc_id=doc_id,
                source_path=file_path,
                corpus=corpus,
                chunking_method="semantic",
                chunk_index=idx,
                start_char=start_char,
                end_char=end_char,
                text=chunk_text,
                num_words=count_words(chunk_text),
            )
        )

    return {"fixed": fixed_chunks, "semantic": sem_chunks}


def write_jsonl(chunks: List[Chunk], out_path: str) -> None:
    """Write a list of Chunk objects into a JSONL file (one JSON object per line)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


# ============================================================
# Utilities: materialize + sanity checks
# ============================================================

def materialize_from_offsets(source_path: str, start_char: int, end_char: int) -> str:
    """Reconstruct chunk text directly from file using stored character offsets."""
    text = extract_text(source_path)
    return text[start_char:end_char]


def sanity_check_chunks(chunks: List[Chunk]) -> None:
    """
    Validate that offsets and stored text are consistent.
    This helps catch bugs in sentence splitting or offset computation.
    """
    cache: Dict[str, str] = {}

    for c in chunks:
        if c.source_path not in cache:
            cache[c.source_path] = extract_text(c.source_path)

        full_text = cache[c.source_path]

        # Offsets must be in bounds and properly ordered.
        assert 0 <= c.start_char <= c.end_char <= len(full_text), (
            f"Invalid offsets in {c.source_path} chunk_index={c.chunk_index}: "
            f"{c.start_char}-{c.end_char} (len={len(full_text)})"
        )

        # Slicing with offsets must reproduce the stored text exactly.
        materialized = full_text[c.start_char:c.end_char]
        assert materialized == c.text, (
            f"Offset-text mismatch in {c.source_path} chunk_index={c.chunk_index}"
        )

        # Word count must match cached value.
        w = count_words(c.text)
        assert w == c.num_words, (
            f"num_words mismatch in {c.source_path} chunk_index={c.chunk_index}: "
            f"stored={c.num_words}, computed={w}"
        )


def run_sanity_checks(all_fixed: List[Chunk], all_sem: List[Chunk]) -> None:
    """Run sanity checks for both chunking strategies."""
    sanity_check_chunks(all_fixed)
    sanity_check_chunks(all_sem)
    print("Sanity checks passed ✅")


# ============================================================
# Main entry point
# ============================================================

def main() -> None:
    files = iter_text_files(DATA_DIR)
    if not files:
        raise FileNotFoundError(f"No .txt files found under: {DATA_DIR}")

    all_fixed: List[Chunk] = []
    all_sem: List[Chunk] = []

    for idx, fp in enumerate(files, 1):
        print(f"Processing file {idx}/{len(files)}: {os.path.basename(fp)}")
        result = build_chunks_for_file(fp)
        all_fixed.extend(result["fixed"])
        all_sem.extend(result["semantic"])

    write_jsonl(all_fixed, os.path.join(OUTPUT_CHUNCKS_DIR, "chunks_fixed.jsonl"))
    write_jsonl(all_sem, os.path.join(OUTPUT_CHUNCKS_DIR, "chunks_semantic.jsonl"))

    run_sanity_checks(all_fixed, all_sem)

    print(f"Done. Fixed chunks: {len(all_fixed)} | Semantic chunks: {len(all_sem)}")
    print(f"Saved to: {OUTPUT_CHUNCKS_DIR}")


if __name__ == "__main__":
    main()
