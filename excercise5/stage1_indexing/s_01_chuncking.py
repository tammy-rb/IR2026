"""
s_01_chunking.py

Entry point for Stage 2 (Temporal Indexing) of the Temporal RAG assignment.

Build temporally-aware chunk corpora by:
1) Iterating over all debate text files in the raw corpus (Exercise 2 data).
2) Applying two chunking strategies:
   - Fixed-size chunking with sentence overlap.
   - Semantic chunking based on embedding similarity.
3) Extracting and attaching temporal metadata to each chunk (from filename).
4) Writing the resulting chunks to JSONL files.

Outputs:
- outputs/chunks/chunks_fixed.jsonl
- outputs/chunks/chunks_semantic.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import CORPUS_DIRS, CHUNKS_DIR, ensure_dirs, CHUNKS_SEMANTIC_JSONL, CHUNKS_FIXED_JSONL
from utils.text_utils import iter_text_files
from models.chunk import Chunk
from chunckers.fixed_chuncker import FixedChunker
from chunckers.semantic_chuncker import SemanticChunker


def write_jsonl(chunks: List[Chunk], out_path) -> None:
    Chunk.write_jsonl(chunks, out_path)


def summarize_time_coverage(chunks: List[Chunk], label: str) -> None:
    with_date = sum(1 for c in chunks if c.doc_timestamp is not None)
    print(f"[{label}] chunks with timestamp: {with_date}/{len(chunks)}")


def main() -> None:
    ensure_dirs()

    # Collect files from all corpus directories
    files = []
    for corpus_dir in CORPUS_DIRS:
        files.extend(iter_text_files(str(corpus_dir)))
    
    if not files:
        raise FileNotFoundError(f"No .txt files found in corpus directories: {CORPUS_DIRS}")

    fixed_chunker = FixedChunker(max_words=660, overlap_sentences=3)
    semantic_chunker = SemanticChunker(
        max_words=660,
        sim_threshold=0.62,
        min_sentences_per_chunk=4,
        overlap_sentences=0,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    all_fixed: List[Chunk] = []
    all_sem: List[Chunk] = []

    for idx, fp in enumerate(files, 1):
        # fp is a string path
        print(f"Processing {idx}/{len(files)}: {fp.split('/')[-1]}")
        all_fixed.extend(fixed_chunker.build_chunks_for_file(fp))
        all_sem.extend(semantic_chunker.build_chunks_for_file(fp))

    out_fixed = CHUNKS_FIXED_JSONL
    out_sem = CHUNKS_SEMANTIC_JSONL

    write_jsonl(all_fixed, out_fixed)
    write_jsonl(all_sem, out_sem)

    summarize_time_coverage(all_fixed, "fixed")
    summarize_time_coverage(all_sem, "semantic")

    print(f"✅ Saved: {out_fixed}")
    print(f"✅ Saved: {out_sem}")
    print(f"Done. Fixed chunks={len(all_fixed)} | Semantic chunks={len(all_sem)}")


if __name__ == "__main__":
    main()
