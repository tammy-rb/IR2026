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

import json
from dataclasses import asdict
from typing import List

from paths import EXERCISE2_DATA_DIR, CHUNKS_DIR, ensure_dirs
from utils.text_utils import iter_text_files
from models import Chunk
from chunckers.fixed_chuncker import FixedChunker
from chunckers.semantic_chuncker import SemanticChunker


def write_jsonl(chunks: List[Chunk], out_path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def summarize_time_coverage(chunks: List[Chunk], label: str) -> None:
    with_date = sum(1 for c in chunks if c.doc_timestamp is not None)
    print(f"[{label}] chunks with timestamp: {with_date}/{len(chunks)}")


def main() -> None:
    ensure_dirs()

    files = iter_text_files(str(EXERCISE2_DATA_DIR))
    if not files:
        raise FileNotFoundError(f"No .txt files found under: {EXERCISE2_DATA_DIR}")

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

    out_fixed = CHUNKS_DIR / "chunks_fixed.jsonl"
    out_sem = CHUNKS_DIR / "chunks_semantic.jsonl"

    write_jsonl(all_fixed, out_fixed)
    write_jsonl(all_sem, out_sem)

    summarize_time_coverage(all_fixed, "fixed")
    summarize_time_coverage(all_sem, "semantic")

    print(f"✅ Saved: {out_fixed}")
    print(f"✅ Saved: {out_sem}")
    print(f"Done. Fixed chunks={len(all_fixed)} | Semantic chunks={len(all_sem)}")


if __name__ == "__main__":
    main()
