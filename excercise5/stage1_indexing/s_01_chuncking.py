"""
s_01_chunking.py

Entry point for Stage 2 (Temporal Indexing) of the Temporal RAG assignment.

Build temporally-aware chunk corpora by:
1) Iterating over debate text files in the raw corpus (Exercise 2 data).
2) Applying dataset-specific chunkers:
    - Semantic chunking for British Parliament debates.
    - US cleaner chunking for Congressional Record transcripts.
3) Extracting temporal metadata from filenames.
4) Writing the resulting chunks to JSONL files.

Outputs:
- outputs/chunks/debates_chunks/chunks_british_semantic.jsonl
- outputs/chunks/debates_chunks/chunks_us_clean.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import (
    CORPUS_DIRS,
    ensure_dirs,
    BRITISH_PARLIAMENT_DIR,
    US_CONGRESS_DIR,
    CHUNKS_BRITISH_SEMANTIC_JSONL,
    CHUNKS_US_CLEAN_JSONL,
)
from utils.text_utils import iter_text_files
from models.chunk import Chunk
from chunckers.semantic_chuncker import SemanticChunker
from chunckers.us_clean_chunker import USCleanerChunker


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

    semantic_chunker = SemanticChunker(
        max_words=660,
        sim_threshold=0.62,
        min_sentences_per_chunk=4,
        overlap_sentences=0,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    us_clean_chunker = USCleanerChunker()

    british_semantic: List[Chunk] = []
    us_clean: List[Chunk] = []

    for idx, fp in enumerate(files, 1):
        # fp is a string path
        path_obj = Path(fp)
        print(f"Processing {idx}/{len(files)}: {path_obj.name}")
        resolved = path_obj.resolve()

        if BRITISH_PARLIAMENT_DIR in resolved.parents:
            british_semantic.extend(semantic_chunker.build_chunks_for_file(fp))
        elif US_CONGRESS_DIR in resolved.parents:
            us_chunks = us_clean_chunker.build_chunks_for_file(fp)
            if us_chunks:
                us_clean.extend(us_chunks)

    write_jsonl(british_semantic, CHUNKS_BRITISH_SEMANTIC_JSONL)
    write_jsonl(us_clean, CHUNKS_US_CLEAN_JSONL)

    summarize_time_coverage(british_semantic, "british_semantic")
    summarize_time_coverage(us_clean, "us_clean")

    print(f"✅ Saved: {CHUNKS_BRITISH_SEMANTIC_JSONL}")
    print(f"✅ Saved: {CHUNKS_US_CLEAN_JSONL}")
    print(
        "Done. British semantic chunks={} | US clean chunks={}".format(
            len(british_semantic), len(us_clean)
        )
    )


if __name__ == "__main__":
    main()
