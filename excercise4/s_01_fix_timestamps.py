"""
s_01_fix_timestamps.py

Exercise 4 — Stage 2 (Temporal Indexing): Fix missing temporal metadata.

Purpose:
- Read chunk JSONL files and ensure all chunks have temporal metadata.
- If doc_date_iso or doc_timestamp are missing, extract them from source_path.
- Write updated chunks back to the same JSONL files.

This script updates chunks in-place to ensure temporal consistency.

Inputs:
- outputs/chunks/chunks_fixed.jsonl
- outputs/chunks/chunks_semantic.jsonl

Outputs:
- Same files, updated with temporal metadata.

Run:
  python s_01_fix_timestamps.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List

from models.chunk import Chunk
from paths import CHUNKS_DIR
from utils.time_utils import timestamp_from_path

def fix_timestamps_in_chunks(chunks: List[Chunk]) -> tuple[List[Chunk], int]:
    """
    Fix missing temporal metadata in chunks by extracting from source_path.
    
    Returns:
        (updated_chunks, updated_count)
    """
    updated_count = 0
    updated_chunks: List[Chunk] = []

    for c in chunks:
        needs_fix = (c.doc_date_iso is None) or (c.doc_timestamp is None)
        if not needs_fix:
            updated_chunks.append(c)
            continue

        iso_date, timestamp = timestamp_from_path(c.source_path)
        if iso_date is None or timestamp is None:
            updated_chunks.append(c)
            continue

        updated_chunks.append(
            replace(
                c,
                doc_date_iso=iso_date,
                doc_timestamp=timestamp,
            )
        )
        updated_count += 1

    return updated_chunks, updated_count


def process_jsonl_file(jsonl_path: Path) -> None:
    """
    Process a single JSONL file: fix timestamps and write back.
    
    Args:
        jsonl_path: Path to the JSONL file to process.
    """
    if not jsonl_path.exists():
        print(f"⚠️  File not found: {jsonl_path}")
        return
    
    print(f"\nProcessing: {jsonl_path.name}")
    print("-" * 70)
    
    # Read chunks
    chunks = Chunk.read_jsonl(jsonl_path)
    total = len(chunks)
    print(f"  📄 Total chunks: {total}")
    
    # Fix timestamps
    chunks, updated = fix_timestamps_in_chunks(chunks)
    print(f"  🔧 Updated chunks: {updated}")
    
    # Write back
    Chunk.write_jsonl(chunks, jsonl_path)
    print(f"  ✅ Saved to: {jsonl_path}")


def main() -> None:
    """Fix timestamps in both fixed and semantic chunk JSONL files."""
    print("\n" + "=" * 70)
    print("Exercise 4 — Fixing Temporal Metadata in Chunks")
    print("=" * 70)
    
    fixed_jsonl = CHUNKS_DIR / "chunks_fixed.jsonl"
    semantic_jsonl = CHUNKS_DIR / "chunks_semantic.jsonl"
    
    # Process both files
    process_jsonl_file(fixed_jsonl)
    process_jsonl_file(semantic_jsonl)
    
    print("\n" + "=" * 70)
    print("✅ Timestamp fixing completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
