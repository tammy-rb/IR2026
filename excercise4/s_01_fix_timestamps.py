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

import json
from pathlib import Path
from typing import Any, Dict, List

from paths import CHUNKS_DIR
from utils.time_utils import timestamp_from_path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.
    
    Args:
        path: Path to a JSONL file.
    
    Returns:
        List of parsed JSON objects (dicts).
    """
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(chunks: List[Dict[str, Any]], path: Path) -> None:
    """
    Write a list of chunk dicts to a JSONL file.
    
    Args:
        chunks: List of chunk dictionaries.
        path: Output JSONL file path.
    """
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def fix_timestamps_in_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Fix missing temporal metadata in chunks by extracting from source_path.
    
    Args:
        chunks: List of chunk dictionaries.
    
    Returns:
        Number of chunks that were updated.
    """
    updated_count = 0
    
    for chunk in chunks:
        # Check if temporal fields are missing or None
        needs_fix = (
            chunk.get("doc_date_iso") is None 
            or chunk.get("doc_timestamp") is None
        )
        
        if needs_fix:
            source_path = chunk.get("source_path", "")
            if source_path:
                iso_date, timestamp = timestamp_from_path(source_path)
                
                if iso_date is not None and timestamp is not None:
                    chunk["doc_date_iso"] = iso_date
                    chunk["doc_timestamp"] = timestamp
                    updated_count += 1
    
    return updated_count


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
    chunks = read_jsonl(jsonl_path)
    total = len(chunks)
    print(f"  📄 Total chunks: {total}")
    
    # Fix timestamps
    updated = fix_timestamps_in_chunks(chunks)
    print(f"  🔧 Updated chunks: {updated}")
    
    # Write back
    write_jsonl(chunks, jsonl_path)
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
