"""
s_02_embedding.py

Exercise 5 - Stage 1 (Indexing): Semantic embedding script.

Purpose:
- Build dense (OpenAI) embeddings for semantic chunk corpora produced by s_01_chuncking.py.
- Preserve temporal metadata (doc_date_iso + doc_timestamp) for time-aware retrieval.

Chunk schema:
- Chunks are defined by the canonical Chunk class in chunk.py and stored as JSONL.

Outputs:
- outputs/embeddings/openai_qdrant/british_parliament_semantic/
- outputs/embeddings/openai_qdrant/us_congress_semantic/

Requirements:
- OPENAI_API_KEY must be available (e.g., via .env loaded by python-dotenv)
- Run the Qdrant service before executing OpenAI embeddings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import (
    CHUNKS_BRITISH_SEMANTIC_JSONL,
    CHUNKS_US_CLEAN_JSONL,
    OPENAI_QDRANT_BRITISH_SEMANTIC_DIR,
    OPENAI_QDRANT_US_CONGRESS_SEMANTIC_DIR,
    ensure_dirs,
)
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_BRITISH_PARLIAMENT,
    QDRANT_COLLECTION_US_CONGRESS,
    OPENAI_EMBED_MODEL,
    SEMANTIC_DOCS_BATCH_SIZE,
    SEMANTIC_EMBED_BATCH_SIZE,
    VECTOR_DISTANCE,
)
from embedders.openai_embedder import OpenAIEmbedder

def _embed_corpus(label: str) -> None:
    if label == "british":
        jsonl_path = CHUNKS_BRITISH_SEMANTIC_JSONL
        output_dir = OPENAI_QDRANT_BRITISH_SEMANTIC_DIR
        collection = QDRANT_COLLECTION_BRITISH_PARLIAMENT
        banner = "British Parliament semantic chunks"
    elif label == "us":
        jsonl_path = CHUNKS_US_CLEAN_JSONL
        output_dir = OPENAI_QDRANT_US_CONGRESS_SEMANTIC_DIR
        collection = QDRANT_COLLECTION_US_CONGRESS
        banner = "US Congress cleaned chunks"
    else:
        raise ValueError(f"Unknown corpus label: {label}")

    print(f"\n[OpenAI] {banner}")
    print("-" * 70)

    embedder = OpenAIEmbedder(
        output_dir=output_dir,
        docs_batch_size=SEMANTIC_DOCS_BATCH_SIZE,
        embed_batch_size=SEMANTIC_EMBED_BATCH_SIZE,
        collection_name=collection,
        model=OPENAI_EMBED_MODEL,
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        distance=VECTOR_DISTANCE,
    )
    embedder.embed_chunks(jsonl_path)


def main(argv: list[str] | None = None) -> None:
    """Build OpenAI embeddings for British and/or US semantic corpora."""
    parser = argparse.ArgumentParser(description="Exercise 5 - Stage 1: Build semantic embedding indexes")
    parser.add_argument(
        "--corpus",
        choices=["both", "british", "us"],
        default="both",
        help="Which semantic corpus to embed (default: both)",
    )
    args = parser.parse_args(argv)

    ensure_dirs()

    print("\n" + "=" * 70)
    print("Exercise 5 - Stage 1: Indexing")
    print("Building semantic embeddings for British Parliament and US Congress")
    print("=" * 70)

    targets = []
    if args.corpus in ("both", "british"):
        targets.append("british")
    if args.corpus in ("both", "us"):
        targets.append("us")

    for label in targets:
        _embed_corpus(label)

    print("\n" + "=" * 70)
    print("Semantic embeddings completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
