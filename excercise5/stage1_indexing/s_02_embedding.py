"""
s_02_embedding.py

Exercise 4 — Stage 2 (Temporal Indexing): Unified embedding script.

Purpose:
- Build both sparse (BM25) and dense (OpenAI) embeddings over chunk corpora
  produced by s_01_chunking.py.
- Preserve temporal metadata (doc_date_iso + doc_timestamp) for time-aware retrieval.

Chunk schema:
- Chunks are defined by the canonical `Chunk` class in chunk.py and stored as JSONL.

Outputs:
- outputs/embeddings/bm25/fixed/             (BM25 for fixed chunking)
- outputs/embeddings/bm25/semantic/          (BM25 for semantic chunking)
- outputs/embeddings/openai_qdrant/fixed/    (OpenAI+Qdrant for fixed chunking)
- outputs/embeddings/openai_qdrant/semantic/ (OpenAI+Qdrant for semantic chunking)

Requirements:
- OPENAI_API_KEY must be available (e.g., via .env loaded by python-dotenv)
- run qdrant docker container for OpenAI+Qdrant embeddings
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
    CHUNKS_DIR,
    BM25_DIR,
    EMBEDDINGS_DIR,
    OPENAI_QDRANT_FIXED_DIR,
    OPENAI_QDRANT_SEMANTIC_DIR,
    ensure_dirs,
)
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_FIXED,
    QDRANT_COLLECTION_SEMANTIC,
    OPENAI_EMBED_MODEL,
    FIXED_DOCS_BATCH_SIZE,
    FIXED_EMBED_BATCH_SIZE,
    SEMANTIC_DOCS_BATCH_SIZE,
    SEMANTIC_EMBED_BATCH_SIZE,
    VECTOR_DISTANCE,
)
from embedders.bm25_embedder import BM25Embedder
from embedders.openai_embedder import OpenAIEmbedder

# Input JSONL files
CHUNKS_FIXED_JSONL = CHUNKS_DIR / "chunks_fixed.jsonl"
CHUNKS_SEM_JSONL = CHUNKS_DIR / "chunks_semantic.jsonl"


def build_bm25_embeddings(*, strategy: str = "both") -> None:
    """Build BM25 indexes for fixed and/or semantic chunk corpora."""
    print("\n" + "=" * 70)
    print("Building BM25 Embeddings")
    print("=" * 70)
    
    if strategy in ("both", "fixed"):
        print("\n[BM25] Fixed chunking strategy")
        print("-" * 70)
        fixed_embedder = BM25Embedder(output_dir=BM25_DIR / "fixed")
        fixed_embedder.embed_chunks(CHUNKS_FIXED_JSONL)

    if strategy in ("both", "semantic"):
        print("\n[BM25] Semantic chunking strategy")
        print("-" * 70)
        semantic_embedder = BM25Embedder(output_dir=BM25_DIR / "semantic")
        semantic_embedder.embed_chunks(CHUNKS_SEM_JSONL)


def build_openai_embeddings(*, strategy: str = "both") -> None:
    """Build OpenAI + Qdrant indexes for fixed and/or semantic chunk corpora."""
    print("\n" + "=" * 70)
    print("Building OpenAI Embeddings")
    print("=" * 70)
    
    if strategy in ("both", "fixed"):
        print("\n[OpenAI] Fixed chunking strategy")
        print("-" * 70)
        fixed_embedder = OpenAIEmbedder(
            output_dir=OPENAI_QDRANT_FIXED_DIR,
            docs_batch_size=FIXED_DOCS_BATCH_SIZE,
            embed_batch_size=FIXED_EMBED_BATCH_SIZE,
            collection_name=QDRANT_COLLECTION_FIXED,
            model=OPENAI_EMBED_MODEL,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            distance=VECTOR_DISTANCE,
        )
        fixed_embedder.embed_chunks(CHUNKS_FIXED_JSONL)

    if strategy in ("both", "semantic"):
        print("\n[OpenAI] Semantic chunking strategy")
        print("-" * 70)
        semantic_embedder = OpenAIEmbedder(
            output_dir=OPENAI_QDRANT_SEMANTIC_DIR,
            docs_batch_size=SEMANTIC_DOCS_BATCH_SIZE,
            embed_batch_size=SEMANTIC_EMBED_BATCH_SIZE,
            collection_name=QDRANT_COLLECTION_SEMANTIC,
            model=OPENAI_EMBED_MODEL,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            distance=VECTOR_DISTANCE,
        )
        semantic_embedder.embed_chunks(CHUNKS_SEM_JSONL)


def main(argv: list[str] | None = None) -> None:
    """Build embeddings with optional stage/strategy selection."""
    parser = argparse.ArgumentParser(description="Exercise 4 — Stage 2: Build embedding indexes")
    parser.add_argument(
        "--only",
        choices=["all", "bm25", "openai"],
        default="all",
        help="Which embedding family to run (default: all)",
    )
    parser.add_argument(
        "--strategy",
        choices=["both", "fixed", "semantic"],
        default="both",
        help="Which chunking strategy to process (default: both)",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    
    print("\n" + "=" * 70)
    print("Exercise 4 — Stage 2: Temporal Indexing")
    print("Building embeddings for fixed and semantic chunking strategies")
    print("=" * 70)
    
    if args.only in ("all", "bm25"):
        build_bm25_embeddings(strategy=args.strategy)

    if args.only in ("all", "openai"):
        build_openai_embeddings(strategy=args.strategy)
    
    print("\n" + "=" * 70)
    print("✅ All embeddings completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
