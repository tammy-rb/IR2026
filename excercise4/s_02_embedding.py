"""
s_02_embedding.py

Exercise 4 — Stage 2 (Temporal Indexing): Unified embedding script.

Purpose:
- Build both sparse (BM25) and dense (OpenAI) embeddings over chunk corpora
  produced by s_01_chunking.py.
- Preserve temporal metadata (doc_date_iso + doc_timestamp) for time-aware retrieval.

Outputs:
- outputs/bm25/fixed/          (BM25 for fixed chunking)
- outputs/bm25/semantic/       (BM25 for semantic chunking)
- outputs/embeddings/openai/fixed_faiss/     (OpenAI for fixed chunking)
- outputs/embeddings/openai/semantic_faiss/  (OpenAI for semantic chunking)

Requirements:
- OPENAI_API_KEY must be available (e.g., via .env loaded by python-dotenv)
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from paths import CHUNKS_DIR, BM25_DIR, EMBEDDINGS_DIR, ensure_dirs
from embedders.bm25_embedder import BM25Embedder
from embedders.openai_embedder import OpenAIEmbedder

# Input JSONL files
CHUNKS_FIXED_JSONL = CHUNKS_DIR / "chunks_fixed.jsonl"
CHUNKS_SEM_JSONL = CHUNKS_DIR / "chunks_semantic.jsonl"


def build_bm25_embeddings() -> None:
    """Build BM25 indexes for both fixed and semantic chunk corpora."""
    print("\n" + "=" * 70)
    print("Building BM25 Embeddings")
    print("=" * 70)
    
    # Fixed chunking
    print("\n[1/2] Fixed chunking strategy")
    print("-" * 70)
    fixed_embedder = BM25Embedder(output_dir=BM25_DIR / "fixed")
    fixed_embedder.embed_chunks(CHUNKS_FIXED_JSONL)
    
    # Semantic chunking
    print("\n[2/2] Semantic chunking strategy")
    print("-" * 70)
    semantic_embedder = BM25Embedder(output_dir=BM25_DIR / "semantic")
    semantic_embedder.embed_chunks(CHUNKS_SEM_JSONL)


def build_openai_embeddings() -> None:
    """Build OpenAI + FAISS indexes for both fixed and semantic chunk corpora."""
    print("\n" + "=" * 70)
    print("Building OpenAI Embeddings")
    print("=" * 70)
    
    # Fixed chunking
    print("\n[1/2] Fixed chunking strategy")
    print("-" * 70)
    fixed_embedder = OpenAIEmbedder(
        output_dir=EMBEDDINGS_DIR / "openai" / "fixed_faiss",
        docs_batch_size=256,
        embed_chunk_size=32,
    )
    fixed_embedder.embed_chunks(CHUNKS_FIXED_JSONL)
    
    # Semantic chunking
    print("\n[2/2] Semantic chunking strategy")
    print("-" * 70)
    semantic_embedder = OpenAIEmbedder(
        output_dir=EMBEDDINGS_DIR / "openai" / "semantic_faiss",
        docs_batch_size=256,
        embed_chunk_size=32,
    )
    semantic_embedder.embed_chunks(CHUNKS_SEM_JSONL)


def main() -> None:
    """Build all embeddings: BM25 and OpenAI for both chunking strategies."""
    ensure_dirs()
    
    print("\n" + "=" * 70)
    print("Exercise 4 — Stage 2: Temporal Indexing")
    print("Building embeddings for fixed and semantic chunking strategies")
    print("=" * 70)
    
    # Build BM25 (sparse) embeddings
    build_bm25_embeddings()
    
    # Build OpenAI (dense) embeddings
    build_openai_embeddings()
    
    print("\n" + "=" * 70)
    print("✅ All embeddings completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
