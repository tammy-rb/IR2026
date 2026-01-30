"""
s_02_embedding_news.py

Stage 1 (News Indexing): Build BM25 and OpenAI+Qdrant embeddings for BBC and NBC news chunks.

This script expects the per-document chunk JSONL files produced by
s_01_chuncking_news.py.

Outputs:
- BM25 indexes under outputs/embeddings/bm25/news/(bbc|nbc)
- Qdrant payload exports under outputs/embeddings/openai_qdrant/news/(bbc|nbc)
- Upserts embeddings into dedicated Qdrant collections for BBC and NBC news.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Allow running as a script from this folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import (
    BBC_NEWS_CHUNKS_JSONL,
    NBC_NEWS_CHUNKS_JSONL,
    BM25_DIR,
    OPENAI_QDRANT_DIR,
    ensure_dirs,
)
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    OPENAI_EMBED_MODEL,
    FIXED_DOCS_BATCH_SIZE,
    FIXED_EMBED_BATCH_SIZE,
    VECTOR_DISTANCE,
)
from embedders.bm25_embedder import BM25Embedder
from embedders.openai_embedder import OpenAIEmbedder

# ---------------------------------------------------------------------------
# Target configuration
# ---------------------------------------------------------------------------
NEWS_BM25_DIR = BM25_DIR / "news"
NEWS_QDRANT_DIR = OPENAI_QDRANT_DIR / "news"

TARGETS = {
    "bbc": {
        "label": "BBC News",
        "jsonl": BBC_NEWS_CHUNKS_JSONL,
        "bm25_dir": NEWS_BM25_DIR / "bbc",
        "qdrant_dir": NEWS_QDRANT_DIR / "bbc",
        "collection": "bbc_news_chunks",
    },
    "nbc": {
        "label": "NBC News",
        "jsonl": NBC_NEWS_CHUNKS_JSONL,
        "bm25_dir": NEWS_BM25_DIR / "nbc",
        "qdrant_dir": NEWS_QDRANT_DIR / "nbc",
        "collection": "nbc_news_chunks",
    },
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_bm25_embeddings(target_names: list[str]) -> None:
    print("\n" + "=" * 70)
    print("Building BM25 embeddings for news chunks")
    print("=" * 70)

    for name in target_names:
        target = TARGETS[name]
        out_dir = target["bm25_dir"]
        jsonl_path = target["jsonl"]

        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[BM25] {target['label']}")
        print("-" * 70)
        embedder = BM25Embedder(output_dir=out_dir)
        embedder.embed_chunks(jsonl_path)


def build_openai_embeddings(target_names: list[str]) -> None:
    print("\n" + "=" * 70)
    print("Building OpenAI embeddings for news chunks")
    print("=" * 70)

    for name in target_names:
        target = TARGETS[name]
        out_dir = target["qdrant_dir"]
        jsonl_path = target["jsonl"]

        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[OpenAI] {target['label']}")
        print("-" * 70)
        embedder = OpenAIEmbedder(
            output_dir=out_dir,
            docs_batch_size=FIXED_DOCS_BATCH_SIZE,
            embed_batch_size=FIXED_EMBED_BATCH_SIZE,
            collection_name=target["collection"],
            model=OPENAI_EMBED_MODEL,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            distance=VECTOR_DISTANCE,
        )
        embedder.embed_chunks(jsonl_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 1 News Indexing: Build embeddings for BBC/NBC chunks")
    parser.add_argument(
        "--only",
        choices=["all", "bm25", "openai"],
        default="all",
        help="Which embedding family to run (default: all)",
    )
    parser.add_argument(
        "--sources",
        choices=["bbc", "nbc", "both"],
        default="both",
        help="Which news sources to process (default: both)",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    NEWS_BM25_DIR.mkdir(parents=True, exist_ok=True)
    NEWS_QDRANT_DIR.mkdir(parents=True, exist_ok=True)

    if args.sources == "both":
        targets = ["bbc", "nbc"]
    else:
        targets = [args.sources]

    if args.only in ("all", "bm25"):
        build_bm25_embeddings(targets)

    if args.only in ("all", "openai"):
        build_openai_embeddings(targets)

    print("\n" + "=" * 70)
    print("✅ News embeddings completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
