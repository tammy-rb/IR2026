"""
s_01_chuncking_news.py

Stage 1 (News Indexing): Build one-chunk-per-document corpora for BBC and NBC news dumps.

Outputs:
- outputs/chunks/news_chuncks/bbc_chunks.jsonl
- outputs/chunks/news_chuncks/nbc_chunks.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List

# Allow running as a script from this folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import (
    BBC_NEWS_DIR,
    NBC_NEWS_DIR,
    BBC_NEWS_CHUNKS_JSONL,
    NBC_NEWS_CHUNKS_JSONL,
    NEWS_CHUNKS_DIR,
    ensure_dirs,
)
from utils.text_utils import iter_text_files
from models.chunk import Chunk
from chunckers.fixed_chuncker import FixedChunker


def write_jsonl(chunks: Iterable[Chunk], out_path: Path) -> None:
    Chunk.write_jsonl(chunks, out_path)


def summarize_time_coverage(chunks: List[Chunk], label: str) -> None:
    with_date = sum(1 for c in chunks if c.doc_timestamp is not None)
    print(f"[{label}] chunks with timestamp: {with_date}/{len(chunks)}")


def main() -> None:
    ensure_dirs()
    NEWS_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    chunker = _single_chunk_chunker()

    bbc_chunks = _chunk_corpus(BBC_NEWS_DIR, "BBC", BBC_NEWS_CHUNKS_JSONL, chunker)
    summarize_time_coverage(bbc_chunks, "BBC")
    print(f"Saved BBC chunks -> {BBC_NEWS_CHUNKS_JSONL}")

    nbc_chunks = _chunk_corpus(NBC_NEWS_DIR, "NBC", NBC_NEWS_CHUNKS_JSONL, chunker)
    summarize_time_coverage(nbc_chunks, "NBC")
    print(f"Saved NBC chunks -> {NBC_NEWS_CHUNKS_JSONL}")

    print(
        "Done. BBC chunks={bbc_count} | NBC chunks={nbc_count}".format(
            bbc_count=len(bbc_chunks),
            nbc_count=len(nbc_chunks),
        )
    )


def _chunk_corpus(
    corpus_dir: Path,
    label: str,
    out_path: Path,
    chunker: FixedChunker,
) -> List[Chunk]:
    files = sorted(iter_text_files(str(corpus_dir)))
    if not files:
        raise FileNotFoundError(f"No .txt files found in corpus directory: {corpus_dir}")

    total = len(files)
    chunks: List[Chunk] = []

    for idx, fp in enumerate(files, 1):
        print(f"[{label}] {idx}/{total}: {Path(fp).name}")
        doc_chunks = chunker.build_chunks_for_file(fp)
        if not doc_chunks:
            print(f"[{label}] WARNING: {Path(fp).name} produced no chunks")
            continue
        chunks.extend(doc_chunks)

    write_jsonl(chunks, out_path)
    return chunks


def _single_chunk_chunker() -> FixedChunker:
    return FixedChunker(max_words=float("inf"), overlap_sentences=0)


if __name__ == "__main__":
    main()
