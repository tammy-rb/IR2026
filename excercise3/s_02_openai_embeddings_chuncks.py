from __future__ import annotations
import os, json
from typing import Any, Dict, Iterable
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# -------------------------
# Paths / Configuration
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

CHUNKS_DIR = os.path.join(BASE_DIR, "outputs", "chunks")

# Path to save FAISS indexes
EMB_OUT_DIR = os.path.join(BASE_DIR, "outputs", "embeddings_openai")

# Chunk files
CHUNKS_FIXED_JSONL = os.path.join(CHUNKS_DIR, "chunks_fixed.jsonl")
CHUNKS_SEM_JSONL   = os.path.join(CHUNKS_DIR, "chunks_semantic.jsonl")

OPENAI_EMBED_MODEL = "text-embedding-3-large"


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def read_jsonl_stream(path: str) -> Iterable[Dict[str, Any]]:
    """
    Stream-read a JSONL file (one JSON object per line).
    This avoids loading the entire file into memory.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunk_to_doc(c: Dict[str, Any]) -> Document:
    """
    Convert a chunk JSON object into a LangChain Document.
    - page_content: the actual chunk text to embed
    - metadata: retrieval pointers back to the original document
    """
    meta = {
        "doc_id": c.get("doc_id"),
        "source_path": c.get("source_path"),
        "corpus": c.get("corpus"),
        "chunking_method": c.get("chunking_method"),
        "chunk_index": c.get("chunk_index"),
        "start_char": c.get("start_char"),
        "end_char": c.get("end_char"),
        "num_words": c.get("num_words"),
    }
    return Document(page_content=c["text"], metadata=meta)


def batched(it: Iterable[Dict[str, Any]], n: int):
    """
    Yield items from an iterator in lists of size n.
    Example: n=256 yields batches of 256 chunks.
    """
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def build_faiss_index_streaming(
    chunks_jsonl: str,
    out_dir: str,
    docs_batch_size: int = 256,
    embed_chunk_size: int = 32
) -> None:
    """
    Build a FAISS index from a JSONL chunks file in a memory-safe way:
    - Stream JSONL from disk
    - Convert chunks to Documents in manageable batches
    - Embed texts in small embedding batches to avoid MemoryError
    - Incrementally add vectors to FAISS
    - Periodically save checkpoints
    """
    ensure_dir(out_dir)

    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBED_MODEL,
        chunk_size=embed_chunk_size,
    )

    vectorstore: FAISS | None = None
    total = 0

    print(
        f"Building FAISS index from {os.path.basename(chunks_jsonl)} | "
        f"docs_batch={docs_batch_size}, embed_chunk={embed_chunk_size}"
    )

    for i, batch in enumerate(batched(read_jsonl_stream(chunks_jsonl), docs_batch_size), 1):
        docs = [chunk_to_doc(c) for c in batch]

        if vectorstore is None:
            vectorstore = FAISS.from_documents(docs, embeddings)
        else:
            vectorstore.add_documents(docs)

        total += len(docs)
        print(f"  ✓ batch {i} | total documents indexed: {total}")

        # Save progress periodically
        if i % 20 == 0:
            vectorstore.save_local(out_dir)
            print(f"  💾 checkpoint saved to {out_dir}")

    assert vectorstore is not None
    vectorstore.save_local(out_dir)
    print(f"✅ DONE. FAISS index saved to: {out_dir}")


def main():
    """Build FAISS indexes for both fixed and semantic chunking."""
    ensure_dir(EMB_OUT_DIR)

    fixed_out = os.path.join(EMB_OUT_DIR, "fixed_faiss")
    semantic_out = os.path.join(EMB_OUT_DIR, "semantic_faiss")

    # Fixed-size chunking index
    # build_faiss_index_streaming(
    #     CHUNKS_FIXED_JSONL,
    #     fixed_out,
    #     docs_batch_size=256,
    #     embed_chunk_size=32
    # )

    # Semantic chunking index
    build_faiss_index_streaming(
        CHUNKS_SEM_JSONL,
        semantic_out,
        docs_batch_size=256,
        embed_chunk_size=32
    )


if __name__ == "__main__":
    main()
