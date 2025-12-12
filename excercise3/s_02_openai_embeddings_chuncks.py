# s_02_openai_embeddings_chunks.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain + OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Vector store
from langchain_community.vectorstores import FAISS


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

CHUNKS_DIR = os.path.join(BASE_DIR, "outputs", "chunks")
EMB_OUT_DIR = os.path.join(BASE_DIR, "outputs", "embeddings_openai")

CHUNKS_FIXED_JSONL = os.path.join(CHUNKS_DIR, "chunks_fixed.jsonl")
CHUNKS_SEM_JSONL = os.path.join(CHUNKS_DIR, "chunks_semantic.jsonl")

# OpenAI embedding model:
# - "text-embedding-3-small" is cheaper/faster
# - "text-embedding-3-large" is stronger but more expensive
OPENAI_EMBED_MODEL = "text-embedding-3-large"

# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def chunks_to_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert your chunk JSON objects into LangChain Documents.
    We keep retrieval pointers in metadata (source file + offsets).
    """
    docs: List[Document] = []

    for c in chunks:
        text = c["text"]
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
        docs.append(Document(page_content=text, metadata=meta))

    return docs


def build_faiss_index(chunks_jsonl: str, out_dir: str) -> None:
    ensure_dir(out_dir)

    chunks = read_jsonl(chunks_jsonl)
    if not chunks:
        raise ValueError(f"No chunks found in: {chunks_jsonl}")

    docs = chunks_to_documents(chunks)
    print(f"Loaded {len(docs)} chunks from: {os.path.basename(chunks_jsonl)}")

    # Requires OPENAI_API_KEY in env (or configured via LangChain)
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)

    # Build FAISS from documents (embeds each chunk and stores vector+metadata)
    # If you want progress control, keep prints around larger steps.
    print(f"Building FAISS index using OpenAI embeddings: {OPENAI_EMBED_MODEL}")
    print(f"Embedding {len(docs)} chunks... (this may take a while)")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Persist to disk (creates index + docstore metadata)
    vectorstore.save_local(out_dir)
    print(f"✅ Saved FAISS index to: {out_dir}")


def smoke_test_query(index_dir: str, query: str, top_k: int = 5) -> None:
    """
    Quick sanity test: load FAISS index and retrieve top_k chunks for a query.
    """
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

    results = vs.similarity_search_with_score(query, k=top_k)

    print("\n--- Dense Retrieval Smoke Test (FAISS) ---")
    print(f"Query: {query}")
    for rank, (doc, score) in enumerate(results, 1):
        m = doc.metadata
        print(
            f"{rank}. score={score:.4f} | file={os.path.basename(m['source_path'])} "
            f"| chunk={m['chunk_index']} | offsets=({m['start_char']},{m['end_char']})"
        )


def main() -> None:
    ensure_dir(EMB_OUT_DIR)

    fixed_out = os.path.join(EMB_OUT_DIR, "fixed_faiss")
    sem_out = os.path.join(EMB_OUT_DIR, "semantic_faiss")

    #build_faiss_index(CHUNKS_FIXED_JSONL, fixed_out)
    build_faiss_index(CHUNKS_SEM_JSONL, sem_out)

    smoke_test_query(sem_out, query="foreign policy and security cooperation", top_k=5)


if __name__ == "__main__":
    main()
