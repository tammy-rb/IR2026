"""
embedders/openai_embedder.py

OpenAI + FAISS dense embedding strategy.

Builds FAISS vector indexes over chunk corpora while storing temporal
metadata alongside each embedded chunk for later time-aware retrieval.

Metadata storage:
- FAISS stores vectors
- LangChain's FAISS wrapper stores texts + metadata in a docstore
- We attach doc_timestamp (Unix seconds UTC) and doc_date_iso to each Document.metadata

Outputs per strategy:
- FAISS index directory containing:
  - index.faiss
  - index.pkl (docstore with metadata)

Requirements:
- OPENAI_API_KEY must be available (e.g., via .env loaded by python-dotenv)
"""

from __future__ import annotations

import time
from itertools import islice
from pathlib import Path
from typing import Callable, Iterator, List, Optional, TypeVar

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from embedders.base import BaseEmbedder
from models.chunk import Chunk

# Default OpenAI embedding model
OPENAI_EMBED_MODEL = "text-embedding-3-large"


T = TypeVar("T")


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI + FAISS dense embedding implementation.
    
    Converts chunks into dense vectors using OpenAI's embeddings API
    and stores them in a FAISS index with metadata.
    """
    
    def __init__(
        self,
        output_dir: Path,
        model: str = OPENAI_EMBED_MODEL,
        docs_batch_size: int = 256,
        embed_chunk_size: int = 32,
    ):
        """
        Initialize OpenAI embedder.
        
        Args:
            output_dir: Directory to save FAISS index.
            model: OpenAI embedding model name.
            docs_batch_size: How many Documents to add per FAISS add_documents call.
            embed_chunk_size: Internal batching for OpenAIEmbeddings.
        """
        super().__init__(output_dir)
        self.model = model
        self.docs_batch_size = docs_batch_size
        self.embed_chunk_size = embed_chunk_size
    
    @property
    def name(self) -> str:
        return "openai"
    
    def chunk_to_doc(self, c: Chunk) -> Document:
        """
        Convert a Chunk into a LangChain Document.
        
        page_content:
            The chunk text that will be embedded.
        metadata:
            Retrieval pointers back to the original document + temporal fields.
        
        Args:
            c: Chunk object.
        
        Returns:
            LangChain Document.
        """
        meta = self.extract_metadata(c)
        return Document(page_content=c.text, metadata=meta)
    
    @staticmethod
    def batched(it: Iterator[Chunk], n: int) -> Iterator[List[Chunk]]:
        """
        Yield items from an iterable in lists of size n.
        
        Args:
            it: Iterable over chunk dicts.
            n: Batch size.
        
        Yields:
            Lists of chunk dicts of length <= n.
        """
        batch: List[Chunk] = []
        for x in it:
            batch.append(x)
            if len(batch) >= n:
                yield batch
                batch = []
        if batch:
            yield batch
    
    def embed_chunks(self, chunks_jsonl: Path) -> None:
        """
        Build a FAISS index from a JSONL chunks file in a memory-safe streaming way.
        
        Workflow:
        - Stream JSONL from disk (no full file load)
        - Convert each chunk into a Document with metadata including timestamp
        - Embed documents using OpenAIEmbeddings in smaller batches
        - Incrementally add vectors to FAISS
        - Periodically save checkpoints
        
        Args:
            chunks_jsonl: Path to chunks JSONL file.
        """
        embeddings = OpenAIEmbeddings(
            model=self.model,
            chunk_size=self.embed_chunk_size,
        )

        vectorstore: FAISS | None = self._try_load_existing_index(embeddings)
        already_indexed = self._get_index_size(vectorstore)
        total = already_indexed
        
        print(
            f"Building FAISS index from {chunks_jsonl.name} | "
            f"docs_batch={self.docs_batch_size}, embed_chunk={self.embed_chunk_size}"
        )

        if vectorstore is not None and already_indexed > 0:
            print(f"  ↩️  resuming from existing FAISS index: {already_indexed} docs")

        chunks_iter = self.read_chunks_stream(chunks_jsonl)
        if already_indexed > 0:
            chunks_iter = islice(chunks_iter, already_indexed, None)

        start_batch = (already_indexed // self.docs_batch_size) + 1
        for i, batch in enumerate(self.batched(chunks_iter, self.docs_batch_size), start_batch):
            docs = [self.chunk_to_doc(c) for c in batch]

            if vectorstore is None:
                vectorstore = self._with_retry(lambda: FAISS.from_documents(docs, embeddings))
            else:
                self._with_retry(lambda: vectorstore.add_documents(docs))

            total += len(docs)
            print(f"  ✓ batch {i} | total documents indexed: {total}")

            # Periodic checkpoints
            if i % 20 == 0:
                vectorstore.save_local(str(self.output_dir))
                print(f"  💾 checkpoint saved to {self.output_dir}")

        assert vectorstore is not None
        vectorstore.save_local(str(self.output_dir))
        print(f"✅ FAISS index saved to: {self.output_dir}")

    def _with_retry(self, fn: Callable[[], T], *, max_retries: int = 6, base_sleep_s: float = 2.0) -> T:
        """Retry wrapper for transient OpenAI/network failures."""

        last_err: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 - intentionally broad for network/provider issues
                last_err = e
                if attempt >= max_retries:
                    raise

                sleep_s = base_sleep_s * (2 ** attempt)
                sleep_s = min(sleep_s, 60.0)
                print(
                    f"  ⚠️  embedding call failed ({e.__class__.__name__}: {e}). "
                    f"Retrying in {sleep_s:.1f}s..."
                )
                time.sleep(sleep_s)

        assert last_err is not None
        raise last_err

    def _try_load_existing_index(self, embeddings: OpenAIEmbeddings) -> FAISS | None:
        """Load an existing FAISS index from output_dir if present."""

        faiss_path = self.output_dir / "index.faiss"
        pkl_path = self.output_dir / "index.pkl"
        if not (faiss_path.is_file() and pkl_path.is_file()):
            return None

        try:
            return FAISS.load_local(
                str(self.output_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except TypeError:
            return FAISS.load_local(str(self.output_dir), embeddings)

    @staticmethod
    def _get_index_size(vectorstore: FAISS | None) -> int:
        if vectorstore is None:
            return 0

        index = getattr(vectorstore, "index", None)
        if index is None:
            return 0

        ntotal = getattr(index, "ntotal", None)
        if isinstance(ntotal, int):
            return ntotal
        try:
            return int(ntotal)
        except Exception:
            return 0
