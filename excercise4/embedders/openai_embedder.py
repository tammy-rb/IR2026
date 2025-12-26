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

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from embedders.base import BaseEmbedder

# Default OpenAI embedding model
OPENAI_EMBED_MODEL = "text-embedding-3-large"


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
    
    def chunk_to_doc(self, c: Dict[str, Any]) -> Document:
        """
        Convert a chunk JSON dict into a LangChain Document.
        
        page_content:
            The chunk text that will be embedded.
        metadata:
            Retrieval pointers back to the original document + temporal fields.
        
        Args:
            c: Chunk dict from JSONL.
        
        Returns:
            LangChain Document.
        """
        meta = self.extract_metadata(c)
        return Document(page_content=c["text"], metadata=meta)
    
    @staticmethod
    def batched(it: Iterable[Dict[str, Any]], n: int) -> Iterator[List[Dict[str, Any]]]:
        """
        Yield items from an iterable in lists of size n.
        
        Args:
            it: Iterable over chunk dicts.
            n: Batch size.
        
        Yields:
            Lists of chunk dicts of length <= n.
        """
        batch: List[Dict[str, Any]] = []
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
        
        vectorstore: FAISS | None = None
        total = 0
        
        print(
            f"Building FAISS index from {chunks_jsonl.name} | "
            f"docs_batch={self.docs_batch_size}, embed_chunk={self.embed_chunk_size}"
        )
        
        for i, batch in enumerate(self.batched(self.read_jsonl_stream(chunks_jsonl), self.docs_batch_size), 1):
            docs = [self.chunk_to_doc(c) for c in batch]
            
            if vectorstore is None:
                vectorstore = FAISS.from_documents(docs, embeddings)
            else:
                vectorstore.add_documents(docs)
            
            total += len(docs)
            print(f"  ✓ batch {i} | total documents indexed: {total}")
            
            # Periodic checkpoints
            if i % 20 == 0:
                vectorstore.save_local(str(self.output_dir))
                print(f"  💾 checkpoint saved to {self.output_dir}")
        
        assert vectorstore is not None
        vectorstore.save_local(str(self.output_dir))
        print(f"✅ FAISS index saved to: {self.output_dir}")
