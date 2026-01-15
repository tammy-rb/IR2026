"""
embedders/base.py

Base class for embedding strategies.

Provides common functionality for:
- Loading chunks from JSONL
- Saving metadata with temporal fields
- Abstract interface for embedding implementations
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List

from models.chunk import Chunk


class BaseEmbedder(ABC):
    """
    Abstract base class for chunk embedding strategies.
    
    Subclasses implement specific embedding approaches (BM25, OpenAI, etc.)
    while inheriting common I/O and metadata handling.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize the embedder.
        
        Args:
            output_dir: Directory where embedding artifacts will be saved.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def read_chunks(path: Path) -> List[Chunk]:
        """Load and validate chunks from JSONL into `Chunk` objects."""
        return Chunk.read_jsonl(path)
    
    @staticmethod
    def read_chunks_stream(path: Path) -> Iterator[Chunk]:
        """Stream-read chunks from JSONL as validated `Chunk` objects."""
        if not path.is_file():
            raise FileNotFoundError(str(path))

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Chunk.from_dict(json.loads(line))
                except Exception as e:
                    raise ValueError(f"Invalid chunk in {path.name} at line {line_no}: {e}") from e
    
    @staticmethod
    def extract_metadata(chunk: Chunk) -> Dict[str, Any]:
        """Extract metadata from a `Chunk`, preserving temporal fields."""
        return {
            "chunk_uid": chunk.chunk_uid,   # NEW
            "doc_id": chunk.doc_id,
            "source_path": chunk.source_path,
            "corpus": chunk.corpus,
            "chunking_method": chunk.chunking_method,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "num_words": chunk.num_words,
            "doc_date_iso": chunk.doc_date_iso,
            "doc_timestamp": chunk.doc_timestamp,
        }

    
    def save_metadata(self, metadata: List[Dict[str, Any]], filename: str = "meta.json") -> None:
        """
        Save metadata to JSON file.
        
        Args:
            metadata: List of metadata dicts.
            filename: Output filename (default: meta.json).
        """
        with (self.output_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)
    
    @abstractmethod
    def embed_chunks(self, chunks_jsonl: Path) -> None:
        """
        Build and save embeddings for chunks from a JSONL file.
        
        Args:
            chunks_jsonl: Path to chunks JSONL file.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this embedding strategy."""
        pass
