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
    def read_jsonl(path: Path) -> List[Dict[str, Any]]:
        """
        Load a JSONL file into a list of dictionaries.
        
        Args:
            path: Path to a JSONL file.
        
        Returns:
            List of parsed JSON objects (dicts).
        """
        items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    
    @staticmethod
    def read_jsonl_stream(path: Path) -> Iterator[Dict[str, Any]]:
        """
        Stream-read a JSONL file (one JSON object per line).
        
        Args:
            path: Path to JSONL file.
        
        Yields:
            Chunk dicts one by one (memory efficient).
        """
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    
    @staticmethod
    def extract_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from a chunk dict, preserving temporal fields.
        
        Args:
            chunk: Chunk dictionary from JSONL.
        
        Returns:
            Metadata dict with all relevant fields.
        """
        return {
            "doc_id": chunk.get("doc_id"),
            "source_path": chunk.get("source_path"),
            "corpus": chunk.get("corpus"),
            "chunking_method": chunk.get("chunking_method"),
            "chunk_index": chunk.get("chunk_index"),
            "start_char": chunk.get("start_char"),
            "end_char": chunk.get("end_char"),
            "num_words": chunk.get("num_words"),
            "doc_date_iso": chunk.get("doc_date_iso"),
            "doc_timestamp": chunk.get("doc_timestamp"),
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
