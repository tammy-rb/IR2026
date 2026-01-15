"""
Chunk data model.

A Chunk is an immutable value object representing a contiguous character span
within a source document, enriched with provenance and optional temporal metadata.

Field semantics:
- chunk_uid: Unique identifier for the chunk in format "{corpus}:{doc_id}:{chunking_method}:{chunk_index}".
- doc_id: Stable document identifier (typically the filename without extension).
- source_path: Full path to the raw debate .txt file, used as a provenance pointer.
- corpus: Corpus label derived from the directory structure (e.g., "us" or "british").
- chunking_method: Chunking strategy used to produce the chunk ("fixed" or "semantic").
- chunk_index: Zero-based ordinal of the chunk within its source document.
- start_char, end_char: Character offsets into the original document text
  defining the exact slice boundaries (end offset is exclusive).
- text: The textual content of the chunk.
- num_words: Word count of the chunk, used to enforce maximum chunk-size constraints.
- doc_date_iso: Normalized document date in ISO 8601 format (YYYY-MM-DD).
- doc_timestamp: Normalized document date as a Unix timestamp (UTC), used for
  efficient temporal filtering and scoring.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class Chunk:
    """Immutable chunk record used across the pipeline."""

    # Provenance / identity
    chunk_uid: str  # "{corpus}:{doc_id}:{chunking_method}:{chunk_index}"
    doc_id: str
    source_path: str
    corpus: str
    chunking_method: str  # "fixed" / "semantic"
    chunk_index: int

    # Span in source document (end_char is exclusive)
    start_char: int
    end_char: int

    # Content
    text: str
    num_words: int

    # Temporal metadata (optional)
    doc_date_iso: Optional[str]
    doc_timestamp: Optional[int]

    def __post_init__(self) -> None:
        """Validate basic invariants."""
        if self.start_char < 0 or self.end_char < 0:
            raise ValueError("start_char/end_char must be >= 0")
        if self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")
        if self.num_words < 0:
            raise ValueError("num_words must be >= 0")

    # ------------------------------------------------------------------
    # Factories / parsing
    # ------------------------------------------------------------------

    @classmethod
    def create_chunk(
        cls,
        *,
        doc_id: str,
        source_path: str,
        corpus: str,
        chunking_method: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        text: str,
        num_words: int,
        doc_date_iso: Optional[str] = None,
        doc_timestamp: Optional[int] = None,
    ) -> "Chunk":
        """Project-wide factory: coerces types and normalizes optional fields."""
        corpus_str = str(corpus)
        doc_id_str = str(doc_id)
        chunking_method_str = str(chunking_method)
        chunk_index_int = int(chunk_index)
        chunk_uid = f"{corpus_str}:{doc_id_str}:{chunking_method_str}:{chunk_index_int}"
        
        return cls(
            chunk_uid=chunk_uid,
            doc_id=doc_id_str,
            source_path=str(source_path),
            corpus=corpus_str,
            chunking_method=chunking_method_str,
            chunk_index=chunk_index_int,
            start_char=int(start_char),
            end_char=int(end_char),
            text=str(text),
            num_words=int(num_words),
            doc_date_iso=None if doc_date_iso in (None, "") else str(doc_date_iso),
            doc_timestamp=None if doc_timestamp in (None, "") else int(doc_timestamp),
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Chunk":
        """Parse a Chunk from a dict/JSON payload (strict)."""

        def req(key: str) -> Any:
            if key not in d:
                raise ValueError(f"Missing required field: {key}")
            return d[key]

        def opt_str(key: str) -> Optional[str]:
            v = d.get(key, None)
            return None if v in (None, "") else str(v)

        def opt_int(key: str) -> Optional[int]:
            v = d.get(key, None)
            return None if v in (None, "") else int(v)

        try:
            # Check if chunk_uid exists in dict (backward compatibility)
            # If not present, create_chunk will generate it
            return cls.create_chunk(
                doc_id=req("doc_id"),
                source_path=req("source_path"),
                corpus=req("corpus"),
                chunking_method=req("chunking_method"),
                chunk_index=req("chunk_index"),
                start_char=req("start_char"),
                end_char=req("end_char"),
                text=req("text"),
                num_words=req("num_words"),
                doc_date_iso=opt_str("doc_date_iso"),
                doc_timestamp=opt_int("doc_timestamp"),
            )
        except Exception as e:
            raise ValueError(f"Invalid Chunk payload: {e}") from e

    # ------------------------------------------------------------------
    # Serialization / JSONL helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def read_jsonl(cls, path: Path) -> List["Chunk"]:
        """Read JSONL where each line is one chunk object."""
        if not path.is_file():
            raise FileNotFoundError(str(path))

        chunks: List[Chunk] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(cls.from_dict(json.loads(line)))
                except Exception as e:
                    raise ValueError(
                        f"Invalid chunk in {path.name} at line {line_no}: {e}"
                    ) from e
        return chunks

    @staticmethod
    def write_jsonl(chunks: Iterable["Chunk"], path: Path) -> None:
        """Write chunks to JSONL (one JSON object per line)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
