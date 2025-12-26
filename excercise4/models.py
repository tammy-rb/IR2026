# chunking/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass
class Chunk:
    doc_id: str
    source_path: str
    corpus: str
    chunking_method: str     # "fixed" / "semantic"
    chunk_index: int

    start_char: int
    end_char: int

    text: str
    num_words: int

    doc_date_iso: Optional[str]
    doc_timestamp: Optional[int]
