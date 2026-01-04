"""
Temporal utility functions for RAG retrieval.

This module contains helper functions for temporal/evolution analysis:
- Computing corpus time bounds from chunk metadata
- Converting between time units (months, seconds, timestamps)
- Formatting evolution context for LLM prompts
- Timestamp/date conversions
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple


def compute_corpus_bounds(chunks_jsonl_path: str) -> Tuple[int, int]:
    """
    Scan chunks JSONL and return (min_doc_timestamp, max_doc_timestamp).
    
    Args:
        chunks_jsonl_path: Path to chunks JSONL file
        
    Returns:
        Tuple of (min_timestamp, max_timestamp) as Unix timestamps
        
    Raises:
        RuntimeError: If no valid timestamps found in the chunks file
        
    Notes:
        Requires chunks to contain 'doc_timestamp' field (Stage 2 output).
    """
    min_ts: Optional[int] = None
    max_ts: Optional[int] = None

    with open(chunks_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ts = obj.get("doc_timestamp")
            if ts is None:
                continue
            ts = int(ts)
            if min_ts is None or ts < min_ts:
                min_ts = ts
            if max_ts is None or ts > max_ts:
                max_ts = ts

    if min_ts is None or max_ts is None:
        raise RuntimeError(
            f"Cannot compute corpus bounds: missing doc_timestamp in {chunks_jsonl_path!r}"
        )
    return min_ts, max_ts


def months_to_seconds(months: int) -> int:
    """
    Approximate conversion: 1 month ~ 30.44 days.
    
    Args:
        months: Number of months
        
    Returns:
        Equivalent seconds (approximate)
        
    Notes:
        Good enough for windowing. Uses average month length.
    """
    return int(months * 30.44 * 24 * 60 * 60)


def ts_to_iso(ts: int) -> str:
    """
    Convert Unix timestamp to ISO date string.
    
    Args:
        ts: Unix timestamp (seconds since epoch)
        
    Returns:
        ISO date string (YYYY-MM-DD)
    """
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()


def format_evolution_context(
    early_items: List[Tuple[Any, float]],
    late_items: List[Tuple[Any, float]],
    *,
    early_range: Tuple[int, int],
    late_range: Tuple[int, int],
) -> str:
    """
    Build a single context string for evolution queries.
    
    Includes:
      - EARLY window (closest->farthest)
      - LATE window  (closest->farthest)

    "Closest" here means closest to the *center boundary* of the window:
      - EARLY: closer to early_end (newer inside early)
      - LATE:  closer to late_start (older inside late)
    
    Args:
        early_items: List of (chunk, score) tuples from early period
        late_items: List of (chunk, score) tuples from late period
        early_range: (start_ts, end_ts) for early window
        late_range: (start_ts, end_ts) for late window
        
    Returns:
        Formatted context string ready for LLM consumption
    """
    early_start, early_end = early_range
    late_start, late_end = late_range  # late_end is max_ts

    def _chunk_block(label: str, items: List[Tuple[Any, float]]) -> str:
        """Format a list of chunks with labels and metadata."""
        lines: List[str] = []
        for i, (c, s) in enumerate(items, start=1):
            ts = getattr(c, "doc_timestamp", None)
            iso = getattr(c, "doc_date_iso", None) or (ts_to_iso(ts) if ts else "unknown-date")
            lines.append(f"[{label}{i} | {iso} | score={float(s):.6f}]")
            lines.append(getattr(c, "text", "") or "")
            lines.append("")  # spacer
        return "\n".join(lines).strip()

    # Sort by distance to boundary
    # EARLY: closest to early_end == newer within early
    early_sorted = sorted(
        early_items,
        key=lambda x: abs(int(getattr(x[0], "doc_timestamp", 0)) - int(early_end))
    )

    # LATE: closest to late_start (older boundary)
    late_sorted = sorted(
        late_items,
        key=lambda x: abs(int(getattr(x[0], "doc_timestamp", 0)) - int(late_start))
    )

    header_early = f"EARLY WINDOW: {ts_to_iso(early_start)} .. {ts_to_iso(early_end)}"
    header_late = f"LATE WINDOW: {ts_to_iso(late_start)} .. {ts_to_iso(late_end)}"

    parts = [
        header_early,
        "",
        _chunk_block("E", early_sorted),
        "",
        header_late,
        "",
        _chunk_block("L", late_sorted),
    ]
    return "\n".join([p for p in parts if p is not None]).strip()
