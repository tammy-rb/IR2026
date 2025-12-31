from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .utils import ts_from_chunk


def hard_filter(
    items: List[Tuple[Any, float]],
    *,
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> List[Tuple[Any, float]]:
    """
    Keep only chunks whose doc_timestamp is within inclusive [start_ts, end_ts].
    None means open bound.
    """
    filtered: List[Tuple[Any, float]] = []
    for chunk, score in items:
        ts = ts_from_chunk(chunk)
        if ts is None:
            continue
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        filtered.append((chunk, score))
    return filtered