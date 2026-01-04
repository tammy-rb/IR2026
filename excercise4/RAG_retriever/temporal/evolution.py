"""
Evolution retrieval logic for temporal RAG.

This module handles the double-retrieval pattern used for evolution queries:
- Filtering candidates into early/late temporal windows
- Sorting by relevance within each window
- Selecting top-K from each period
"""

from __future__ import annotations

from typing import Any, List, Tuple


def retrieve_evolution_windows(
    candidates: List[Tuple[Any, float]],
    early_range: Tuple[int, int],
    late_range: Tuple[int, int],
    k: int,
) -> Tuple[List[Tuple[Any, float]], List[Tuple[Any, float]]]:
    """
    Filter and sort candidates into early and late temporal windows.
    
    Args:
        candidates: List of (chunk, score) tuples from retrieval
        early_range: (start_ts, end_ts) for early window in Unix timestamps
        late_range: (start_ts, end_ts) for late window in Unix timestamps
        k: Number of chunks to retrieve from each window
        
    Returns:
        Tuple of (early_items, late_items) where each is a list of top-K
        (chunk, score) tuples sorted by relevance
        
    Notes:
        - Chunks without doc_timestamp are excluded
        - Within each window, items are sorted by score (descending)
        - If fewer than K items exist in a window, returns what's available
    """
    early_start_ts, early_end_ts = early_range
    late_start_ts, late_end_ts = late_range

    def _in_early(c: Any) -> bool:
        """Check if chunk falls within early time window."""
        ts = getattr(c, "doc_timestamp", None)
        return ts is not None and early_start_ts <= int(ts) <= early_end_ts

    def _in_late(c: Any) -> bool:
        """Check if chunk falls within late time window."""
        ts = getattr(c, "doc_timestamp", None)
        return ts is not None and late_start_ts <= int(ts) <= late_end_ts

    # Filter candidates into respective windows
    early_cands = [(c, s) for (c, s) in candidates if _in_early(c)]
    late_cands = [(c, s) for (c, s) in candidates if _in_late(c)]

    # Sort by semantic score (relevance) within each period, then take top-k
    early_cands.sort(key=lambda x: x[1], reverse=True)
    late_cands.sort(key=lambda x: x[1], reverse=True)

    early_items = early_cands[:k]
    late_items = late_cands[:k]

    return early_items, late_items


def has_sufficient_results(
    early_items: List[Tuple[Any, float]],
    late_items: List[Tuple[Any, float]],
    k: int,
) -> bool:
    """
    Check if both windows have sufficient results.
    
    Args:
        early_items: Items from early window
        late_items: Items from late window
        k: Target number of items per window
        
    Returns:
        True if both windows have at least K items
    """
    return len(early_items) >= k and len(late_items) >= k
