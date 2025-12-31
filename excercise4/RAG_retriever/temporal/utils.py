from __future__ import annotations

from typing import Any, List, Optional
import math

def ts_from_chunk(c: Any) -> Optional[int]:
    """
    Safely extract doc_timestamp from a Chunk-like object.

    Returns:
        Unix timestamp (int, seconds) if available and valid,
        otherwise None.
    """
    ts = getattr(c, "doc_timestamp", None)
    if ts is None:
        return None
    try:
        return int(ts)
    except Exception:
        return None


def delta_days(
    ref_ts: int,
    doc_ts: Optional[int],
    *,
    clamp_future: bool = True,
) -> float:
    """
    Compute time difference in DAYS between ref_ts and doc_ts.

    Args:
        ref_ts: Reference timestamp (usually query time), in seconds.
        doc_ts: Document timestamp (seconds) or None.
        clamp_future: If True, future documents are not boosted
                      (negative deltas are clamped to 0).

    Returns:
        Δt in days.
        If doc_ts is None -> returns a large value (max temporal penalty).
    """
    if doc_ts is None:
        # Treat missing timestamps as very old documents
        return 10_000.0

    dt_sec = float(ref_ts - int(doc_ts)) # delta in seconds
    if clamp_future and dt_sec < 0:
        dt_sec = 0.0

    return dt_sec / 86400.0  # seconds per day 

import math
from typing import List


def log_normalize(vals: List[float], eps: float = 1e-12) -> List[float]:
    """
    Normalize non-negative scores to [0,1] using logarithmic scaling.

    Reduces the dynamic range of unbounded scores (e.g., BM25) while
    preserving their relative ordering, making them suitable for
    combination with temporal decay signals.

    Args:
        vals: List of non-negative scores.
        eps: Small constant to handle degenerate cases.

    Returns:
        Log-normalized scores in [0,1].
    """
    if not vals:
        return []

    max_v = max(vals)
    if max_v <= eps:
        return [0.0 for _ in vals]

    return [math.log(1 + v) / math.log(1 + max_v) for v in vals]

def minmax_normalize(vals: List[float], eps: float = 1e-12) -> List[float]:
    """
    Min-max normalize values into [0,1].

    Used mainly for BM25 scores (unbounded) before mixing with time_score.

    Args:
        vals: List of numeric scores.
        eps: Small constant to avoid division by zero.

    Returns:
        List of normalized values in [0,1].
        If all values are (almost) equal, returns zeros.
    """
    if not vals:
        return []

    vmin = min(vals)
    vmax = max(vals)

    if abs(vmax - vmin) < eps:
        # No meaningful relative differences
        return [0.0 for _ in vals]

    return [(v - vmin) / (vmax - vmin + eps) for v in vals]
