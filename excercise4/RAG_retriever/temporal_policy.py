from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RetrievalPlan:
    strategy: str  # "hard" | "soft" | "none"
    # For hard filtering (inclusive bounds). None = open bound.
    start_ts: Optional[int]
    end_ts: Optional[int]

    # For soft decay
    ref_ts: int
    alpha: float
    lam: float  # lambda in 1/days (because Δt is in days)

    # Candidate oversampling for filtering/reranking
    oversample: int


def _iso_date_to_ts(iso_date: str) -> int:
    """
    Convert YYYY-MM-DD into a UTC midnight unix timestamp (seconds).
    """
    d = datetime.fromisoformat(iso_date).date()
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def _combine_ranges_intersection(ranges: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int]]:
    """
    Combine possibly multiple time ranges into a single constraint using INTERSECTION.

    - start_ts = max(all non-null starts)
    - end_ts   = min(all non-null ends)
    """
    starts: List[int] = []
    ends: List[int] = []

    for r in ranges:
        s = r.get("start")
        e = r.get("end")
        if isinstance(s, str) and s:
            starts.append(_iso_date_to_ts(s))
        if isinstance(e, str) and e:
            ends.append(_iso_date_to_ts(e))

    start_ts = max(starts) if starts else None
    end_ts = min(ends) if ends else None
    return start_ts, end_ts


def build_retrieval_plan(time_info: Dict[str, Any], *, k: int) -> RetrievalPlan:
    """
    Map Duckling analysis output to a retrieval plan.

    Strategy rules:
    - mode == "explicit" and has ranges -> HARD (filter strictly)
    - mode in {"current","recent","none"} -> SOFT (recency weighting)

    Parameter heuristics (good defaults; tune in experiments):
    - current: stronger recency preference
    - recent: medium
    - none: mild (tie-break)
    """
    mode = str(time_info.get("mode", "none"))
    ranges = time_info.get("ranges") or []
    now_iso = str(time_info.get("now_iso"))
    ref_ts = _iso_date_to_ts(now_iso)

    # Oversampling defaults
    soft_oversample = max(50, int(k) * 10)
    hard_oversample = max(200, int(k) * 20)

    if mode == "explicit" and ranges:
        start_ts, end_ts = _combine_ranges_intersection(ranges)
        return RetrievalPlan(
            strategy="hard",
            start_ts=start_ts,
            end_ts=end_ts,
            ref_ts=ref_ts,
            alpha=0.0,
            lam=0.0,
            oversample=hard_oversample,
        )

    # Soft decay defaults by mode
    if mode == "current":
        # prioritize newer documents strongly
        alpha = 0.35
        lam = 1.0 / 60.0   # ~60-day scale
        oversample = soft_oversample
    elif mode == "recent":
        alpha = 0.25
        lam = 1.0 / 180.0  # ~6-month scale
        oversample = soft_oversample
    else:  # none
        alpha = 0.15
        lam = 1.0 / 365.0  # ~1-year scale (mild)
        oversample = soft_oversample

    return RetrievalPlan(
        strategy="soft",
        start_ts=None,
        end_ts=None,
        ref_ts=ref_ts,
        alpha=float(alpha),
        lam=float(lam),
        oversample=int(oversample),
    )
