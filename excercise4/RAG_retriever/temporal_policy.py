from __future__ import annotations

import math
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
    h: float  # half-life in days

    # Candidate oversampling for filtering/reranking
    oversample: int

    @property
    def lam(self) -> float:
        """Legacy accessor for exponential-decay lambda."""
        if not self.h or math.isinf(self.h):
            return 0.0
        return math.log(2.0) / float(self.h)


def _iso_date_to_ts(iso_date: str) -> int:
    """
    Convert YYYY-MM-DD into a UTC midnight unix timestamp (seconds).
    better for time calculations.
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
            h=float("inf"),
            oversample=hard_oversample,
        )

    # Soft decay parameters
    if mode == "current":
        alpha = 0.6
        h = 365
        oversample = soft_oversample
    elif mode == "recent":
        alpha = 0.65 
        h = 365  # 1 year: typical policy cycle
        oversample = soft_oversample
    else:  # none
        alpha = 0.7
        h = 730  # 2 years: parliamentary term context
        oversample = soft_oversample

    return RetrievalPlan(
        strategy="soft",
        start_ts=None,
        end_ts=None,
        ref_ts=ref_ts,
        alpha=float(alpha),
        h=float(h),
        oversample=int(oversample),
    )
