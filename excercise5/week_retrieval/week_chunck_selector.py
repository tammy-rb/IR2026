"""
Week-based candidate selection utilities.

This module converts week-based requests (e.g., 2-week windows starting on
Monday 00:00 UTC) into metadata filters and uses CandidateSelector to select
candidate chunks.

It is retriever-agnostic: it only selects candidates and does not perform
retrieval or scoring.

Typical usage:
    selector = CandidateSelector.from_jsonl(chunks_jsonl)
    week_selector = WeekSelector(selector)

    flt, selection = week_selector.select(WeekRequest(week_start_ts))
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Set, Tuple, Union

from RAG_retriever.prefilter.chuncks_selector import (
    CandidateSelector,
    ChunkFilter,
    Selection,
)

SECONDS_WEEK = 7 * 24 * 60 * 60


def week_start_utc_from_ts(ts: int) -> int:
    """Return Monday 00:00 UTC timestamp for the week containing ts."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    monday = dt - timedelta(days=dt.weekday())
    monday0 = datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)
    return int(monday0.timestamp())


def week_range(week_start_ts: int, *, window_weeks: int = 2) -> Tuple[int, int]:
    """[start, end) in UTC seconds for a week-based window."""
    start = int(week_start_ts)
    end = start + int(window_weeks) * SECONDS_WEEK
    return start, end


@dataclass(frozen=True)
class WeekRequest:
    """
    Request for a week-based candidate set.

    week_start_ts must be Monday 00:00 UTC.
    """
    week_start_ts: int
    window_weeks: int = 2

    corpora: Optional[Set[str]] = None
    chunking_methods: Optional[Set[str]] = None
    doc_ids: Optional[Set[str]] = None

    require_timestamp: bool = True


class WeekSelector:
    """
    Week-based candidate selection built on top of CandidateSelector.

    Produces:
      - ChunkFilter (backend-agnostic)
      - Selection (row_ids + chunk_uids)
    """

    def __init__(self, selector: CandidateSelector) -> None:
        self._selector = selector

    def build_filter(self, req: WeekRequest) -> ChunkFilter:
        start_ts, end_ts = week_range(req.week_start_ts, window_weeks=req.window_weeks)
        return ChunkFilter(
            time_min_ts=start_ts,
            time_max_ts=end_ts,
            require_timestamp=req.require_timestamp,
            corpora=req.corpora,
            chunking_methods=req.chunking_methods,
            doc_ids=req.doc_ids,
        )

    def select(self, req: WeekRequest) -> Tuple[ChunkFilter, Selection]:
        flt = self.build_filter(req)
        selection = self._selector.select(flt)
        return flt, selection

    def select_per_corpus(self, req: WeekRequest) -> Dict[str, Tuple[ChunkFilter, Selection]]:
        """
        Convenience: return one selection per corpus.
        If req.corpora is None, it will raise (caller must specify corpora list).
        """
        if not req.corpora:
            raise ValueError("select_per_corpus requires req.corpora to be set")

        out: Dict[str, Tuple[ChunkFilter, Selection]] = {}
        for c in sorted(req.corpora):
            flt = self.build_filter(
                WeekRequest(
                    week_start_ts=req.week_start_ts,
                    window_weeks=req.window_weeks,
                    corpora={c},
                    chunking_methods=req.chunking_methods,
                    doc_ids=req.doc_ids,
                    require_timestamp=req.require_timestamp,
                )
            )
            out[c] = (flt, self._selector.select(flt))
        return out
