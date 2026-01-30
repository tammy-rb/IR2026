"""
utils/time_utils.py

Time extraction and normalization utilities for Exercise 4 (Temporal RAG).

Goal:
- Convert unstructured date cues (primarily from filenames/paths) into
  structured, normalized temporal metadata that can be attached to chunks.

Normalization:
- ISO 8601 date string: "YYYY-MM-DD"
- Unix timestamp (seconds since epoch), normalized to UTC

Expected input pattern:
- Filenames/paths containing a date in the form YYYY-MM-DD
  e.g., debates2023-07-10.txt  ->  ("2023-07-10", 1688947200)
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Optional, Tuple

# Matches dates in "YYYY-MM-DD" with basic month/day validation.
# Works even when attached to letters, e.g. "debates2025-07-15.txt".
# Avoids matching inside longer digit sequences (e.g. "...12025-07-15...").
_RE_YYYY_MM_DD = re.compile(
    r"(?<!\d)((?:19|20)\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(?!\d)"
)

# Matches filenames like "Fri_01_Aug_2025_07_01_00_GMT" (BBC/NBC news dumps).
_RE_GMT_TIMESTAMP = re.compile(
    r"\b"
    r"(?:(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[_\s\-]+)?"
    r"(\d{1,2})[_\s\-]+"
    r"([A-Za-z]{3})[_\s\-]+"
    r"(\d{4})"
    r"(?:[_\s\-:]+(\d{1,2}))?"
    r"(?:[_\s\-:]+(\d{1,2}))?"
    r"(?:[_\s\-:]+(\d{1,2}))?"
    r"[_\s\-]*"
    r"(GMT|UTC)"
    r"\b",
    re.IGNORECASE,
)

_MONTH_NAME_TO_NUM = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _parse_gmt_timestamp(s: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse timestamps that include an explicit GMT/UTC marker."""
    if not s:
        return None, None

    m = _RE_GMT_TIMESTAMP.search(s)
    if not m:
        return None, None

    (_dow, day_s, month_s, year_s, hour_s, minute_s, second_s, _tz) = m.groups()

    month_num = _MONTH_NAME_TO_NUM.get(month_s.lower())
    if month_num is None:
        return None, None

    try:
        dt = datetime(
            year=int(year_s),
            month=month_num,
            day=int(day_s),
            hour=int(hour_s or 0),
            minute=int(minute_s or 0),
            second=int(second_s or 0),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None, None

    return dt.date().isoformat(), int(dt.timestamp())


def timestamp_from_string(s: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract a date in YYYY-MM-DD format from an arbitrary string and return:
      (iso_date, unix_timestamp_seconds_utc)

    Args:
        s: Input string (filename/path/text).

    Returns:
        (iso_date, unix_ts) if a date is found, otherwise (None, None).

    Notes:
        - The returned timestamp is normalized to UTC midnight of the extracted date.
        - This is sufficient for the assignment since documents are dated at day granularity.
    """
    if not s:
        return None, None

    m = _RE_YYYY_MM_DD.search(s)
    if m:
        iso_date = m.group(0)  # "YYYY-MM-DD"
        dt = datetime.strptime(iso_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return iso_date, int(dt.timestamp())

    upper = s.upper()
    if "GMT" in upper or "UTC" in upper:
        gmt_iso, gmt_ts = _parse_gmt_timestamp(s)
        if gmt_iso is not None:
            return gmt_iso, gmt_ts

    return None, None


def timestamp_from_path(path: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Convenience wrapper to extract a date from a filesystem path.

    Strategy:
    - First try the basename (filename) for speed and to avoid accidental matches
      in parent directories.
    - If not found, fall back to searching the full path.

    Args:
        path: Full path to a file.

    Returns:
        (iso_date, unix_ts) if a date is found, otherwise (None, None).
    """
    base = os.path.basename(path)
    iso, ts = timestamp_from_string(base)
    if iso is not None:
        return iso, ts
    return timestamp_from_string(path)
