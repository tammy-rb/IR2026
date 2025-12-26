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
    if not m:
        return None, None

    iso_date = m.group(0)  # "YYYY-MM-DD"
    dt = datetime.strptime(iso_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return iso_date, int(dt.timestamp())


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
