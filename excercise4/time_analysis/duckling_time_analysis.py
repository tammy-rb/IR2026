"""
time_analysis/duckling_time_analysis.py

Stage 3 - Time-aware retrieval (temporal signal extraction)

This module:
1) Sends a POST request to a local Duckling server with the query text.
2) Parses Duckling's response and derives a high-level temporal "granularity".
3) Returns a JSON-like dict with all temporal fields needed downstream for:
   - hard filtering
   - soft decay / recency weighting
   (Decision logic for hard/soft/reference point is NOT done here.)

Expected Duckling endpoint:
  POST http://localhost:8000/parse
Payload (form data):
  locale=en_US
  text=<query>
  dims=["time"]

Key behaviors:
- Duckling does not return a single "granularity" field. We derive it from:
  - value.type:  "value" vs "interval"
  - value.grain: year / month / day / hour / ...
- Open-ended intervals are supported (both directions):
  - Open-end   (from only):  start=<date>, end=None
  - Open-start (to only):    start=None,   end=<date>   ✅ (added)
- Mode semantics (high-level):
  - explicit: Duckling returned a concrete time value or interval
  - current: query explicitly asks for "current/now/today/present"
  - recent: query implies recency, but fuzzy/underspecified (e.g., "recently", "past decade")
  - none: no temporal intent detected
- Priority:
  - If query intent is current -> mode="current" (even if Duckling returns "now/today" as a value)
  - If query intent is recent -> mode="recent" ONLY when there are no explicit bounded ranges
    (If there are explicit ranges, mode stays "explicit")

priority logic rationale:
- We treat EXPLICIT time spans returned by Duckling as the strongest signal.
    In particular, any INTERVAL (bounded or open) implies the user stated a
    concrete temporal constraint ("between", "since", "until"), so we force
    mode="explicit" even if the query also contains words like "now/today".
    This prevents "current" intent from accidentally overriding a real
    time filter (e.g., "from 2020 until now" must behave like an explicit range).
- If Duckling found only POINT values (single dates/months/years), we allow
    "current" intent to win, because phrases like "now/today/current" are often
    the actual retrieval goal even when a historical date is mentioned in context
    (e.g., "what is the current stance about events in 1917?").
- If there are no Duckling ranges at all, we fall back to the keyword intent
    detector: current > recent > none.

Important:
- This module does NOT decide hard/soft strategy or reference points.
- It only produces normalized time signals for downstream decisions.

"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import requests


# -----------------------------
# Configuration
# -----------------------------
DEFAULT_DUCKLING_URL = "http://localhost:8000/parse"
DEFAULT_LOCALE = "en_US"

# Strongly "current" intent (anchored to now)
CURRENT_KEYWORDS_RE = re.compile(r"\b(current|today|now|present)\b", re.IGNORECASE)

# "Recent" intent (recency preference, but fuzzy / not a strict boundary)
RECENT_INTENT_RE = re.compile(
    r"\b("
    r"recent\s+years|"
    r"past\s+decade|"
    r"last\s+decade|"
    r"recently|"
    r"lately|"
    r"in\s+recent\s+times|"
    r"over\s+the\s+past\s+\d+\s+years|"
    r"past\s+\d+\s+years"
    r")\b",
    re.IGNORECASE,
)

# -----------------------------
# Granularity mapping
# -----------------------------
# Maps Duckling grain -> system granularity
# Design decision:
# - Sub-day resolutions are normalized to "day"
DUCKLING_GRAIN_TO_GRANULARITY = {
    "year": "year",
    "quarter": "quarter",
    "month": "month",
    "week": "week",
    "day": "day",

    # Sub-day → day
    "hour": "day",
    "minute": "day",
    "second": "day",
}


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class DucklingTimeSpan:
    dim: str                 # Duckling dimension (always "time" here)
    body: str                # Exact matched text in the query (e.g., "April 2022")
    start: int               # Start char index of the match in the query string
    end: int                 # End char index of the match in the query string
    latent: bool             # True if inferred, False if explicitly stated
    value: Dict[str, Any]    # Raw Duckling value object (type, grain, value/from/to, ...)


@dataclass(frozen=True)
class TimeRange:
    """
    Normalized time range.

    - start/end are ISO dates (YYYY-MM-DD) or None.
    - Supports:
        * point:        start=end=<date>
        * bounded:      start=<date>, end=<date>
        * open-end:     start=<date>, end=None
        * open-start:   start=None,   end=<date>
    """
    id: str
    source: str                 # "duckling"
    text: str                   # matched text ("body")
    start: Optional[str]        # ISO date or None (open-start)
    end: Optional[str]          # ISO date or None (open-end)
    open_ended: bool            # True if start is None OR end is None
    duckling_type: str          # "value" | "interval"
    duckling_grain: str         # year/month/day/week/...
    granularity: str            # year/quarter/month/week/day/range/unknown
    kind: str                   # "point" | "bounded_range" | "open_range"


# -----------------------------
# Helpers
# -----------------------------
def _parse_iso_datetime(s: str) -> datetime:
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def _date_from_duckling_value(v: str) -> date:
    return _parse_iso_datetime(v).date()


def _inclusive_end_from_duckling_to(to_value: str) -> date:
    # Duckling interval 'to' is typically exclusive -> convert to inclusive date end
    return _date_from_duckling_value(to_value) - timedelta(days=1)


def _derive_granularity(duckling_type: str, duckling_grain: Optional[str]) -> str:
    """
    Derive high-level temporal granularity from Duckling output.

    Rules:
    - interval  -> "range"
    - value     -> mapped via DUCKLING_GRAIN_TO_GRANULARITY
    - unknown / unsupported grains -> "unknown"
    """
    if duckling_type == "interval":
        return "range"

    if duckling_type == "value":
        if duckling_grain is None:
            return "unknown"
        return DUCKLING_GRAIN_TO_GRANULARITY.get(duckling_grain, "unknown")

    return "unknown"


def _intent_priority(query: str) -> str:
    """
    Optional. Detect high-level temporal intent keywords in the query and apply precedence.
    This helps with mixed-intent queries like "what is the current position about the war happened in 1917?",
    where "current" should take priority over a historical year mention.

    Priority order:
      1) current (now/today/current/present) -> strongest anchoring to now
      2) recent  (recently/decade/...)      -> recency preference, fuzzy
      3) none
    """
    if CURRENT_KEYWORDS_RE.search(query):
        return "current"
    if RECENT_INTENT_RE.search(query):
        return "recent"
    return "none"


def _final_mode(query: str, ranges: List[TimeRange]) -> str:
    """
    Decide mode with priority rules.

    NOTE (priority logic rationale):
    - We treat EXPLICIT time spans returned by Duckling as the strongest signal.
      Any INTERVAL (bounded or open — including open-start/open-end) implies a
      concrete temporal constraint ("between", "since", "until"), so we force
      mode="explicit" even if the query also contains words like "now/today".
    - If Duckling found only POINT values (single dates/months/years), we allow
      "current" intent to win, because phrases like "now/today/current" are often
      the actual retrieval goal even when a historical date is mentioned in context.
    - If there are no Duckling ranges at all, we fall back to the keyword intent
      detector: current > recent > none.

    Returns: "explicit" | "current" | "recent" | "none"
    """
    intent = _intent_priority(query)

    if not ranges:
        return intent

    if any(r.duckling_type == "interval" for r in ranges):
        return "explicit"

    if intent == "current":
        return "current"

    return "explicit"


# -----------------------------
# Duckling client
# -----------------------------
def call_duckling(
    query: str,
    *,
    url: str = DEFAULT_DUCKLING_URL,
    locale: str = DEFAULT_LOCALE,
    timeout: float = 10.0,
) -> List[DucklingTimeSpan]:
    resp = requests.post(
        url,
        data={"locale": locale, "text": query, "dims": '["time"]'},
        timeout=timeout,
    )
    resp.raise_for_status()
    raw = resp.json()

    spans: List[DucklingTimeSpan] = []
    for e in raw:
        if e.get("dim") == "time":
            spans.append(
                DucklingTimeSpan(
                    dim=e.get("dim", "time"),
                    body=e.get("body", ""),
                    start=int(e.get("start", 0)),
                    end=int(e.get("end", 0)),
                    latent=bool(e.get("latent", False)),
                    value=dict(e.get("value", {})),
                )
            )
    return spans


# -----------------------------
# Main API
# -----------------------------
def analyze_query_time(
    query: str,
    *,
    duckling_url: str = DEFAULT_DUCKLING_URL,
    locale: str = DEFAULT_LOCALE,
    now: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Analyze a query and return a JSON-like dict containing temporal metadata.

    Output schema:
    {
      "query": <original query>,
      "now_iso": "YYYY-MM-DD",
      "mode": "explicit" | "current" | "recent" | "none",
      "ranges": [
        {
          "id": "t1",
          "source": "duckling",
          "text": <matched body>,
          "start": "YYYY-MM-DD" | null,   # null => open-start
          "end": "YYYY-MM-DD" | null,     # null => open-end
          "open_ended": true|false,       # true if start is null OR end is null
          "duckling_type": "value" | "interval",
          "duckling_grain": "year" | "month" | "day" | "week" | ...,
          "granularity": "year" | "quarter" | "month" | "week" | "day" | "range" | "unknown",
          "kind": "point" | "bounded_range" | "open_range"
        },
        ...
      ],
      "duckling_raw": [ ... ]   # minimal raw fields for debugging/repro
    }
    """
    now = now or date.today()

    spans = call_duckling(query, url=duckling_url, locale=locale)

    ranges: List[TimeRange] = []
    for idx, span in enumerate(spans, start=1):
        val = span.value
        vtype = val.get("type")

        # Duckling sometimes nests values under `values` but also provides top-level fields.
        top = val
        values_list = val.get("values")
        if (vtype is None) and isinstance(values_list, list) and values_list:
            top = values_list[0]
            vtype = top.get("type")

        # -------------------------
        # Case 1: point value
        # -------------------------
        if vtype == "value":
            grain = top.get("grain") or val.get("grain")
            v = top.get("value") or val.get("value")
            if isinstance(v, str):
                d = _date_from_duckling_value(v)
                ranges.append(
                    TimeRange(
                        id=f"t{idx}",
                        source="duckling",
                        text=span.body,
                        start=d.isoformat(),
                        end=d.isoformat(),
                        open_ended=False,
                        duckling_type="value",
                        duckling_grain=str(grain) if grain else "unknown",
                        granularity=_derive_granularity("value", str(grain) if grain else None),
                        kind="point",
                    )
                )

        # -------------------------
        # Case 2: interval value
        # -------------------------
        elif vtype == "interval":
            frm = top.get("from") or val.get("from") or {}
            to = top.get("to") or val.get("to") or {}

            frm_val = frm.get("value")
            to_val = to.get("value")

            interval_grain = (
                frm.get("grain")
                or to.get("grain")
                or top.get("grain")
                or val.get("grain")
                or "unknown"
            )

            # A) bounded interval: from + to
            if isinstance(frm_val, str) and isinstance(to_val, str):
                start_date = _date_from_duckling_value(frm_val)
                end_date = _inclusive_end_from_duckling_to(to_val)
                ranges.append(
                    TimeRange(
                        id=f"t{idx}",
                        source="duckling",
                        text=span.body,
                        start=start_date.isoformat(),
                        end=end_date.isoformat(),
                        open_ended=False,
                        duckling_type="interval",
                        duckling_grain=str(interval_grain),
                        granularity=_derive_granularity("interval", str(interval_grain)),
                        kind="bounded_range",
                    )
                )

            # B) open-end interval: from only (start known, end unknown)
            elif isinstance(frm_val, str) and not to_val:
                start_date = _date_from_duckling_value(frm_val)
                ranges.append(
                    TimeRange(
                        id=f"t{idx}",
                        source="duckling",
                        text=span.body,
                        start=start_date.isoformat(),
                        end=None,
                        open_ended=True,
                        duckling_type="interval",
                        duckling_grain=str(interval_grain),
                        granularity="range",
                        kind="open_range",
                    )
                )

            # C) open-start interval: to only (start unknown, end known) ✅ added
            elif isinstance(to_val, str) and not frm_val:
                end_date = _inclusive_end_from_duckling_to(to_val)
                ranges.append(
                    TimeRange(
                        id=f"t{idx}",
                        source="duckling",
                        text=span.body,
                        start=None,
                        end=end_date.isoformat(),
                        open_ended=True,
                        duckling_type="interval",
                        duckling_grain=str(interval_grain),
                        granularity="range",
                        kind="open_range",
                    )
                )

        else:
            continue

    mode = _final_mode(query, ranges)

    duckling_raw = [
        {
            "body": s.body,
            "start": s.start,
            "end": s.end,
            "latent": s.latent, # True if inferred, False if explicit
            "value": s.value,
        }
        for s in spans
    ]

    return {
        "query": query,
        "now_iso": now.isoformat(),
        "mode": mode,  # explicit/current/recent/none
        "ranges": [asdict(r) for r in ranges],
        "duckling_raw": duckling_raw,
    }


# -----------------------------
# CLI / Quick manual test
# -----------------------------
if __name__ == "__main__":
    tests = [
        "what happended in april in 2022?",
        "give me the reports from march 15, 2023 until now",
        "show me data between january 1, 2020 and december 31, 2020",
        "show me reports until january 1, 2020", 
        "what was the situation last week?",
        "who is the prime minister now?",
        "what is the biggest news today?",
        "how has the economy changed recently?",
        "over the past decade, what changed?",
        "since 2019, what has changed?",
        "who is the hero in superball?",
    ]

    for q in tests:
        print("=" * 80)
        print(q)
        try:
            res = analyze_query_time(q)
            print(res)
        except Exception as e:
            print(f"ERROR: {e}")
