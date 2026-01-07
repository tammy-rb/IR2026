from __future__ import annotations

import json
import os
import re
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paths import OUTPUTS_DIR

# ======================================================
# PATHS
# ======================================================

RAG_RUNS_DIR = OUTPUTS_DIR / "rag_runs"
ANALYSIS_OUT_DIR = OUTPUTS_DIR / "analysis"

ANALYSIS_OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = ANALYSIS_OUT_DIR / "recency_runs_flat.csv"
OUT_JSONL = ANALYSIS_OUT_DIR / "recency_runs_flat.jsonl"

# ======================================================
# RECENCY FILES
# ======================================================

FILES = [
    "cli_single_timeaware_evo_20260104_185110.json",
    "cli_single_timeaware_evo_20260104_185526.json",
    "cli_single_timeaware_evo_20260104_190757.json",
    "cli_single_timeaware_evo_20260104_193609.json",
    "cli_single_timeaware_evo_20260104_193736.json",
    "cli_single_timeaware_evo_20260104_194845.json",
    "cli_single_timeaware_evo_20260104_195037.json",
    "cli_single_timeaware_evo_20260104_200351.json",
    "cli_single_timeaware_evo_20260104_200540.json",
    "cli_single_timeaware_evo_20260104_201025.json",
    "cli_single_timeaware_evo_20260104_201307.json",
    "cli_single_timeaware_evo_20260104_201529.json",
    "cli_single_timeaware_evo_20260104_201729.json",
    "cli_single_timeaware_evo_20260104_204439.json",
    "cli_single_timeaware_evo_20260104_204619.json",
    "cli_single_timeaware_evo_20260104_205322.json",
    "cli_single_timeaware_evo_20260104_205643.json",
    "cli_single_timeaware_evo_20260104_210544.json",
    "cli_single_timeaware_evo_20260104_210807.json",
]

# ======================================================
# HELPERS
# ======================================================

_TS_RE = re.compile(r"_(\d{8})_(\d{6})\.json$")

def parse_timestamp_from_filename(fname: str) -> Optional[str]:
    """
    Extract YYYYMMDD_HHMMSS → YYYY-MM-DD HH:MM:SS
    """
    m = _TS_RE.search(fname)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def normalize_root(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("results", "runs", "items"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        return [obj]
    return []


def is_idk(answer: str) -> bool:
    a = (answer or "").lower()
    patterns = [
        "i don't know",
        "i do not know",
        "cannot answer",
        "can't answer",
        "not enough information",
        "insufficient information",
        "based on the retrieved chunks",
    ]
    return any(p in a for p in patterns)


def extract_retrieved_dates(run: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[int], int]:
    retrieved = run.get("retrieved")
    dates: List[date] = []

    def ingest(item: Dict[str, Any]):
        chunk = item.get("chunk", item)
        d = chunk.get("doc_date_iso")
        if isinstance(d, str):
            try:
                dates.append(datetime.strptime(d, "%Y-%m-%d").date())
            except ValueError:
                pass

    if isinstance(retrieved, list):
        for it in retrieved:
            ingest(it)
    elif isinstance(retrieved, dict):
        for arr in retrieved.values():
            if isinstance(arr, list):
                for it in arr:
                    ingest(it)

    if not dates:
        return None, None, None, 0

    dmin, dmax = min(dates), max(dates)
    return dmin.isoformat(), dmax.isoformat(), (dmax - dmin).days, len(dates)

# ======================================================
# DATA MODEL
# ======================================================

@dataclass
class FlatRun:
    file_name: str
    file_timestamp: Optional[str]

    query: Optional[str]
    query_group: Optional[str]

    pipeline_chunking: Optional[str]
    pipeline_representation: Optional[str]

    k: Optional[int]
    retrieval_mode: Optional[str]
    timeaware: Optional[bool]

    answer_is_idk: bool
    answer_length: int
    answer: str

    retrieved_min_date: Optional[str]
    retrieved_max_date: Optional[str]
    retrieved_span_days: Optional[int]
    retrieved_items_with_date: int


# ======================================================
# CORE LOGIC
# ======================================================

def flatten_file(path: str) -> List[FlatRun]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = normalize_root(data)
    out: List[FlatRun] = []

    file_name = os.path.basename(path)
    file_ts = parse_timestamp_from_filename(file_name)

    for run in runs:
        pipeline = run.get("pipeline", {}) or {}

        answer = run.get("answer", "") or ""
        rmin, rmax, rspan, rcount = extract_retrieved_dates(run)

        out.append(
            FlatRun(
                file_name=file_name,
                file_timestamp=file_ts,
                query=run.get("query"),
                query_group=run.get("query_group"),
                pipeline_chunking=pipeline.get("chunking"),
                pipeline_representation=pipeline.get("representation"),
                k=run.get("k"),
                retrieval_mode=run.get("retrieval_mode"),
                timeaware=run.get("timeaware"),
                answer_is_idk=is_idk(answer),
                answer_length=len(answer),
                answer=answer,
                retrieved_min_date=rmin,
                retrieved_max_date=rmax,
                retrieved_span_days=rspan,
                retrieved_items_with_date=rcount,
            )
        )

    return out


def save_csv(rows: List[FlatRun]) -> None:
    with open(str(OUT_CSV), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def save_jsonl(rows: List[FlatRun]) -> None:
    with open(str(OUT_JSONL), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


# ======================================================
# ENTRY POINT
# ======================================================

def main() -> None:
    all_rows: List[FlatRun] = []

    for fname in FILES:
        path = RAG_RUNS_DIR / fname
        if not path.exists():
            print(f"[WARN] Missing file: {path}")
            continue
        all_rows.extend(flatten_file(str(path)))

    all_rows.sort(
        key=lambda r: (
            r.query or "",
            r.pipeline_chunking or "",
            r.pipeline_representation or "",
            r.k or -1,
        )
    )

    if not all_rows:
        print("No runs extracted.")
        return

    save_csv(all_rows)
    save_jsonl(all_rows)

    print(f"Saved {len(all_rows)} rows")
    print(f"→ {OUT_CSV}")
    print(f"→ {OUT_JSONL}")


if __name__ == "__main__":
    main()
