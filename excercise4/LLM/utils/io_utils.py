# utils/io_utils.py
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


# Temporal bucket constants for query grouping
TEMPORAL_BUCKETS = (
    "point_in_time",
    "recency",
    "explicit_range",
    "comparison",
    "evolution",
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_queries_json(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Supports:
      {
        "factual": ["q1", {"query": "...", "expected_source": [...]}, ...],
        "conceptual": [...]
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def normalize(lst):
        out = []
        for q in lst:
            if isinstance(q, str):
                out.append({"query": q, "expected_source": []})
            else:
                out.append(q)
        return out

    return {
        "factual": normalize(data.get("factual", [])),
        "conceptual": normalize(data.get("conceptual", [])),
    }


def build_output_path(
    base_dir: str,
    out_root: str,
    subdir: Optional[str],
    filename: Optional[str],
    tag: str,
) -> str:
    """
    out_root: e.g. "outputs/rag_runs"
    subdir: optional extra subfolder
    filename: optional exact file name (must end with .json)
    tag: used when filename not provided
    """
    out_dir = os.path.join(base_dir, out_root)
    if subdir:
        out_dir = os.path.join(out_dir, subdir)
    ensure_dir(out_dir)

    if filename:
        if not filename.endswith(".json"):
            filename += ".json"
        return os.path.join(out_dir, filename)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{tag}_{ts}.json")


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_temporal_queries_json(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load evaluation queries from JSON using *temporal buckets*.

    Expected format:
    {
      "point_in_time": [ "q1", {"query": "...", ...}, ... ],
      "recency": [ ... ],
      "explicit_range": [ ... ],
      "comparison": [ ... ],
      "evolution": [ ... ]
    }

    Returns:
        Dict mapping bucket -> list[{"query": str, "expected_source": [...], ...}]

    Notes:
    - Only buckets present in the JSON are returned.
    - Each item is normalized to a dict with at least {"query": ..., "expected_source": []}.
    - This function does NOT affect legacy code paths.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def normalize(lst):
        out = []
        for q in lst:
            if isinstance(q, str):
                out.append({"query": q, "expected_source": []})
            else:
                # Ensure expected_source exists to match your pipeline expectations
                q = dict(q)
                q.setdefault("expected_source", [])
                out.append(q)
        return out

    out: Dict[str, List[Dict[str, Any]]] = {}
    for bucket in TEMPORAL_BUCKETS:
        if bucket in data:
            out[bucket] = normalize(data[bucket])

    if not out:
        raise ValueError(
            f"No temporal buckets found in {path}. "
            f"Expected one of: {list(TEMPORAL_BUCKETS)}"
        )

    return out


def flatten_temporal_queries(path: str) -> List[str]:
    """
    Convenience helper: read a temporal-bucket JSON and return a flat list of query strings.
    Keeps bucket order defined in TEMPORAL_BUCKETS.
    """
    by_bucket = load_temporal_queries_json(path)
    out: List[str] = []
    for bucket in TEMPORAL_BUCKETS:
        for q in by_bucket.get(bucket, []):
            out.append(q["query"])
    return out


def flatten_temporal_queries_with_groups(path: str) -> List[Dict[str, Any]]:
    """
    Read temporal-bucket JSON and return a flat list with query_group metadata.
    
    Returns:
        List of dicts: [{"query": str, "query_group": str, "expected_source": [...]}, ...]
    """
    by_bucket = load_temporal_queries_json(path)
    out: List[Dict[str, Any]] = []
    for bucket in TEMPORAL_BUCKETS:
        for q in by_bucket.get(bucket, []):
            item = dict(q)
            item["query_group"] = bucket
            out.append(item)
    return out


def detect_query_schema(path: str) -> str:
    """
    Detect whether a queries JSON uses temporal buckets or legacy factual/conceptual schema.
    
    Returns:
        "temporal" if any TEMPORAL_BUCKETS key is found
        "legacy" if factual or conceptual keys are found
        "unknown" otherwise
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    keys = set(data.keys())
    
    if any(bucket in keys for bucket in TEMPORAL_BUCKETS):
        return "temporal"
    elif "factual" in keys or "conceptual" in keys:
        return "legacy"
    else:
        return "unknown"