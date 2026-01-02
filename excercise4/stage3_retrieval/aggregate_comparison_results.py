"""
Stage 4 — Aggregate Stage 3 Temporal Analysis Results

Builds an aggregated JSON report from Stage 3 output (baseline vs time-aware retrieval comparisons).

Purpose
-------
Analyzes the temporal analysis results to compute:
- Delta statistics (entered/left/overlap counts and rates)
- Jaccard similarity and churn rates per pipeline/k
- Year distribution histograms (baseline/timeaware/entered/left)
- Breakdown by query_group (temporal buckets)

Input
-----
JSON list from s_03_temporal_analysis.py, where each row contains:
- query_group, pipeline, k, topn
- baseline_top, timeaware_top
- delta (entered, left, overlap_count)

Output
------
Compact aggregated JSON with:
- Per (pipeline, k) statistics
- Overall averages
- Year histograms showing temporal distribution shifts
- Query group breakdowns

Usage Examples
--------------
Basic:
    python aggregate_comparison_results.py --in_json outputs/rag_runs/stage3_temporal_analysis/stage3_given_temporal_queries_20260102_080846.json

Custom output:
    python aggregate_comparison_results.py --in_json outputs/rag_runs/stage3_temporal_analysis/stage3_*.json --out_name my_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import OUTPUTS_DIR, STAGE3_COMPARISON_DIR, STAGE3_SUMMARY_DIR, ensure_dirs


# ============================================================
# Helper Functions
# ============================================================

def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    """Save JSON file with pretty formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_year(date_iso: Optional[str]) -> Optional[int]:
    """
    Extract year from 'YYYY-MM-DD' ISO format.
    Returns None if missing or invalid.
    """
    if not date_iso or not isinstance(date_iso, str):
        return None
    if len(date_iso) < 4:
        return None
    try:
        return int(date_iso[:4])
    except (ValueError, TypeError):
        return None


def pipeline_key(p: Dict[str, Any]) -> str:
    """Generate pipeline key string: 'chunking/representation'."""
    ch = (p or {}).get("chunking", "unknown")
    rep = (p or {}).get("representation", "unknown")
    return f"{ch}/{rep}"


def bump_year_hist(hist: Counter, items: List[Dict[str, Any]]) -> None:
    """
    Update year histogram from items with 'doc_date_iso' field.
    Items without valid dates are counted as 'unknown'.
    """
    for it in items or []:
        y = parse_year(it.get("doc_date_iso"))
        if y is not None:
            hist[str(y)] += 1
        else:
            hist["unknown"] += 1


def year_hist_to_dict(hist: Counter) -> Dict[str, int]:
    """
    Convert Counter to ordered dict: numeric years ascending, then 'unknown'.
    """
    years = [k for k in hist.keys() if k != "unknown"]
    years_sorted = sorted(years, key=lambda s: int(s) if s.isdigit() else 10**9)
    out: Dict[str, int] = {}
    for y in years_sorted:
        out[y] = int(hist[y])
    if "unknown" in hist:
        out["unknown"] = int(hist["unknown"])
    return out


# ============================================================
# Aggregation Data Structure
# ============================================================

@dataclass
class DeltaAggregator:
    """Aggregates delta statistics across multiple queries."""
    
    n_rows: int = 0
    sum_entered: int = 0
    sum_left: int = 0
    sum_overlap: int = 0
    sum_jaccard: float = 0.0
    sum_churn: float = 0.0

    # Year distribution histograms
    years_baseline_top: Counter = None  # type: ignore
    years_timeaware_top: Counter = None  # type: ignore
    years_entered: Counter = None  # type: ignore
    years_left: Counter = None  # type: ignore

    def __post_init__(self) -> None:
        if self.years_baseline_top is None:
            self.years_baseline_top = Counter()
        if self.years_timeaware_top is None:
            self.years_timeaware_top = Counter()
        if self.years_entered is None:
            self.years_entered = Counter()
        if self.years_left is None:
            self.years_left = Counter()

    def add_row(self, row: Dict[str, Any]) -> None:
        """Process one comparison row from stage 3 output."""
        self.n_rows += 1

        topn = int(row.get("topn", 0) or 0)
        delta = row.get("delta") or {}
        entered = delta.get("entered") or []
        left = delta.get("left") or []
        overlap = int(delta.get("overlap_count", 0) or 0)

        e = len(entered)
        l = len(left)

        # Compute Jaccard similarity on topN membership
        baseline_ids = {x.get("chunk_id") for x in (row.get("baseline_top") or []) if x.get("chunk_id")}
        timeaware_ids = {x.get("chunk_id") for x in (row.get("timeaware_top") or []) if x.get("chunk_id")}
        inter = len(baseline_ids & timeaware_ids)
        union = len(baseline_ids | timeaware_ids)
        jacc = (inter / union) if union > 0 else 0.0

        # Churn rate: fraction of changes in topN (normalized by 2*topn)
        churn = ((e + l) / (2 * topn)) if topn > 0 else 0.0

        self.sum_entered += e
        self.sum_left += l
        self.sum_overlap += overlap
        self.sum_jaccard += jacc
        self.sum_churn += churn

        # Update year histograms for baseline/timeaware tops
        bump_year_hist(self.years_baseline_top, row.get("baseline_top") or [])
        bump_year_hist(self.years_timeaware_top, row.get("timeaware_top") or [])

        # Build year lookup maps for entered/left chunks
        id_to_year_baseline: Dict[str, Optional[int]] = {}
        for it in row.get("baseline_top") or []:
            cid = it.get("chunk_id")
            if cid:
                id_to_year_baseline[cid] = parse_year(it.get("doc_date_iso"))

        id_to_year_timeaware: Dict[str, Optional[int]] = {}
        for it in row.get("timeaware_top") or []:
            cid = it.get("chunk_id")
            if cid:
                id_to_year_timeaware[cid] = parse_year(it.get("doc_date_iso"))

        # Track years for entered chunks (from timeaware perspective)
        for cid in entered:
            y = id_to_year_timeaware.get(cid)
            if y is None:
                self.years_entered["unknown"] += 1
            else:
                self.years_entered[str(y)] += 1

        # Track years for left chunks (from baseline perspective)
        for cid in left:
            y = id_to_year_baseline.get(cid)
            if y is None:
                self.years_left["unknown"] += 1
            else:
                self.years_left[str(y)] += 1

    def to_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        n = self.n_rows or 1
        return {
            "n_rows": self.n_rows,
            "total_entered": self.sum_entered,
            "total_left": self.sum_left,
            "total_overlap": self.sum_overlap,
            "avg_entered": round(self.sum_entered / n, 2),
            "avg_left": round(self.sum_left / n, 2),
            "avg_overlap": round(self.sum_overlap / n, 2),
            "avg_jaccard": round(self.sum_jaccard / n, 4),
            "avg_churn": round(self.sum_churn / n, 4),
            "year_distributions": {
                "baseline_top": year_hist_to_dict(self.years_baseline_top),
                "timeaware_top": year_hist_to_dict(self.years_timeaware_top),
                "entered": year_hist_to_dict(self.years_entered),
                "left": year_hist_to_dict(self.years_left),
            },
        }


# ============================================================
# Report Builder
# ============================================================

def build_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate stage 3 results into summary report.
    Groups by (pipeline, k) and by query_group.
    """
    agg: Dict[Tuple[str, int], DeltaAggregator] = {}
    agg_by_group: Dict[Tuple[str, int], Dict[str, DeltaAggregator]] = {}

    for row in rows:
        pkey = pipeline_key(row.get("pipeline") or {})
        k = int(row.get("k", 0) or 0)
        qg = row.get("query_group", "unknown") or "unknown"

        key = (pkey, k)

        # Initialize aggregators if needed
        if key not in agg:
            agg[key] = DeltaAggregator()
            agg_by_group[key] = {}

        if qg not in agg_by_group[key]:
            agg_by_group[key][qg] = DeltaAggregator()

        # Add row to aggregators
        agg[key].add_row(row)
        agg_by_group[key][qg].add_row(row)

    # Build output structure
    pipelines_out: List[Dict[str, Any]] = []
    for (pkey, k), a in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        by_group_out: Dict[str, Any] = {}
        for qg, ga in sorted(agg_by_group[(pkey, k)].items(), key=lambda x: x[0]):
            by_group_out[qg] = ga.to_summary()

        pipelines_out.append(
            {
                "pipeline": pkey,
                "k": k,
                "overall": a.to_summary(),
                "by_query_group": by_group_out,
            }
        )

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "n_input_rows": len(rows),
            "description": "Aggregated Stage 3 temporal analysis: baseline vs timeaware deltas, year distributions, query group breakdowns",
        },
        "pipelines": pipelines_out,
    }


# ============================================================
# Main Entry Point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: Aggregate Stage 3 temporal analysis results"
    )
    parser.add_argument(
        "--in_json",
        required=True,
        help="Path to Stage 3 output JSON (list of comparison rows from compare_baseline_vs_timeaware.py)",
    )
    parser.add_argument(
        "--out_root",
        default=str(STAGE3_SUMMARY_DIR.parent),
        help="Root output directory",
    )
    parser.add_argument(
        "--out_subdir",
        default=STAGE3_SUMMARY_DIR.name,
        help="Subfolder under out_root",
    )
    parser.add_argument(
        "--out_name",
        default=None,
        help="Optional exact output filename (json). Default: derived from input name",
    )

    args = parser.parse_args()

    # Load input
    print(f"Loading {args.in_json}...")
    rows = load_json(args.in_json)
    
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {args.in_json}, got: {type(rows)}")
    
    if not rows:
        raise ValueError(f"Input file {args.in_json} is empty")

    print(f"Processing {len(rows)} comparison rows...")
    
    # Build report
    report = build_report(rows)

    # Determine output path
    in_base = os.path.splitext(os.path.basename(args.in_json))[0]
    out_name = args.out_name or f"{in_base}__summary.json"
    out_path = os.path.join(args.out_root, args.out_subdir, out_name)

    # Save
    save_json(out_path, report)
    print(f"\n✓ Saved Stage 3 summary report to: {out_path}")
    print(f"  - {report['meta']['n_input_rows']} input rows")
    print(f"  - {len(report['pipelines'])} pipeline/k configurations")


if __name__ == "__main__":
    main()
