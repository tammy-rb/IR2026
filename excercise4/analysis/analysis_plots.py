# plot_temporal_eval.py
#
# Usage:
#   python plot_temporal_eval.py --input labels.json --outdir plots
#
# Input format expected:
# {
#   "label_map": {"0":"IDK","1":"Incorrect","2":"Correct"},
#   "rows":[
#     {"temporal_type":"recency","pipeline_chunking":"fixed","pipeline_representation":"bm25","k":3,"answer_label":2, ...},
#     ...
#   ]
# }

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import OUTPUTS_DIR

ANALYSIS_OUT_DIR = OUTPUTS_DIR / "analysis"


LABEL_IDK = 0
LABEL_INCORRECT = 1
LABEL_CORRECT = 2

# ----------------------------
# Point-in-time subtypes (by results file_name)
# ----------------------------
PIT_NUMERIC_FILES = {
    "cli_single_timeaware_evo_20260104_193609.json",  # US security budget 2024
    "cli_single_timeaware_evo_20260104_193736.json",  # UK defence budget 2024
}

PIT_TOPIC_FILES = {
    "cli_single_timeaware_evo_20260104_194845.json",  # US healthcare legislation 2024
    "cli_single_timeaware_evo_20260104_195037.json",  # UK healthcare legislation 2024
}


@dataclass(frozen=True)
class Key:
    temporal_type: str
    pipeline: str
    k: int


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pipeline_name(chunking: str, representation: str) -> str:
    return f"{chunking}/{representation}"


def rows_for_temporal_subtype(
    rows: List[dict],
    base_temporal_type: str,
    new_temporal_type: str,
    allowed_file_names: set,
) -> List[dict]:
    """
    Filter rows by:
      - temporal_type == base_temporal_type
      - file_name in allowed_file_names

    And rewrite temporal_type to new_temporal_type so summarize() groups them separately.
    """
    out: List[dict] = []
    for r in rows:
        t = (r.get("temporal_type") or "").strip() or "unknown"
        if t != base_temporal_type:
            continue

        fn = str(r.get("file_name", "")).strip()
        if fn not in allowed_file_names:
            continue

        r2 = dict(r)
        r2["temporal_type"] = new_temporal_type
        out.append(r2)

    return out

def compute_idk_rate(counts: Dict[int, int]) -> float:
    """IDK rate: IDK / total"""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return counts.get(LABEL_IDK, 0) / total


def make_idk_rate_plot(
    temporal_type: str,
    items: List[Tuple[Key, Dict[int, int]]],
    outdir: str,
) -> str:
    """
    Bar plot: x = pipeline+k, y = IDK rate.
    Treats label 1 and 2 as "answered" (merged), label 0 as IDK.
    """
    scored = []
    for key, counts in items:
        idk_rate = compute_idk_rate(counts)
        scored.append((idk_rate, key, counts))

    # Sort by IDK rate desc, then pipeline, then k
    scored.sort(key=lambda x: (-x[0], x[1].pipeline, x[1].k))

    labels = [f"{key.pipeline} | k={key.k}" for _, key, _ in scored]
    values = [rate for rate, _, _ in scored]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(values)), values)
    plt.ylim(0.0, 1.0)
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    plt.ylabel("IDK rate (IDK / Total)")
    plt.title(f"IDK rate (Answered=Incorrect+Correct) - {temporal_type}")
    plt.tight_layout()

    outpath = os.path.join(outdir, f"{temporal_type}__idk_rate.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath

def compute_accuracy(counts: Dict[int, int]) -> float:
    """Accuracy including IDK in denominator: correct / total"""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return counts.get(LABEL_CORRECT, 0) / total


def compute_accuracy_excluding_idk(counts: Dict[int, int]) -> float:
    """Accuracy excluding IDK: correct / (correct + incorrect)"""
    total = counts.get(LABEL_CORRECT, 0) + counts.get(LABEL_INCORRECT, 0)
    if total == 0:
        return 0.0
    return counts.get(LABEL_CORRECT, 0) / total


def compute_macro_f1(counts: Dict[int, int]) -> float:
    """
    Computes macro-F1 for 3-class classification using only counts,
    assuming single-label multi-class predictions and ground truth labels.
    Here, 'counts' are gold labels only, so true TP/FP/FN aren't known.
    => We can’t compute real F1 from ONLY gold-label distribution.

    So: this script computes *balanced accuracy proxy* instead:
    mean of per-class recall, which is computable from gold counts only
    if we also had predictions, but we do not.

    Because your JSON rows are *judged outputs* (one label per run),
    the label is effectively "correctness class", not a predicted class vs gold.
    In that case, F1 is not meaningful.

    We'll return None-like value by using -1.0 and skip plotting unless enabled.
    """
    return -1.0


def summarize(rows: List[dict]) -> Dict[Key, Dict[int, int]]:
    """
    Returns counts per (temporal_type, pipeline, k):
      counts[label] = how many rows have that label
    """
    agg: Dict[Key, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for r in rows:
        t = (r.get("temporal_type") or "").strip()
        if not t:
            # If missing, group under "unknown"
            t = "unknown"

        chunking = str(r.get("pipeline_chunking", ""))
        rep = str(r.get("pipeline_representation", ""))
        k = int(r.get("k", -1))

        label = r.get("answer_label", None)
        if label is None:
            # fallback: if answer_is_idk true => IDK else unknown -> treat as Incorrect
            if bool(r.get("answer_is_idk", False)):
                label = LABEL_IDK
            else:
                label = LABEL_INCORRECT
        label = int(label)

        key = Key(
            temporal_type=t,
            pipeline=pipeline_name(chunking, rep),
            k=k,
        )
        agg[key][label] += 1

    return agg


def make_accuracy_plot(
    temporal_type: str,
    items: List[Tuple[Key, Dict[int, int]]],
    outdir: str,
    exclude_idk: bool = False,
) -> str:
    """
    Bar plot: x = pipeline+k, y = accuracy
    """
    # Sort by accuracy desc, then by pipeline, then k
    scored = []
    for key, counts in items:
        if exclude_idk:
            acc = compute_accuracy_excluding_idk(counts)
        else:
            acc = compute_accuracy(counts)
        scored.append((acc, key, counts))
    scored.sort(key=lambda x: (-x[0], x[1].pipeline, x[1].k))

    labels = [f"{key.pipeline} | k={key.k}" for _, key, _ in scored]
    values = [acc for acc, _, _ in scored]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(values)), values)
    plt.ylim(0.0, 1.0)
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    
    if exclude_idk:
        plt.ylabel("Accuracy (Correct / (Correct + Incorrect))")
        plt.title(f"Accuracy excluding IDK - {temporal_type}")
        suffix = "__accuracy_no_idk.png"
    else:
        plt.ylabel("Accuracy (Correct / Total)")
        plt.title(f"Accuracy including IDK - {temporal_type}")
        suffix = "__accuracy_with_idk.png"
    
    plt.tight_layout()

    outpath = os.path.join(outdir, f"{temporal_type}{suffix}")
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath


def print_top_k(
    temporal_type: str,
    items: List[Tuple[Key, Dict[int, int]]],
    topn: int = 5,
) -> None:
    scored = []
    for key, counts in items:
        acc = compute_accuracy(counts)
        total = sum(counts.values())
        scored.append((acc, total, key, counts))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2].pipeline, x[2].k))

    print(f"\n=== {temporal_type}: top {topn} by accuracy ===")
    for i, (acc, total, key, counts) in enumerate(scored[:topn], start=1):
        c = counts.get(LABEL_CORRECT, 0)
        inc = counts.get(LABEL_INCORRECT, 0)
        idk = counts.get(LABEL_IDK, 0)
        print(f"{i:>2}. {key.pipeline:>14} | k={key.k:<2} | acc={acc:.3f} | n={total} (C={c}, I={inc}, IDK={idk})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to labels JSON (with label_map + rows)")
    default_outdir = str(ANALYSIS_OUT_DIR / "plots")
    ap.add_argument("--outdir", default=default_outdir, help="Output directory for PNG plots")
    ap.add_argument(
        "--types",
        nargs="*",
        default=["recency", "point_in_time", "evolution", "point_in_time_numeric", "point_in_time_topic"],
        help="Temporal types to plot (default: recency point_in_time evolution + PIT subtypes). "
             "If a listed type isn't found, it will be skipped.",
    )
    args = ap.parse_args()

    data = load_json(args.input)
    rows = data.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input JSON has no 'rows' list (or it's empty).")

    safe_mkdir(args.outdir)

    # -----------------------------------------
    # point-in-time subtype views (2 queries each)
    # -----------------------------------------
    pit_numeric_rows = rows_for_temporal_subtype(
        rows=rows,
        base_temporal_type="point_in_time",
        new_temporal_type="point_in_time_numeric",
        allowed_file_names=PIT_NUMERIC_FILES,
    )

    pit_topic_rows = rows_for_temporal_subtype(
        rows=rows,
        base_temporal_type="point_in_time",
        new_temporal_type="point_in_time_topic",
        allowed_file_names=PIT_TOPIC_FILES,
    )

    # Extend the dataset with these "virtual types"
    rows = rows + pit_numeric_rows + pit_topic_rows

    # IMPORTANT: summarize AFTER extending rows
    agg = summarize(rows)

    # Group by temporal_type
    by_type: Dict[str, List[Tuple[Key, Dict[int, int]]]] = defaultdict(list)
    for key, counts in agg.items():
        by_type[key.temporal_type].append((key, counts))

    # If user specified types, plot those; otherwise plot all present
    types_to_plot = args.types if args.types else sorted(by_type.keys())

    produced = []
    for t in types_to_plot:
        if t not in by_type:
            print(f"[skip] temporal_type '{t}' not found in file.")
            continue

        items = by_type[t]
        print_top_k(t, items, topn=8)

        # Plot IDK rate
        outpath3 = make_idk_rate_plot(t, items, args.outdir)
        produced.append(outpath3)
        print(f"[saved] {outpath3}")

        # Plot with IDK included
        outpath1 = make_accuracy_plot(t, items, args.outdir, exclude_idk=False)
        produced.append(outpath1)
        print(f"[saved] {outpath1}")

        # Plot with IDK excluded
        outpath2 = make_accuracy_plot(t, items, args.outdir, exclude_idk=True)
        produced.append(outpath2)
        print(f"[saved] {outpath2}")

    if not produced:
        print("No plots produced. Check your --types values or whether 'temporal_type' exists in rows.")

if __name__ == "__main__":
    main()
