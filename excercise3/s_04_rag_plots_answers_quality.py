# plot_answer_quality.py
# Plots answer-quality labels by pipeline and K (and by query_type).
#
# Rules:
# - If some answers are missing labels for a (pipeline, K) config, fill the missing ones as "Unknown".
# - Average score uses: Correct=1, Partially correct=0.5, Incorrect=0, Unknown=0 (included).

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend – no Tk, no GUI

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def _load_labels_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_label_order(data: dict) -> List[str]:
    order = data.get("label_order")
    if isinstance(order, list) and order:
        return order
    return ["Correct", "Partially correct", "Incorrect", "Unknown"]


def _score_map(label_order: List[str]) -> Dict[str, float]:
    """
    Correct=1, Partially=0.5, Incorrect=0, Unknown=0
    """
    mapping: Dict[str, float] = {}
    for lab in label_order:
        low = lab.lower()
        if low.startswith("correct"):
            mapping[lab] = 1.0
        elif low.startswith("part"):
            mapping[lab] = 0.5
        elif low.startswith("incorr"):
            mapping[lab] = 0.0
        else:
            mapping[lab] = 0.0
    return mapping


def _group_counts(
    rows: List[dict],
    label_order: List[str],
    default_label: str,
) -> Tuple[List[str], List[int], Dict[Tuple[str, int], Dict[str, int]], List[str]]:
    pipelines_set = set()
    ks_set = set()
    qtypes_set = set()

    counts: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(lambda: {lab: 0 for lab in label_order})

    for r in rows:
        pipeline = r.get("pipeline", "unknown_pipeline")
        k = int(r.get("k", -1))
        qtype = r.get("query_type", "unknown")

        lab = r.get("label", default_label) or default_label
        if lab not in label_order:
            if default_label in label_order:
                lab = default_label
            else:
                label_order.append(lab)
                for key in list(counts.keys()):
                    counts[key].setdefault(lab, 0)

        pipelines_set.add(pipeline)
        ks_set.add(k)
        qtypes_set.add(qtype)

        counts[(pipeline, k)][lab] = counts[(pipeline, k)].get(lab, 0) + 1

    pipelines = sorted(pipelines_set)
    ks = sorted(ks_set)
    qtypes = sorted(qtypes_set)
    return pipelines, ks, counts, qtypes


def _group_counts_by_qtype(
    rows: List[dict],
    label_order: List[str],
    default_label: str,
) -> Dict[str, Dict[Tuple[str, int], Dict[str, int]]]:
    out: Dict[str, Dict[Tuple[str, int], Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {lab: 0 for lab in label_order})
    )

    for r in rows:
        qtype = r.get("query_type", "unknown")
        pipeline = r.get("pipeline", "unknown_pipeline")
        k = int(r.get("k", -1))

        lab = r.get("label", default_label) or default_label
        if lab not in label_order:
            lab = default_label if default_label in label_order else lab
            if lab not in label_order:
                label_order.append(lab)

        out[qtype][(pipeline, k)][lab] = out[qtype][(pipeline, k)].get(lab, 0) + 1

    return out


def _fill_unknowns_per_config(
    counts: Dict[Tuple[str, int], Dict[str, int]],
    pipelines: List[str],
    ks: List[int],
    label_order: List[str],
    expected_n_per_config: int,
) -> None:
    """
    For each (pipeline, k), if total labeled answers < expected_n_per_config,
    add the missing amount to Unknown.
    """
    if "Unknown" not in label_order:
        label_order.append("Unknown")
        # ensure existing dicts have Unknown key
        for key in list(counts.keys()):
            counts[key].setdefault("Unknown", 0)

    for p in pipelines:
        for k in ks:
            c = counts[(p, k)]
            # make sure any newly added labels exist in dict
            for lab in label_order:
                c.setdefault(lab, 0)

            total = sum(int(c.get(lab, 0)) for lab in label_order)
            missing = expected_n_per_config - total
            if missing > 0:
                c["Unknown"] = int(c.get("Unknown", 0)) + missing


def _prepare_matrix(
    pipelines: List[str],
    ks: List[int],
    label_order: List[str],
    counts: Dict[Tuple[str, int], Dict[str, int]],
) -> Dict[str, np.ndarray]:
    mats = {lab: np.zeros((len(pipelines), len(ks)), dtype=int) for lab in label_order}
    for i, p in enumerate(pipelines):
        for j, k in enumerate(ks):
            c = counts.get((p, k), {lab: 0 for lab in label_order})
            for lab in label_order:
                mats[lab][i, j] = int(c.get(lab, 0))
    return mats


# ----------------------------
# Plots
# ----------------------------
def _plot_stacked_bars(
    pipelines: List[str],
    ks: List[int],
    label_order: List[str],
    mats_counts: Dict[str, np.ndarray],
    out_path: str,
    title: str,
    normalize: bool,
) -> None:
    n = len(pipelines)
    if n == 0:
        return

    fig_h = max(3.5, 2.6 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    x = np.arange(len(ks))
    x_labels = [str(k) for k in ks]

    for i, p in enumerate(pipelines):
        ax = axes[i]

        totals = np.zeros(len(ks), dtype=float)
        for lab in label_order:
            totals += mats_counts[lab][i, :].astype(float)

        bottom = np.zeros(len(ks), dtype=float)

        for lab in label_order:
            vals = mats_counts[lab][i, :].astype(float)
            if normalize:
                with np.errstate(divide="ignore", invalid="ignore"):
                    vals = np.where(totals > 0, (vals / totals) * 100.0, 0.0)

            ax.bar(x, vals, bottom=bottom, label=lab)
            bottom += vals

        ax.set_ylabel(p)
        ax.grid(True, axis="y", alpha=0.25)
        if normalize:
            ax.set_ylim(0, 100)

        for j in range(len(ks)):
            t = int(totals[j])
            ax.text(
                j,
                (100.5 if normalize else bottom[j] + 0.2),
                f"n={t}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels)
    axes[-1].set_xlabel("K")

    fig.suptitle(title, y=0.995)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(label_order)), bbox_to_anchor=(0.5, 0.98))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_avg_score_heatmap(
    pipelines: List[str],
    ks: List[int],
    label_order: List[str],
    counts: Dict[Tuple[str, int], Dict[str, int]],
    out_path: str,
    title: str,
) -> None:
    score = _score_map(label_order)
    mat = np.full((len(pipelines), len(ks)), np.nan, dtype=float)

    for i, p in enumerate(pipelines):
        for j, k in enumerate(ks):
            c = counts.get((p, k))
            if not c:
                continue

            num = 0.0
            den = 0.0
            for lab in label_order:
                v = float(c.get(lab, 0))
                s = float(score.get(lab, 0.0))
                num += s * v
                den += v

            mat[i, j] = (num / den) if den > 0 else np.nan

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.6 * len(pipelines) + 2.5)))
    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticks(np.arange(len(pipelines)))
    ax.set_yticklabels(pipelines)

    ax.set_xlabel("K")
    ax.set_ylabel("Pipeline")
    ax.set_title(title)

    for i in range(len(pipelines)):
        for j in range(len(ks)):
            v = mat[i, j]
            ax.text(j, i, "-" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Average quality score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_json", default="answer_quality_labels.json", help="Path to the labels JSON file.")
    ap.add_argument("--out_dir", default=os.path.join("outputs", "answer_quality_plots"), help="Directory to save plots.")
    ap.add_argument(
        "--expected_n_per_config",
        type=int,
        default=8,
        help="Total expected answers per (pipeline, K). Missing labels are filled as Unknown.",
    )
    args = ap.parse_args()

    data = _load_labels_json(args.labels_json)
    label_order = _get_label_order(data)
    default_label = data.get("default_label", "Unknown")

    rows = data.get("labels", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("No labels found in JSON (key: 'labels').")

    _ensure_dir(args.out_dir)

    pipelines, ks, counts, qtypes = _group_counts(rows, label_order, default_label)

    # Fill missing labels as Unknown so each (pipeline,k) has expected_n answers.
    if args.expected_n_per_config and args.expected_n_per_config > 0:
        _fill_unknowns_per_config(counts, pipelines, ks, label_order, args.expected_n_per_config)

    mats_counts = _prepare_matrix(pipelines, ks, label_order, counts)

    out1 = os.path.join(args.out_dir, "stacked_counts_by_pipeline_k.png")
    _plot_stacked_bars(
        pipelines=pipelines,
        ks=ks,
        label_order=label_order,
        mats_counts=mats_counts,
        out_path=out1,
        title="Answer quality distribution (counts) by Pipeline and K",
        normalize=False,
    )

    out2 = os.path.join(args.out_dir, "stacked_percent_by_pipeline_k.png")
    _plot_stacked_bars(
        pipelines=pipelines,
        ks=ks,
        label_order=label_order,
        mats_counts=mats_counts,
        out_path=out2,
        title="Answer quality distribution (%) by Pipeline and K",
        normalize=True,
    )

    out3 = os.path.join(args.out_dir, "avg_quality_score_heatmap.png")
    _plot_avg_score_heatmap(
        pipelines=pipelines,
        ks=ks,
        label_order=label_order,
        counts=counts,
        out_path=out3,
        title="Average answer quality score (Correct=1, Partial=0.5, Incorrect=0, Unknown=0; Unknown filled if missing)",
    )

    # Note: by_query_type plots are based on labeled rows only (without filling),
    # because we don't know the expected count per query_type without extra metadata.
    by_qtype = _group_counts_by_qtype(rows, label_order, default_label)
    if len(by_qtype) > 1:
        for qt, qt_counts in by_qtype.items():
            qt_mats = _prepare_matrix(pipelines, ks, label_order, qt_counts)
            out_qt = os.path.join(args.out_dir, f"stacked_percent_{_safe_name(qt)}.png")
            _plot_stacked_bars(
                pipelines=pipelines,
                ks=ks,
                label_order=label_order,
                mats_counts=qt_mats,
                out_path=out_qt,
                title=f"Answer quality distribution (%) by Pipeline and K — query_type={qt}",
                normalize=True,
            )

    print("Saved plots to:", os.path.abspath(args.out_dir))
    print("Pipelines:", pipelines)
    print("Ks:", ks)
    print("Query types:", qtypes)
    print("Expected N per (pipeline,K):", args.expected_n_per_config)


if __name__ == "__main__":
    main()
