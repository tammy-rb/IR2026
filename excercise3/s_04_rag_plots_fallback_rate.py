# rag_plots.py
"""
Plotting utilities for Exercise 3 (RAG evaluation).

This script is meant to generate all plots/figures for the report from the saved
RAG run JSON files (e.g., rag_*_4pipelines_k3-5-10_*.json).

We start with Plot #1:
(1) Heatmap of "fallback rate" = % of answers that equal the enforced fallback:
    "I don't know based on the retrieved chunks."

Usage example (later, after you add more plots):
    python rag_plots.py --rag_json outputs/rag_runs/rag_given_queries_4pipelines_k3-5-10_YYYYMMDD_HHMMSS.json --out_dir outputs/plots --plot fallback_heatmap
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend (no GUI needed)
import matplotlib.pyplot as plt
from datetime import datetime

# seaborn is optional, but recommended for nicer heatmaps
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    sns = None
    _HAS_SEABORN = False


FALLBACK_TEXT = "I don't know based on the retrieved chunks."


# -----------------------------
# I/O helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_rag_results(rag_json_path: str) -> pd.DataFrame:
    """
    Load a RAG run JSON (list of dicts) into a pandas DataFrame.

    Expected fields per row (based on your runner):
      - query_type
      - pipeline: {chunking, representation}
      - k
      - answer
      - query
      - references

    Returns:
      DataFrame with normalized columns:
        query_type, chunking, representation, pipeline, k, answer, is_fallback
    """
    with open(rag_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {rag_json_path}, got: {type(data)}")

    rows = []
    for r in data:
        pipe = r.get("pipeline", {}) or {}
        chunking = pipe.get("chunking")
        representation = pipe.get("representation")
        k = r.get("k")
        answer = (r.get("answer") or "").strip()
        query_type = r.get("query_type")
        query = r.get("query")

        if chunking is None or representation is None:
            # fallback: allow "pipeline" to be a string if you ever change format
            p = r.get("pipeline")
            if isinstance(p, str) and "_" in p:
                chunking, representation = p.split("_", 1)

        pipeline_name = f"{chunking}_{representation}"

        rows.append({
            "query_type": query_type,
            "chunking": chunking,
            "representation": representation,
            "pipeline": pipeline_name,
            "k": int(k) if k is not None else None,
            "query": query,
            "answer": answer,
            "is_fallback": (answer == FALLBACK_TEXT),
        })

    df = pd.DataFrame(rows)
    # Basic sanity checks
    if df.empty:
        raise ValueError(f"No rows loaded from {rag_json_path}")

    missing_cols = [c for c in ["pipeline", "k", "answer", "is_fallback"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns after normalization: {missing_cols}")

    return df


# -----------------------------
# Plot #1: fallback heatmap
# -----------------------------
def compute_fallback_rate_table(
    df: pd.DataFrame,
    by_query_type: bool = False,
) -> pd.DataFrame:
    """
    Compute fallback rate table to feed a heatmap.

    If by_query_type=False:
      index = pipeline, columns = k, values = fallback_rate (%)

    If by_query_type=True:
      returns a "long" table with columns:
        query_type, pipeline, k, fallback_rate
      (useful if you want separate heatmaps per query type)

    fallback_rate = 100 * mean(is_fallback)
    """
    if by_query_type:
        grp = df.groupby(["query_type", "pipeline", "k"], dropna=False)["is_fallback"].mean().reset_index()
        grp["fallback_rate"] = 100.0 * grp["is_fallback"]
        return grp[["query_type", "pipeline", "k", "fallback_rate"]]

    pivot = (
        df.groupby(["pipeline", "k"], dropna=False)["is_fallback"]
          .mean()
          .mul(100.0)
          .reset_index(name="fallback_rate")
          .pivot(index="pipeline", columns="k", values="fallback_rate")
          .sort_index(axis=0)
    )
    # Sort K columns numerically if possible
    try:
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    except Exception:
        pass
    return pivot


def plot_fallback_heatmap(
    df: pd.DataFrame,
    out_path: Optional[str] = None,
    title: str = "Fallback rate (%) by pipeline and K",
    annotate: bool = True,
    figsize: tuple[int, int] = (8, 4),
) -> plt.Figure:
    """
    Plot a heatmap of fallback rate (%), where fallback means the model returned:
      "I don't know based on the retrieved chunks."

    Saves to out_path if provided; returns the matplotlib Figure.
    """
    table = compute_fallback_rate_table(df, by_query_type=False)

    fig, ax = plt.subplots(figsize=figsize)

    if _HAS_SEABORN:
        sns.heatmap(
            table,
            ax=ax,
            annot=annotate,
            fmt=".1f",
            cbar_kws={"label": "Fallback rate (%)"},
        )
    else:
        # Minimal matplotlib fallback (no seaborn)
        im = ax.imshow(table.values, aspect="auto")
        ax.figure.colorbar(im, ax=ax, label="Fallback rate (%)")

        if annotate:
            for i in range(table.shape[0]):
                for j in range(table.shape[1]):
                    val = table.values[i, j]
                    if pd.notna(val):
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center")

        ax.set_yticks(range(table.shape[0]))
        ax.set_yticklabels(table.index.tolist())
        ax.set_xticks(range(table.shape[1]))
        ax.set_xticklabels(table.columns.tolist())

    ax.set_title(title)
    ax.set_xlabel("K (top retrieved chunks)")
    ax.set_ylabel("Pipeline")

    fig.tight_layout()

    if out_path:
        ensure_dir(os.path.dirname(out_path) or ".")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig


# -----------------------------
# CLI (we'll extend later)
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_json", required=True, help="Path to rag_*.json output from RAG_llm_runner.py")
    parser.add_argument("--out_dir", default="outputs/plots", help="Directory to save plots")
    parser.add_argument("--plot", default="fallback_heatmap", choices=["fallback_heatmap"], help="Which plot to generate")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    df = load_rag_results(args.rag_json)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.plot == "fallback_heatmap":
        out_path = os.path.join(args.out_dir, f"fallback_rate_heatmap_{timestamp}.png")
        plot_fallback_heatmap(df, out_path=out_path)
        print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
