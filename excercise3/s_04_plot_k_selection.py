# plot_k_selection_from_labels.py
from __future__ import annotations

import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total"] = df["direct"] + df["supporting"] + df["irrelevant"]
    # if you always label exactly K chunks, total==k; but keep robust:
    denom = df["total"].replace(0, pd.NA)

    df["relevant"] = df["direct"] + df["supporting"]
    df["relevant_rate"] = (df["relevant"] / denom).astype(float)
    df["noise_rate"] = (df["irrelevant"] / denom).astype(float)

    # simple weighted evidence score (tweakable)
    df["evidence_score"] = 2 * df["direct"] + 1 * df["supporting"] - 1 * df["irrelevant"]
    return df

def plot_stacked_distribution(agg: pd.DataFrame, title: str, outpath: str) -> None:
    # agg must have index k and columns direct/supporting/irrelevant as proportions (0..1) or counts
    ks = agg.index.tolist()
    direct = agg["direct"].tolist()
    supporting = agg["supporting"].tolist()
    irrelevant = agg["irrelevant"].tolist()

    plt.figure()
    plt.bar(ks, direct, label="Directly Relevant")
    plt.bar(ks, supporting, bottom=direct, label="Supporting")
    bottom2 = [d + s for d, s in zip(direct, supporting)]
    plt.bar(ks, irrelevant, bottom=bottom2, label="Irrelevant")

    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Proportion of retrieved chunks")
    plt.xticks(ks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_lines_by_k(agg: pd.DataFrame, title: str, outpath: str) -> None:
    # agg must have index k and columns relevant_rate/noise_rate/evidence_score
    ks = agg.index.tolist()

    plt.figure()
    plt.plot(ks, agg["relevant_rate"].tolist(), marker="o", label="Relevant@K (Direct+Supporting)")
    plt.plot(ks, agg["noise_rate"].tolist(), marker="o", label="Noise@K (Irrelevant)")
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Rate")
    plt.xticks(ks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_score_by_k(agg: pd.DataFrame, title: str, outpath: str) -> None:
    ks = agg.index.tolist()
    plt.figure()
    plt.plot(ks, agg["evidence_score"].tolist(), marker="o")
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Weighted evidence score (2*Direct + Supporting - Irrelevant)")
    plt.xticks(ks)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True, help="CSV with columns: query_set,query_type,pipeline,k,direct,supporting,irrelevant")
    ap.add_argument("--outdir", default="outputs/plots", help="Where to write PNG plots")
    ap.add_argument("--pipeline", default="semantic_dense", help="Pipeline to focus on for K-selection plots")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    df = pd.read_csv(args.labels_csv)
    required = {"query_set", "query_type", "pipeline", "k", "direct", "supporting", "irrelevant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["k"] = df["k"].astype(int)
    for c in ["direct", "supporting", "irrelevant"]:
        df[c] = df[c].astype(int)

    df = add_metrics(df)

    # Focus on best pipeline (you can still plot all pipelines later)
    dff = df[df["pipeline"] == args.pipeline].copy()
    if dff.empty:
        raise ValueError(f"No rows found for pipeline={args.pipeline}")

    # ---- Plot per query_set × query_type (best pipeline)
    for (qset, qtype), g in dff.groupby(["query_set", "query_type"]):
        # aggregate across queries
        agg_counts = g.groupby("k")[["direct", "supporting", "irrelevant"]].sum().sort_index()

        # convert to proportions for stacked plot
        agg_props = agg_counts.div(agg_counts.sum(axis=1), axis=0)

        # also aggregate metrics
        agg_metrics = g.groupby("k")[["relevant_rate", "noise_rate", "evidence_score"]].mean().sort_index()

        base = f"{args.pipeline}_{qset}_{qtype}"

        plot_stacked_distribution(
            agg_props,
            title=f"Chunk label composition vs K ({args.pipeline}, {qset}, {qtype})",
            outpath=os.path.join(args.outdir, f"{base}_stacked.png"),
        )

        plot_lines_by_k(
            agg_metrics,
            title=f"Relevant vs noise vs K ({args.pipeline}, {qset}, {qtype})",
            outpath=os.path.join(args.outdir, f"{base}_rates.png"),
        )

        plot_score_by_k(
            agg_metrics,
            title=f"Weighted evidence score vs K ({args.pipeline}, {qset}, {qtype})",
            outpath=os.path.join(args.outdir, f"{base}_score.png"),
        )

    print(f"Saved plots to: {args.outdir}")

if __name__ == "__main__":
    main()
