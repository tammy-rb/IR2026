#!/usr/bin/env python3
"""
qdrant_to_bertopic_fixed20.py

Train a 20-topic BERTopic model from an existing Qdrant collection that already
stores (text, embedding vector, metadata). The script scrolls all points (or a
filtered subset), trains BERTopic on your embeddings, and exports artifacts for:

1) Sanity/quality inspection (diagnostics.json + optional HTML plots)
2) Stage A (topic discovery): 20 topics with labels/keywords
3) Stage B (time/dominance): per-document topic assignments (+ probabilities)

Notes:
- Topics are always reduced (best effort) to TARGET_NR_TOPICS (=20).
- topics_over_time.csv uses fixed 2-week bins without overlap (good for sanity),
  and does NOT replace the required sliding-window analysis in your assignment.

Outputs (written under --out-dir):
- bertopic_model/         : pickled BERTopic model (reloadable)
- topics_info.csv         : topic table (ids, sizes, labels)
- topics_20.json          : 20 topics (label + top keywords), excludes topic=-1
- docs_topics.csv         : per-document topic assignment + key payload metadata
- docs_topic_probs.npy    : full topic-probability distribution per document
- topic_embeddings.npy    : per-topic vectors (if available)
- topics_over_time.csv    : 2-week time-binned topic frequencies (if timestamps exist)
- viz_*.html              : optional interactive plots (when supported)
- diagnostics.json        : run config + outlier rate + topic size stats (before/after reduce)

Run example:
    python qdrant_to_bertopic_fixed20.py \
        --qdrant-host localhost --qdrant-port 6333 \
        --collections bbc_news_chunks chunks_openai_semantic_large \
        --filter-corpus british \
        --out-dir artifacts/bertopic_uk
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from umap import UMAP
import hdbscan


# -----------------------------
# Constants
# -----------------------------
TARGET_NR_TOPICS = 20


# -----------------------------
# Small utilities
# -----------------------------
def now_iso() -> str:
    """Current UTC time (ISO-8601). Used in run metadata and diagnostics."""
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: Path) -> None:
    """Create directory p (and parents) if missing."""
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    """Write JSON to disk (UTF-8), pretty-printed for easy diffs between runs."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_int(x: Any) -> Optional[int]:
    """Best-effort int conversion. Returns None for missing/invalid values."""
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def two_week_bin(ts: int) -> str:
    """
    Convert unix timestamp (seconds) into a stable 2-week bin label.

    Implementation: epoch-based bins (floor(ts / 14 days)).
    Return value: ISO date of the bin start (UTC).
    Used only for quick aggregation/sanity checks (non-overlapping windows).
    """
    seconds_14d = 14 * 24 * 60 * 60
    bin_idx = ts // seconds_14d
    bin_start_ts = bin_idx * seconds_14d
    dt = datetime.fromtimestamp(bin_start_ts, tz=timezone.utc).date()
    return dt.isoformat()


@dataclass
class PointRecord:
    """One Qdrant point we train on: id + text + embedding + payload metadata."""
    point_id: str
    text: str
    vector: List[float]
    payload: Dict[str, Any]


# -----------------------------
# Qdrant extraction
# -----------------------------
def build_filter(
    filter_corpus: Optional[str],
    filter_doc_id_prefix: Optional[str],
) -> Optional[qmodels.Filter]:
    """
    Build an optional Qdrant filter.

    - filter_corpus: exact match on payload['corpus'] (recommended for official runs)
    - filter_doc_id_prefix: MatchText on payload['doc_id'] (substring-like; mainly for debugging)
    """
    must: List[qmodels.FieldCondition] = []

    if filter_corpus:
        must.append(
            qmodels.FieldCondition(
                key="corpus",
                match=qmodels.MatchValue(value=filter_corpus),
            )
        )

    if filter_doc_id_prefix:
        # MatchText is substring-like; strict prefix requires a dedicated field or different schema.
        must.append(
            qmodels.FieldCondition(
                key="doc_id",
                match=qmodels.MatchText(text=filter_doc_id_prefix),
            )
        )

    if not must:
        return None

    return qmodels.Filter(must=must)


def scroll_with_retry(client: QdrantClient, *, max_retries: int = 6, **kwargs: Any) -> Tuple[List[Any], Any]:
    """Scroll with retry and backoff to survive transient disconnects."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.scroll(**kwargs)
        except Exception as exc:  # noqa: BLE001 keep broad to retry transport errors
            last_exc = exc
            sleep_s = min(20.0, 1.5 * attempt)
            print(f"[WARN] scroll failed (attempt {attempt}/{max_retries}): {exc}. sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("scroll_with_retry exhausted retries without raising an exception")


def scroll_all_points(
    client: QdrantClient,
    collection: str,
    *,
    batch_size: int = 128,
    qdrant_filter: Optional[qmodels.Filter] = None,
    with_vectors: bool = True,
    with_payload: bool = True,
    max_points: Optional[int] = None,
) -> List[PointRecord]:
    """
    Scroll through all points in a collection (or filtered subset) and return
    PointRecord objects with text + vectors.

    batch_size: number of points returned per scroll call (paging size)
    offset: server-side cursor token returned by Qdrant, used to continue paging
    max_points: optional cap for quick experiments/debugging
    """
    records: List[PointRecord] = []
    offset = None
    total = 0

    while True:
        points, offset = scroll_with_retry(
            client,
            collection_name=collection,
            scroll_filter=qdrant_filter,
            limit=batch_size,
            offset=offset,
            with_vectors=with_vectors,
            with_payload=with_payload,
        )

        if not points:
            break

        for p in points:
            pid = str(p.id)
            payload = p.payload or {}

            text = payload.get("text", None)
            if not isinstance(text, str) or not text.strip():
                continue

            if not with_vectors:
                continue

            # Qdrant may return a single vector or named vectors (dict); handle both.
            if isinstance(p.vector, dict):
                vec = next(iter(p.vector.values()))
            else:
                vec = p.vector

            if vec is None:
                continue

            records.append(PointRecord(point_id=pid, text=text, vector=list(vec), payload=payload))
            total += 1
            if max_points is not None and total >= max_points:
                return records

        # offset becomes None when there are no more points (finished paging).
        if offset is None:
            break

    return records


# -----------------------------
# Multi-collection aggregation
# -----------------------------
def scroll_many_collections(
    client: QdrantClient,
    collections: List[str],
    *,
    batch_size: int,
    filter_corpus: Optional[str],
    filter_doc_id_prefix: Optional[str],
    max_points: Optional[int],
) -> List[PointRecord]:
    """Scroll multiple collections and merge results into one list.

    Applies the same optional filter to each collection so callers can target
    subsets uniformly (or skip filtering entirely).
    """
    all_records: List[PointRecord] = []
    remaining = max_points

    for col in collections:
        qf = build_filter(filter_corpus, filter_doc_id_prefix)

        col_cap = remaining
        recs = scroll_all_points(
            client,
            col,
            batch_size=batch_size,
            qdrant_filter=qf,
            with_vectors=True,
            with_payload=True,
            max_points=col_cap,
        )

        all_records.extend(recs)

        if remaining is not None:
            remaining -= len(recs)
            if remaining <= 0:
                break

    return all_records


# -----------------------------
# BERTopic helpers + diagnostics
# -----------------------------
def topic_keywords(topic_model: BERTopic, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
    """Return top_n (term, weight) pairs for a topic; empty list if unavailable."""
    pairs = topic_model.get_topic(topic_id)
    if not pairs:
        return []
    return [(w, float(s)) for (w, s) in pairs[:top_n]]


def compute_diagnostics(
    *,
    topics: List[int],
    probs: Optional[np.ndarray],
    topic_model: Optional[BERTopic],
    stage: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute a compact "quality snapshot" for one stage of the pipeline.

    What it captures:
    - Outlier rate (topic=-1): how many docs HDBSCAN rejected as noise
    - Topic count and size distribution (excluding outliers)
    - Probability sanity (mean/quantiles of max topic probability per doc)
    - Shape of topic embeddings (if available)
    - Run parameters used (UMAP/HDBSCAN/seed) for reproducibility
    """
    n = len(topics)
    outliers = sum(1 for t in topics if t == -1)
    outlier_rate = (outliers / n) if n else None

    unique_topics = sorted(set(topics))
    n_topics_including_outliers = len(unique_topics)
    n_topics_excluding_outliers = len([t for t in unique_topics if t != -1])

    # Topic size distribution (excluding -1)
    sizes: Dict[int, int] = {}
    for t in topics:
        if t == -1:
            continue
        sizes[t] = sizes.get(t, 0) + 1

    size_values = list(sizes.values())
    size_values_sorted = sorted(size_values)

    def pct(xs: List[int], p: float) -> Optional[float]:
        if not xs:
            return None
        k = int(round((len(xs) - 1) * p))
        return float(xs[k])

    size_stats = {
        "min": int(min(size_values)) if size_values else None,
        "max": int(max(size_values)) if size_values else None,
        "mean": float(np.mean(size_values)) if size_values else None,
        "median": float(np.median(size_values)) if size_values else None,
        "p10": pct(size_values_sorted, 0.10),
        "p25": pct(size_values_sorted, 0.25),
        "p75": pct(size_values_sorted, 0.75),
        "p90": pct(size_values_sorted, 0.90),
    }

    # Heuristics to flag fragmentation
    tiny_threshold = 10
    small_threshold = 30
    tiny_topics = sum(1 for s in size_values if s < tiny_threshold)
    small_topics = sum(1 for s in size_values if s < small_threshold)

    # Probability sanity
    prob_stats = None
    if probs is not None and probs.size > 0:
        maxp = np.max(probs, axis=1)
        prob_stats = {
            "max_prob_mean": float(np.mean(maxp)),
            "max_prob_median": float(np.median(maxp)),
            "max_prob_p10": float(np.quantile(maxp, 0.10)),
            "max_prob_p90": float(np.quantile(maxp, 0.90)),
        }

    # Topic embeddings availability
    topic_emb = getattr(topic_model, "topic_embeddings_", None) if topic_model is not None else None
    topic_emb_shape = list(np.asarray(topic_emb).shape) if topic_emb is not None else None

    return {
        "stage": stage,
        "created_utc": now_iso(),
        "num_docs": n,
        "outliers": {"count": outliers, "rate": outlier_rate},
        "topics": {
            "unique_including_outliers": n_topics_including_outliers,
            "unique_excluding_outliers": n_topics_excluding_outliers,
            "size_stats_excluding_outliers": size_stats,
            "tiny_topics_lt_10": tiny_topics,
            "small_topics_lt_30": small_topics,
        },
        "probabilities": prob_stats,
        "topic_embeddings_shape": topic_emb_shape,
        "params": params,
    }


def print_diagnostics_summary(diagnostics: Dict[str, Any]) -> None:
    """
    Print a short, human-readable summary so you can decide quickly
    whether to adjust parameters before inspecting topics.
    """
    after = diagnostics.get("after_reduce", {})
    topics_meta = after.get("topics", {})
    out_meta = after.get("outliers", {})
    size_stats = topics_meta.get("size_stats_excluding_outliers", {})

    num_docs = after.get("num_docs")
    out_cnt = out_meta.get("count")
    out_rate = out_meta.get("rate")
    uniq_topics = topics_meta.get("unique_excluding_outliers")
    tiny_topics = topics_meta.get("tiny_topics_lt_10")
    small_topics = topics_meta.get("small_topics_lt_30")

    def fmt_rate(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        return f"{value * 100:.1f}%"

    print("[INFO] Diagnostics summary (after_reduce):")
    print(f"[INFO] Documents: {num_docs if num_docs is not None else 'n/a'}")
    print(f"[INFO] Outliers: {out_cnt if out_cnt is not None else 'n/a'} ({fmt_rate(out_rate)})")
    print(f"[INFO] Topics (excluding outliers): {uniq_topics if uniq_topics is not None else 'n/a'}")
    if size_stats:
        print(
            "[INFO] Topic sizes (min/median/max): "
            f"{size_stats.get('min', 'n/a')} / {size_stats.get('median', 'n/a')} / {size_stats.get('max', 'n/a')}"
        )
    if tiny_topics is not None or small_topics is not None:
        print(
            f"[INFO] Topics <10 docs: {tiny_topics if tiny_topics is not None else 'n/a'} | "
            f"Topics <30 docs: {small_topics if small_topics is not None else 'n/a'}"
        )


def train_bertopic_fixed20(
    docs: List[str],
    embeddings: np.ndarray,
    *,
    seed: int = 42,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    umap_min_dist: float = 0.0,
    hdb_min_cluster_size: int = 30,
    hdb_min_samples: Optional[int] = None,
) -> Tuple[BERTopic, List[int], Optional[np.ndarray], Dict[str, Any]]:
    """
    Train BERTopic using precomputed embeddings and reduce the result to 20 topics.

    Returns:
      - topic_model: trained (and reduced) BERTopic instance
      - topics: per-document topic id (after reduction)
      - probs: per-document topic probability distribution (if enabled)
      - diagnostics: metrics before/after reduction for quick model sanity checks
    """
    params = {
        "TARGET_NR_TOPICS": TARGET_NR_TOPICS,
        "seed": seed,
        "umap": {
            "n_neighbors": umap_n_neighbors,
            "n_components": umap_n_components,
            "min_dist": umap_min_dist,
            "metric": "cosine",
        },
        "hdbscan": {
            "min_cluster_size": hdb_min_cluster_size,
            "min_samples": hdb_min_samples,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True,
        },
        "bertopic": {"calculate_probabilities": True},
    }

    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=seed,
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=hdb_min_cluster_size,
        min_samples=hdb_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=CountVectorizer(stop_words="english"),
        calculate_probabilities=True,
        verbose=True,
    )

    # Train with "natural" topic structure first
    topics_0, probs_0 = topic_model.fit_transform(docs, embeddings)
    diag_before = compute_diagnostics(
        topics=topics_0,
        probs=probs_0,
        topic_model=topic_model,
        stage="before_reduce",
        params=params,
    )

    # Reduce to fixed 20 anchors and recompute assignments consistently
    topic_model.reduce_topics(docs, nr_topics=TARGET_NR_TOPICS)
    topics_1, probs_1 = topic_model.transform(docs, embeddings)
    diag_after = compute_diagnostics(
        topics=topics_1,
        probs=probs_1,
        topic_model=topic_model,
        stage="after_reduce",
        params=params,
    )

    diagnostics_bundle = {"before_reduce": diag_before, "after_reduce": diag_after}
    return topic_model, list(map(int, topics_1)), probs_1, diagnostics_bundle


# -----------------------------
# Export artifacts
# -----------------------------
def save_outputs(
    out_dir: Path,
    *,
    topic_model: BERTopic,
    records: List[PointRecord],
    topics: List[int],
    probs: Optional[np.ndarray],
    diagnostics: Dict[str, Any],
    input_collections: List[str],
    filter_corpus_used: Optional[str],
) -> None:
    """
    Write all outputs to out_dir.

    Recommended inspection order:
      1) diagnostics.json (sanity/parameter tuning)
      2) viz_*.html + topics_20.json (topic interpretability)
      3) docs_topics.csv (+ docs_topic_probs.npy) (time series / dominance)

    input_collections and filter_corpus_used are persisted to help trace the
    training data origin when multiple sources are merged.
    """
    ensure_dir(out_dir)

    # Save diagnostics first to support "tune-then-inspect" workflow
    save_json(out_dir / "diagnostics.json", diagnostics)

    # 1) Model (reloadable)
    model_dir = out_dir / "bertopic_model"
    topic_model.save(str(model_dir), serialization="pickle")

    # 2) Per-topic metadata
    info_df = topic_model.get_topic_info()
    info_df.to_csv(out_dir / "topics_info.csv", index=False, encoding="utf-8")

    # 3) Compact 20-topic JSON (exclude outliers)
    topics_list = []
    for _, row in info_df.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        label = str(row.get("Name", ""))
        kws = topic_keywords(topic_model, tid, top_n=12)
        topics_list.append(
            {
                "topic_id": tid,
                "label": label,
                "size": int(row.get("Count", 0)),
                "keywords": [{"term": t, "weight": w} for (t, w) in kws],
            }
        )

    save_json(
        out_dir / "topics_20.json",
        {
            "created_utc": now_iso(),
            "target_nr_topics": TARGET_NR_TOPICS,
            "topics": topics_list,
            "notes": "Topic=-1 are outliers and excluded from this list.",
        },
    )

    # 4) Per-document assignments + payload fields
    rows = []
    for i, r in enumerate(records):
        payload = r.payload or {}
        ts = safe_int(payload.get("doc_timestamp"))

        # Confidence proxy: max probability mass among topics for this doc
        prob = float(np.max(probs[i])) if probs is not None else None

        rows.append(
            {
                "point_id": r.point_id,
                "chunk_uid": payload.get("chunk_uid"),
                "doc_id": payload.get("doc_id"),
                "corpus": payload.get("corpus"),
                "source_path": payload.get("source_path"),
                "chunking_method": payload.get("chunking_method"),
                "chunk_index": payload.get("chunk_index"),
                "doc_date_iso": payload.get("doc_date_iso"),
                "doc_timestamp": ts,
                "two_week_bin": two_week_bin(ts) if ts is not None else None,
                "topic": int(topics[i]),
                "topic_prob": prob,
                "text_preview": (r.text[:400] + "...") if len(r.text) > 400 else r.text,
            }
        )

    doc_df = pd.DataFrame(rows)
    doc_df.to_csv(out_dir / "docs_topics.csv", index=False, encoding="utf-8")

    # 4b) Full probability matrix (for soft dominance scoring)
    if probs is not None:
        np.save(out_dir / "docs_topic_probs.npy", probs)

    # 5) Topic embeddings (useful for similarity analysis)
    topic_emb = getattr(topic_model, "topic_embeddings_", None)
    if topic_emb is not None:
        np.save(out_dir / "topic_embeddings.npy", np.asarray(topic_emb))

    # 6) Quick time aggregation (non-overlapping 2-week bins) -- disabled for stability

    # 7) Optional HTML visualizations
    try:
        fig = topic_model.visualize_topics()
        fig.write_html(str(out_dir / "viz_topics.html"))
    except Exception as e:
        save_json(out_dir / "viz_topics_error.json", {"error": str(e)})

    try:
        fig = topic_model.visualize_barchart(top_n_topics=min(20, len(topics_list)))
        fig.write_html(str(out_dir / "viz_barchart.html"))
    except Exception as e:
        save_json(out_dir / "viz_barchart_error.json", {"error": str(e)})

    try:
        fig = topic_model.visualize_heatmap()
        fig.write_html(str(out_dir / "viz_heatmap.html"))
    except Exception as e:
        save_json(out_dir / "viz_heatmap_error.json", {"error": str(e)})

    # 8) Run index (what was produced)
    save_json(
        out_dir / "run_meta.json",
        {
            "created_utc": now_iso(),
            "num_docs": len(records),
            "target_nr_topics": TARGET_NR_TOPICS,
            "model_saved_to": str(model_dir),
            "input_collections": input_collections,
            "filter_corpus": filter_corpus_used,
            "files": {
                "diagnostics_json": "diagnostics.json",
                "topics_info_csv": "topics_info.csv",
                "topics_20_json": "topics_20.json",
                "docs_topics_csv": "docs_topics.csv",
                "docs_topic_probs_npy": "docs_topic_probs.npy" if probs is not None else None,
                "topic_embeddings_npy": "topic_embeddings.npy" if topic_emb is not None else None,
                "topics_over_time_csv": "topics_over_time.csv" if doc_df["two_week_bin"].notna().any() else None,
                "viz_topics_html": "viz_topics.html",
                "viz_barchart_html": "viz_barchart.html",
                "viz_heatmap_html": "viz_heatmap.html",
            },
        },
    )


# -----------------------------
# CLI / main
# -----------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for extraction + training + export."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    ap.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    ap.add_argument(
        "--collections",
        required=True,
        nargs="+",
        help="One or more Qdrant collections to merge before training",
    )
    ap.add_argument("--out-dir", required=True, help="Output directory for artifacts")

    ap.add_argument(
        "--filter-corpus",
        default=None,
        help="Filter payload['corpus'] (applies to every collection when provided)",
    )
    ap.add_argument(
        "--filter-doc-id-prefix",
        default=None,
        help="Optional MatchText on doc_id (applies to every collection when provided)",
    )

    ap.add_argument("--batch-size", type=int, default=4, help="Scroll page size (does not limit total points)")
    ap.add_argument("--max-points", type=int, default=5000, help="Optional cap for quick debugging runs")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--umap-n-neighbors", type=int, default=15)
    ap.add_argument("--umap-n-components", type=int, default=5)
    ap.add_argument("--umap-min-dist", type=float, default=0.0)
    ap.add_argument("--hdb-min-cluster-size", type=int, default=30)
    ap.add_argument("--hdb-min-samples", type=int, default=None)

    return ap.parse_args()


def main() -> None:
    """Entry point: pull data from Qdrant, train BERTopic, and export artifacts."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.max_points is None:
        print("[WARN] max_points is None. On 8GB RAM this may crash. Consider --max-points 20000 or smaller batches.")

    client = QdrantClient(
        url=f"http://{args.qdrant_host}:{args.qdrant_port}",
        timeout=180,
    )

    print(f"[INFO] Scrolling Qdrant collections={args.collections} ...")
    records = scroll_many_collections(
        client,
        args.collections,
        batch_size=args.batch_size,
        filter_corpus=args.filter_corpus,
        filter_doc_id_prefix=args.filter_doc_id_prefix,
        max_points=args.max_points,
    )

    if not records:
        raise RuntimeError("No records found (check collection/filters and ensure payload contains 'text').")

    print(f"[INFO] Loaded {len(records)} records with text+vector.")
    docs = [r.text for r in records]
    emb = np.asarray([r.vector for r in records], dtype=np.float32)

    print(f"[INFO] Training BERTopic (target topics fixed at {TARGET_NR_TOPICS}) on embeddings shape={emb.shape} ...")
    topic_model, topics, probs, diagnostics = train_bertopic_fixed20(
        docs,
        emb,
        seed=args.seed,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_n_components=args.umap_n_components,
        umap_min_dist=args.umap_min_dist,
        hdb_min_cluster_size=args.hdb_min_cluster_size,
        hdb_min_samples=args.hdb_min_samples,
    )

    print_diagnostics_summary(diagnostics)

    print("[INFO] Saving artifacts ...")
    save_outputs(
        out_dir,
        topic_model=topic_model,
        records=records,
        topics=topics,
        probs=probs,
        diagnostics=diagnostics,
        input_collections=args.collections,
        filter_corpus_used=args.filter_corpus,
    )

    print(f"[INFO] Done. Artifacts written to: {out_dir}")
    print(f"[INFO] First look at: {out_dir / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
