"""
Stage 3 — Temporal Analysis Runner (Retrieval-Level Evaluation)

This script implements the **Stage 3 deliverable** of Exercise 4: Temporal RAG.

Purpose
-------
Produce a **direct comparison artifact** that demonstrates how time-aware
retrieval changes the *ranking of retrieved chunks* compared to a
baseline (time-blind) RAG system.

Importantly:
- This runner operates **only at the retrieval layer**
- It does **NOT call the LLM**
- The output is designed for **analysis and reporting**, not answering

This matches the assignment requirement:
"Create a table showing how the top-5 results change when moving from
baseline retrieval to time-aware scoring."

What This Script Does
---------------------
For each:
- query
- pipeline (chunking + representation)
- K value

the script performs **two retrieval runs**:

1. Baseline retrieval (time-blind)
   - retriever.get_topk(...)
   - ranking based purely on semantic similarity

2. Time-aware retrieval
   - retriever.get_topk_timeaware(...)
   - internally performs:
        a) candidate oversampling
        b) temporal re-ranking (recency / hard filtering)
        c) final Top-K selection

The script then stores the **Top-K final rankings from both modes**
and computes a delta summary indicating which chunks:
- stayed in the Top-K
- entered the Top-K
- were removed due to temporal reasoning

Output Schema
-------------
Each entry contains:
- query_group: Temporal bucket name
- query: The evaluated query string
- pipeline: {chunking, representation}
- k: Number of final retrieved chunks
- baseline_top: Ordered list with rank, chunk_id, score, doc_date_iso
- timeaware_top: Ordered list with rank, chunk_id, score, doc_date_iso
- delta: Membership-only change (overlap_count, entered, left)
- delta_scored: Score-aware delta with baseline_score + timeaware_score
- timeaware_mode: Temporal intent classification
- timeaware_plan: High-level retrieval strategy

Usage Examples
--------------
Single query:
    python s_03_temporal_analysis.py --query "What happened in December 2023?" --k 5

Batch from file:
    python s_03_temporal_analysis.py --queries_json queries/given_temporal_queries.json --k 5 10 --pipelines semantic/dense

Custom output:
    python s_03_temporal_analysis.py --queries_json queries/temporal_queries.json --k 5 --out_name my_analysis.json
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv()

from paths import EXERCISE4_DIR, OUTPUTS_DIR

BASE_DIR = str(EXERCISE4_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from RAG_retriever.RAG_retriever import RAGRetriever
from RAG_retriever.utils import format_ref_id
from LLM.utils.io_utils import (
    build_output_path,
    save_json,
    flatten_temporal_queries_with_groups,
)
from LLM.utils.arg_utils import parse_ks, default_pipelines, parse_pipelines


def run_temporal_comparison(
    retriever: RAGRetriever,
    queries: List[str] | List[Dict[str, Any]],
    pipelines: List[Tuple[str, str]],
    ks: List[int],
    topn: int = 5,
) -> List[Dict[str, Any]]:
    """
    Run temporal analysis comparing baseline vs time-aware retrieval.

    Args:
        retriever: RAG retriever instance
        queries: List of query strings or dicts with {"query": str, "query_group": str, ...}
        pipelines: List of (chunking, representation) tuples
        ks: List of k values (number of chunks to retrieve)
        topn: Number of top results to include in comparison (default 5)

    Returns:
        List of comparison dictionaries with baseline/timeaware results and delta analysis
    """
    results: List[Dict[str, Any]] = []

    # Normalize queries to list of dicts
    normalized_queries: List[Dict[str, Any]] = []
    for q in queries:
        if isinstance(q, str):
            normalized_queries.append({"query": q, "query_group": "unknown"})
        else:
            qdict = dict(q)
            qdict.setdefault("query_group", "unknown")
            normalized_queries.append(qdict)

    for q_item in normalized_queries:
        query = q_item["query"]
        query_group = q_item["query_group"]

        for chunking, repr_ in pipelines:
            for k in ks:
                print(f"[{query_group}] {chunking}/{repr_} | k={k} | {query[:60]}...")

                # Baseline retrieval (time-blind)
                baseline_pack = retriever.get_topk(query, chunking, repr_, k)
                baseline_refs = baseline_pack.get("refs", [])
                baseline_retrieved = baseline_pack.get("retrieved", [])

                # Time-aware retrieval
                timeaware_pack = retriever.get_topk_timeaware(query, chunking, repr_, k)
                timeaware_refs = timeaware_pack.get("refs", [])
                timeaware_retrieved = timeaware_pack.get("retrieved", [])

                # Build FULL score maps from ALL retrieved results (up to k)
                # This allows us to see scores for chunks that entered/left the top-N
                baseline_score_map: Dict[str, Any] = {}
                for i, ref in enumerate(baseline_refs):  # All k results
                    chunk_id = format_ref_id(ref)
                    score = baseline_retrieved[i].get("score") if i < len(baseline_retrieved) else None
                    baseline_score_map[chunk_id] = score

                timeaware_score_map: Dict[str, Any] = {}
                for i, ref in enumerate(timeaware_refs):  # All k results
                    chunk_id = format_ref_id(ref)
                    score = timeaware_retrieved[i].get("score") if i < len(timeaware_retrieved) else None
                    timeaware_score_map[chunk_id] = score

                # Build baseline top results (limited to topn for display)
                baseline_top: List[Dict[str, Any]] = []
                for i, ref in enumerate(baseline_refs[:topn]):
                    chunk_id = format_ref_id(ref)
                    baseline_top.append(
                        {
                            "rank": i + 1,
                            "chunk_id": chunk_id,
                            "score": baseline_score_map.get(chunk_id),
                            "doc_date_iso": ref.get("doc_date_iso"),
                        }
                    )

                # Build time-aware top results (limited to topn for display)
                timeaware_top: List[Dict[str, Any]] = []
                for i, ref in enumerate(timeaware_refs[:topn]):
                    chunk_id = format_ref_id(ref)
                    timeaware_top.append(
                        {
                            "rank": i + 1,
                            "chunk_id": chunk_id,
                            "score": timeaware_score_map.get(chunk_id),
                            "doc_date_iso": ref.get("doc_date_iso"),
                        }
                    )

                # Membership delta (ids only)
                baseline_ids = {x["chunk_id"] for x in baseline_top}
                timeaware_ids = {x["chunk_id"] for x in timeaware_top}

                overlap = baseline_ids & timeaware_ids
                entered = timeaware_ids - baseline_ids
                left = baseline_ids - timeaware_ids

                delta = {
                    "overlap_count": len(overlap),
                    "entered": sorted(list(entered)),
                    "left": sorted(list(left)),
                }

                # Score-aware delta (baseline_score + timeaware_score per chunk)
                # Now includes scores even for chunks outside top-N (if they were in top-K)
                delta_scored = {
                    "overlap": [
                        {
                            "chunk_id": cid,
                            "baseline_score": baseline_score_map.get(cid),
                            "timeaware_score": timeaware_score_map.get(cid),
                        }
                        for cid in sorted(overlap)
                    ],
                    "entered": [
                        {
                            "chunk_id": cid,
                            "baseline_score": baseline_score_map.get(cid),  # May have score if ranked N+1 to K in baseline
                            "timeaware_score": timeaware_score_map.get(cid),
                        }
                        for cid in sorted(entered)
                    ],
                    "left": [
                        {
                            "chunk_id": cid,
                            "baseline_score": baseline_score_map.get(cid),
                            "timeaware_score": timeaware_score_map.get(cid),  # May have score if ranked N+1 to K in timeaware
                        }
                        for cid in sorted(left)
                    ],
                }

                row: Dict[str, Any] = {
                    "query_group": query_group,
                    "query": query,
                    "pipeline": {"chunking": chunking, "representation": repr_},
                    "k": k,
                    "topn": topn,
                    "baseline_top": baseline_top,
                    "timeaware_top": timeaware_top,
                    "delta": delta,
                    "delta_scored": delta_scored,
                    "timeaware_mode": (timeaware_pack.get("time_info") or {}).get("mode"),
                    "timeaware_plan": timeaware_pack.get("plan"),
                }

                if "debug" in timeaware_pack:
                    row["debug"] = timeaware_pack["debug"]

                results.append(row)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Temporal Analysis (Baseline vs Time-Aware)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", nargs="+", help="One or more query strings (wrap each in quotes).")
    mode.add_argument(
        "--queries_json",
        help="Load queries from JSON file (temporal buckets: point_in_time, recency, explicit_range, comparison, evolution).",
    )

    parser.add_argument("--k", nargs="+", type=int, default=[5], help="K values for retrieval, e.g. --k 5 10")
    parser.add_argument("--topn", type=int, default=5, help="Top-N results to compare (default 5)")

    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=None,
        help="Optional pipeline list: fixed/bm25 semantic/bm25 fixed/dense semantic/dense",
    )

    parser.add_argument(
        "--out_root",
        default=str(OUTPUTS_DIR / "rag_runs"),
        help="Root output directory",
    )
    parser.add_argument(
        "--out_subdir",
        default="stage3_temporal_analysis",
        help="Subfolder under out_root",
    )
    parser.add_argument("--out_name", default=None, help="Optional exact output filename (json)")

    args = parser.parse_args()

    ks = parse_ks(args.k)
    pipelines = parse_pipelines(args.pipelines) if args.pipelines is not None else default_pipelines()

    if args.query:
        queries = [{"query": q, "query_group": "single"} for q in args.query]
        tag = "stage3_single"
    else:
        queries = flatten_temporal_queries_with_groups(args.queries_json)
        base = os.path.splitext(os.path.basename(args.queries_json))[0]
        tag = f"stage3_{base}"

    retriever = RAGRetriever()
    results = run_temporal_comparison(
        retriever=retriever,
        queries=queries,
        pipelines=pipelines,
        ks=ks,
        topn=args.topn,
    )

    out_path = build_output_path(
        base_dir=BASE_DIR,
        out_root=args.out_root,
        subdir=args.out_subdir,
        filename=args.out_name,
        tag=tag,
    )
    save_json(out_path, results)
    print(f"\n✓ Saved temporal analysis to: {out_path}")


if __name__ == "__main__":
    main()
