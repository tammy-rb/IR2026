"""
Exercise4 Temporal RAG CLI Runner

This script runs RAG queries with time-aware retrieval using various chunking and representation strategies.

Usage examples:
    # Single query with default settings
    python s_03_RAG_llm_runner.py --query "What were the main topics discussed in January 2023?"
    
    # Batch queries from JSON file with custom k values and pipelines
    python s_03_RAG_llm_runner.py --queries_json queries/temporal_queries.json --k 5 10 --pipelines fixed/bm25 semantic/dense
    
    # Disable time-aware retrieval (baseline mode)
    python s_03_RAG_llm_runner.py --query "What is climate change?" --no-timeaware --k 10
    
    # Custom output location with specific model
    python s_03_RAG_llm_runner.py --queries_json queries/temporal_queries.json --llm_model gpt-4o --out_subdir experiment1 --quiet
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

from dotenv import load_dotenv
load_dotenv()

from paths import EXERCISE4_DIR, OUTPUTS_DIR

BASE_DIR = str(EXERCISE4_DIR)

# Ensure the exercise4 root is importable so `RAG_retriever` is a real package.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from RAG_retriever.RAG_retriever import RAGRetriever
from LLM.LLM_client import LLMClient, LLMConfig
from LLM.utils.io_utils import (
    load_temporal_queries_json,
    detect_query_schema,
    load_queries_json,
    build_output_path,
    save_json,
)
from LLM.utils.arg_utils import parse_ks, default_pipelines, parse_pipelines
from LLM.runners.single_runner import run_single_query
from LLM.runners.batch_runner import run_batch_queries


def main() -> None:
    """
    Main entry point for the RAG CLI runner.
    
    Supports two modes:
    1. Single query mode (--query): Run one query and display results
    2. Batch mode (--queries_json): Process multiple queries from a JSON file
    
    Results are saved as JSON with retrieval metadata, time-aware planning info, and LLM answers.
    """
    parser = argparse.ArgumentParser(description="Exercise4 Temporal RAG runner")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", help="Run a single query string.")
    mode.add_argument("--queries_json", help="Load queries from a JSON file grouped by temporal buckets (e.g., point_in_time, recency, explicit_range, comparison, evolution).")

    # Ks: allow 1 or many
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[5],
        help="One or more K values. Example: --k 5   OR   --k 3 5 10",
    )

    # Pipelines: optional override
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=None,
        help="Optional list like: fixed/bm25 semantic/bm25 fixed/dense semantic/dense",
    )

    # Time-aware toggle (default ON)
    parser.add_argument(
        "--no-timeaware",
        dest="timeaware",
        action="store_false",
        help="Disable time-aware retrieval (baseline).",
    )
    parser.set_defaults(timeaware=True)

    # Query group for single query mode
    parser.add_argument(
        "--query_group",
        default="single",
        help="Query group/bucket name for single query mode (default: 'single').",
    )

    # LLM
    parser.add_argument("--llm_model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)

    # Output control
    parser.add_argument("--out_root", default=str(OUTPUTS_DIR / "rag_runs"), help="Root output dir (relative to project).")
    parser.add_argument("--out_subdir", default=None, help="Optional subfolder under out_root.")
    parser.add_argument("--out_name", default=None, help="Optional exact output filename (json).")

    # Console verbosity
    parser.add_argument("--quiet", action="store_true", help="Less console output.")

    args = parser.parse_args()

    ks = parse_ks(args.k)
    pipelines = parse_pipelines(args.pipelines) if args.pipelines is not None else default_pipelines()

    retriever = RAGRetriever()
    llm = LLMClient(LLMConfig(model=args.llm_model, temperature=args.temperature))

    if args.query:
        results = run_single_query(
            retriever=retriever,
            llm=llm,
            query=args.query,
            pipelines=pipelines,
            ks=ks,
            timeaware=args.timeaware,
            print_console=(not args.quiet),
            query_group=args.query_group,
        )
        mode_tag = "timeaware" if args.timeaware else "baseline"
        tag = f"cli_single_{mode_tag}"
    else:
        # Auto-detect schema and load accordingly
        schema = detect_query_schema(args.queries_json)
        
        if schema == "temporal":
            queries = load_temporal_queries_json(args.queries_json)
        elif schema == "legacy":
            # Fall back to legacy loader for backward compatibility
            queries = load_queries_json(args.queries_json)
        else:
            raise ValueError(
                f"Unknown query schema in {args.queries_json}. "
                "Expected temporal buckets (point_in_time, recency, etc.) or legacy (factual, conceptual)."
            )
        
        results = run_batch_queries(
            retriever=retriever,
            llm=llm,
            queries_by_group=queries,
            pipelines=pipelines,
            ks=ks,
            timeaware=args.timeaware,
            log_progress=(not args.quiet),
        )
        base = os.path.splitext(os.path.basename(args.queries_json))[0]
        mode_tag = "timeaware" if args.timeaware else "baseline"
        tag = f"batch_{base}_{mode_tag}"

    out_path = build_output_path(
        base_dir=BASE_DIR,
        out_root=args.out_root,
        subdir=args.out_subdir,
        filename=args.out_name,
        tag=tag,
    )
    save_json(out_path, results)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
