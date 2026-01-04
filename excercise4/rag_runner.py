"""
Exercise4 RAG CLI Runner

Unified RAG query runner with automatic routing to appropriate retrieval strategies:
- Evolution queries (temporal comparison with early/late windows)
- Time-aware queries (soft decay or hard filtering based on temporal expressions)
- Baseline queries (standard retrieval without temporal features)

Usage examples:
    # Single query with automatic evolution detection
    python rag_runner.py --query "How has climate policy changed over time?"
    
    # Batch queries from JSON file
    python rag_runner.py --queries_json queries/temporal_queries.json --k 5 10
    
    # Disable evolution detection (force timeaware/baseline)
    python rag_runner.py --query "What is climate change?" --no-evolution --k 10
    
    # Disable all temporal features (baseline only)
    python rag_runner.py --query "What is climate change?" --no-timeaware --no-evolution
    
    # Custom evolution window and output location
    python rag_runner.py --queries_json queries/temporal_queries.json --window_months 12 --out_subdir experiment1
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
    
    Automatic query routing:
    - Evolution queries → get_topk_evolution() with evolution LLM prompt
    - Time-aware queries → get_topk_timeaware() with standard prompt
    - Baseline queries → get_topk() with standard prompt
    
    Results are saved as JSON with retrieval metadata and LLM answers.
    """
    parser = argparse.ArgumentParser(
        description="Exercise4 RAG runner with automatic query routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", help="Run a single query string.")
    mode.add_argument(
        "--queries_json", 
        help="Load queries from a JSON file grouped by temporal buckets "
             "(e.g., point_in_time, recency, explicit_range, comparison, evolution)."
    )

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
        help="Disable time-aware retrieval (baseline mode).",
    )
    parser.set_defaults(timeaware=True)

    # Evolution detection toggle (default ON)
    parser.add_argument(
        "--no-evolution",
        dest="enable_evolution",
        action="store_false",
        help="Disable automatic evolution query detection and routing.",
    )
    parser.set_defaults(enable_evolution=True)
    
    # Evolution window size
    parser.add_argument(
        "--window_months",
        type=int,
        default=8,
        help="Window size in months for evolution queries (default: 8).",
    )

    # Query group for single query mode
    parser.add_argument(
        "--query_group",
        default="single",
        help="Query group/bucket name for single query mode (default: 'single').",
    )

    # LLM
    parser.add_argument("--llm_model", default="gpt-4o-mini", help="OpenAI model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")

    # Output control
    parser.add_argument(
        "--out_root", 
        default=str(OUTPUTS_DIR / "rag_runs"), 
        help="Root output directory."
    )
    parser.add_argument(
        "--out_subdir", 
        default=None, 
        help="Optional subfolder under out_root."
    )
    parser.add_argument(
        "--out_name", 
        default=None, 
        help="Optional exact output filename (json)."
    )

    # Console verbosity
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")

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
            enable_evolution=args.enable_evolution,
            window_months=args.window_months,
        )
        mode_tag = "timeaware" if args.timeaware else "baseline"
        if args.enable_evolution:
            mode_tag += "_evo"
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
            enable_evolution=args.enable_evolution,
            window_months=args.window_months,
        )
        base = os.path.splitext(os.path.basename(args.queries_json))[0]
        mode_tag = "timeaware" if args.timeaware else "baseline"
        if args.enable_evolution:
            mode_tag += "_evo"
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
