"""
s_03_RAG_llm_runner.py

LLM orchestration layer for a Retrieval-Augmented Generation (RAG) system.

This module:
- Loads evaluation queries
- Calls the retrieval engine to obtain top-K context
- Prompts an LLM using the retrieved context only
- Saves answers and citations for offline evaluation

This file contains NO retrieval logic.
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List
from datetime import datetime

import sys

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Ensure the exercise4 root is importable so `RAG_retriever` is a real package.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Updated import: new retriever module location
from RAG_retriever.RAG_retriever import RAGRetriever
from RAG_retriever.utils import format_ref_id


OUT_DIR = os.path.join(BASE_DIR, "outputs", "rag_runs")
DEFAULT_MODEL = "gpt-4o-mini"


def ensure_dir(path: str) -> None:
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def load_queries(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load evaluation queries from JSON.

    Args:
        path: Path to queries.json.

    Returns:
        Dictionary with 'factual' and 'conceptual' query lists.
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
        "factual": normalize(data["factual"]),
        "conceptual": normalize(data["conceptual"]),
    }


def answer_with_llm(llm: ChatOpenAI, query: str, context: str) -> str:
    """
    Generate an answer using the LLM based strictly on retrieved context.

    Args:
        llm: Initialized ChatOpenAI model.
        query: User query.
        context: Retrieved context block.

    Returns:
        LLM-generated answer string.
    """
    system = (
        "You are a RAG question-answering assistant. "
        "Answer ONLY using the provided context. "
        "If unsupported, say: "
        "\"I don't know based on the retrieved chunks.\" "
        "Always cite sources in square brackets."
    )

    user = f"Question:\n{query}\n\nContext:\n{context}\n\nAnswer:"

    msg = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    return msg.content


def main() -> None:
    """
    Run RAG evaluation over multiple pipelines and K values.

    Modes:
      - Batch mode: provide --queries_json
      - Single-query mode: provide --query
    """
    parser = argparse.ArgumentParser()

    # Accept either a file OR a single query string
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--queries_json", required=False)
    group.add_argument("--query", required=False, help="Run a single query and print the answer + chunk ids.")

    parser.add_argument("--k1", type=int, default=3)
    parser.add_argument("--k2", type=int, default=5)
    parser.add_argument("--k3", type=int, default=10)
    parser.add_argument("--llm_model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Time-aware retrieval: DEFAULT ON (use --no-timeaware to disable)
    parser.add_argument(
        "--no-timeaware",
        dest="timeaware",
        action="store_false",
        help="Disable time-aware retrieval and use baseline retrieval.",
    )
    parser.set_defaults(timeaware=True)

    parser.add_argument(
        "--out_dir",
        default=None,
        help="Optional output directory for results JSON. "
             "If not provided, uses the default outputs/rag_runs directory."
    )
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else OUT_DIR
    ensure_dir(out_dir)

    retriever = RAGRetriever()
    llm = ChatOpenAI(model=args.llm_model, temperature=args.temperature)

    ks = [args.k1, args.k2, args.k3]
    pipelines = [
        ("fixed", "bm25"),
        ("semantic", "bm25"),
        ("fixed", "dense"),
        ("semantic", "dense"),
    ]

    results = []

    if args.query is not None:
        # ---- Single-query mode (prints to console) ----
        q = {"query": args.query, "expected_source": []}
        qtype = "single"

        for chunking, repr_ in pipelines:
            for k in ks:
                print(f"[{qtype}] {chunking}/{repr_} | k={k} | query='{q['query']}'")

                if args.timeaware:
                    pack = retriever.get_topk_timeaware(q["query"], chunking, repr_, k)
                else:
                    pack = retriever.get_topk(q["query"], chunking, repr_, k)

                answer = answer_with_llm(llm, q["query"], pack["context"])

                print("\nAnswer:")
                print(answer.strip())

                print("\nChunk ids:")
                refs = pack.get("refs", [])
                retrieved = pack.get("retrieved", [])
                for i, ref in enumerate(refs):
                    score = None
                    if i < len(retrieved):
                        score = retrieved[i].get("score")
                    rid = format_ref_id(ref)
                    print(f"- {rid}" + (f" | score={score}" if score is not None else ""))

                print("\n" + "-" * 60 + "\n")

                row: Dict[str, Any] = {
                    "query_type": qtype,
                    "query": q["query"],
                    "expected_source": q.get("expected_source", []),
                    "pipeline": {"chunking": chunking, "representation": repr_},
                    "k": k,
                    "references": pack.get("refs", []),
                    # Convenience: the same human-readable chunk ids printed to console.
                    "retrieved_chunk_ids": [format_ref_id(r) for r in pack.get("refs", [])],
                    # Convenience: ids paired with the final retrieval score (if available).
                    "retrieved_chunk_id_scores": [
                        {
                            "id": format_ref_id(pack.get("refs", [])[i]),
                            "score": pack.get("retrieved", [])[i].get("score"),
                        }
                        for i in range(min(len(pack.get("refs", [])), len(pack.get("retrieved", []))))
                    ],
                    "answer": answer,
                }
                if args.timeaware:
                    row["time_info"] = pack.get("time_info")
                    row["plan"] = pack.get("plan")
                    row["debug"] = pack.get("debug")
                results.append(row)

    else:
        # ---- Batch mode (file) ----
        queries = load_queries(args.queries_json)

        for qtype, qlist in queries.items():
            for q in qlist:
                for chunking, repr_ in pipelines:
                    for k in ks:
                        print(
                            f"[{qtype}] "
                            f"{chunking}/{repr_} | k={k} | "
                            f"query='{q['query'][:50]}...'"
                        )

                        if args.timeaware:
                            pack = retriever.get_topk_timeaware(q["query"], chunking, repr_, k)
                        else:
                            pack = retriever.get_topk(q["query"], chunking, repr_, k)

                        answer = answer_with_llm(llm, q["query"], pack["context"])

                        row: Dict[str, Any] = {
                            "query_type": qtype,
                            "query": q["query"],
                            "expected_source": q.get("expected_source", []),
                            "pipeline": {"chunking": chunking, "representation": repr_},
                            "k": k,
                            "references": pack.get("refs", []),
                            "retrieved_chunk_ids": [format_ref_id(r) for r in pack.get("refs", [])],
                            "retrieved_chunk_id_scores": [
                                {
                                    "id": format_ref_id(pack.get("refs", [])[i]),
                                    "score": pack.get("retrieved", [])[i].get("score"),
                                }
                                for i in range(min(len(pack.get("refs", [])), len(pack.get("retrieved", []))))
                            ],
                            "answer": answer,
                        }

                        if args.timeaware:
                            row["time_info"] = pack.get("time_info")
                            row["plan"] = pack.get("plan")
                            row["debug"] = pack.get("debug")

                        results.append(row)

    # Save results (both modes) if out_dir is available (always is)
    base = "cli_query" if args.query is not None else os.path.splitext(os.path.basename(args.queries_json))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(out_dir, f"rag_{base}_4pipelines_k{args.k1}-{args.k2}-{args.k3}_{timestamp}.json")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
