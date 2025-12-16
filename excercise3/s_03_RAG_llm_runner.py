"""
RAG_llm_runner.py

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

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from s_03_RAG_retriever import RAGRetriever


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
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
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_json", required=True)
    parser.add_argument("--k1", type=int, default=3)
    parser.add_argument("--k2", type=int, default=5)
    parser.add_argument("--k3", type=int, default=10)
    parser.add_argument("--llm_model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    ensure_dir(OUT_DIR)

    retriever = RAGRetriever()
    llm = ChatOpenAI(model=args.llm_model, temperature=args.temperature)

    queries = load_queries(args.queries_json)
    ks = [args.k1, args.k2, args.k3]

    pipelines = [
        ("fixed", "bm25"),
        ("semantic", "bm25"),
        ("fixed", "dense"),
        ("semantic", "dense"),
    ]

    results = []

    for qtype, qlist in queries.items():
        for q in qlist:
            for chunking, repr_ in pipelines:
                for k in ks:
                    print(
                    f"[{qtype}] "
                    f"{chunking}/{repr_} | k={k} | "
                    f"query='{q['query'][:50]}...'"
                    )
                    pack = retriever.get_topk(q["query"], chunking, repr_, k)
                    answer = answer_with_llm(llm, q["query"], pack["context"])

                    results.append({
                        "query_type": qtype,
                        "query": q["query"],
                        "expected_source": q.get("expected_source", []),
                        "pipeline": {"chunking": chunking, "representation": repr_},
                        "k": k,
                        "references": pack["refs"],
                        "answer": answer,
                    })

    out = os.path.join(
        OUT_DIR,
        f"rag_4pipelines_k{args.k1}-{args.k2}-{args.k3}.json"
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved results to {out}")


if __name__ == "__main__":
    main()
