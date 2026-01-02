"""Single query runner for RAG system.

Executes a single query across multiple pipeline configurations and k values,
optionally displaying results to console.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from RAG_retriever.RAG_retriever import RAGRetriever
from RAG_retriever.utils import format_ref_id
from LLM.LLM_client import LLMClient


def run_single_query(
    retriever: RAGRetriever,
    llm: LLMClient,
    query: str,
    pipelines: List[Tuple[str, str]],
    ks: List[int],
    timeaware: bool,
    print_console: bool = True,
    query_group: str = "single",
) -> List[Dict[str, Any]]:
    """
    Run a single query through multiple pipeline configurations.
    
    Args:
        retriever: RAG retriever instance for document retrieval
        llm: LLM client for generating answers
        query: The query string to process
        pipelines: List of (chunking, representation) tuples, e.g., [("fixed", "bm25"), ("semantic", "dense")]
        ks: List of k values (number of chunks to retrieve)
        timeaware: If True, uses time-aware retrieval; otherwise uses baseline retrieval
        print_console: If True, prints progress and results to console
        query_group: Group/bucket name for this query (default "single")
    
    Returns:
        List of result dictionaries, one per pipeline/k combination. Each contains:
        - query metadata (query_group, query, pipeline, k, timeaware)
        - retrieval results (references, retrieved_chunk_ids, scores)
        - LLM answer
        - time_info, plan, debug (if timeaware=True)
    """
    results: List[Dict[str, Any]] = []

    for chunking, repr_ in pipelines:
        for k in ks:
            if print_console:
                print(f"[{query_group}] {chunking}/{repr_} | k={k} | timeaware={timeaware}")
                print(f"Query: {query}")

            pack = (
                retriever.get_topk_timeaware(query, chunking, repr_, k)
                if timeaware
                else retriever.get_topk(query, chunking, repr_, k)
            )

            answer = llm.answer(query, pack["context"])

            refs = pack.get("refs", [])
            retrieved = pack.get("retrieved", [])

            if print_console:
                print("\nAnswer:")
                print(answer.strip())
                print("\nTop refs:")
                for i, ref in enumerate(refs):
                    score = retrieved[i].get("score") if i < len(retrieved) else None
                    rid = format_ref_id(ref)
                    print(f"- {rid}" + (f" | score={score}" if score is not None else ""))
                print("\n" + "-" * 60 + "\n")

            row: Dict[str, Any] = {
                "query_group": query_group,
                "query": query,
                "pipeline": {"chunking": chunking, "representation": repr_},
                "k": k,
                "timeaware": timeaware,
                "references": refs,
                "retrieved_chunk_ids": [format_ref_id(r) for r in refs],
                "retrieved_chunk_id_scores": [
                    {"id": format_ref_id(refs[i]), "score": retrieved[i].get("score")}
                    for i in range(min(len(refs), len(retrieved)))
                ],
                "answer": answer,
            }

            if timeaware:
                row["time_info"] = pack.get("time_info")
                row["plan"] = pack.get("plan")
                row["debug"] = pack.get("debug")

            results.append(row)

    return results
