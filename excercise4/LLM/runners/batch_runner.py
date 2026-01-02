"""Batch query runner for RAG system.

Processes multiple queries (organized by type) across multiple pipeline configurations
and k values, with optional progress logging.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from RAG_retriever.RAG_retriever import RAGRetriever
from RAG_retriever.utils import format_ref_id
from LLM.LLM_client import LLMClient


def run_batch_queries(
    retriever: RAGRetriever,
    llm: LLMClient,
    queries_by_group: Dict[str, List[Dict[str, Any]]],
    pipelines: List[Tuple[str, str]],
    ks: List[int],
    timeaware: bool,
    log_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run multiple queries through multiple pipeline configurations.
    
    Args:
        retriever: RAG retriever instance for document retrieval
        llm: LLM client for generating answers
        queries_by_group: Dictionary mapping query groups (temporal buckets like "point_in_time", 
                         "recency", "explicit_range", "comparison", "evolution", or legacy 
                         "factual"/"conceptual") to lists of query dicts. Each query dict must 
                         have "query" key and optionally "expected_source" key.
        pipelines: List of (chunking, representation) tuples, e.g., [("fixed", "bm25"), ("semantic", "dense")]
        ks: List of k values (number of chunks to retrieve)
        timeaware: If True, uses time-aware retrieval; otherwise uses baseline retrieval
        log_progress: If True, prints progress for each query/pipeline/k combination
    
    Returns:
        List of result dictionaries, one per query/pipeline/k combination. Each contains:
        - query metadata (query_group, query, expected_source, pipeline, k, timeaware)
        - retrieval results (references, retrieved_chunk_ids, scores)
        - LLM answer
        - time_info, plan, debug (if timeaware=True)
    """
    results: List[Dict[str, Any]] = []

    for group_name, qlist in queries_by_group.items():
        for q in qlist:
            query = q["query"]
            expected = q.get("expected_source", [])

            for chunking, repr_ in pipelines:
                for k in ks:
                    if log_progress:
                        short = (query[:60] + "...") if len(query) > 60 else query
                        print(f"[{group_name}] {chunking}/{repr_} | k={k} | timeaware={timeaware} | {short}")

                    pack = (
                        retriever.get_topk_timeaware(query, chunking, repr_, k)
                        if timeaware
                        else retriever.get_topk(query, chunking, repr_, k)
                    )
                    answer = llm.answer(query, pack["context"])

                    refs = pack.get("refs", [])
                    retrieved = pack.get("retrieved", [])

                    row: Dict[str, Any] = {
                        "query_group": group_name,
                        "query": query,
                        "expected_source": expected,
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
