"""Batch query runner for RAG system.

Processes multiple queries with automatic routing to appropriate retrieval strategies.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from RAG_retriever.RAG_retriever import RAGRetriever
from RAG_retriever.utils import format_ref_id
from LLM.LLM_client import LLMClient
from LLM.runners.base_runner import BaseRunner
from LLM.LLM_factory import make_llm_client


class BatchQueryRunner(BaseRunner):
    """Runner for executing multiple queries with automatic retrieval routing."""
    
    def run(
        self,
        queries_by_group: Dict[str, List[Dict[str, Any]]],
        pipelines: List[Tuple[str, str]],
        ks: List[int],
        timeaware: bool,
        log_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run multiple queries through multiple pipeline configurations.
        
        Args:
            queries_by_group: Dictionary mapping query groups to lists of query dicts
            pipelines: List of (chunking, representation) tuples
            ks: List of k values (number of chunks to retrieve)
            timeaware: If True, uses time-aware retrieval (unless evolution detected)
            log_progress: If True, prints progress for each query
        
        Returns:
            List of result dictionaries, one per query/pipeline/k combination
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
                            print(f"[{group_name}] {chunking}/{repr_} | k={k} | {short}")

                        # Route to appropriate retrieval pipeline
                        pack = self.route_query(query, chunking, repr_, k, timeaware)
                        
                        # Get appropriate LLM based on retrieval mode
                        retrieval_mode = pack.get("retrieval_mode", "baseline")
                        if retrieval_mode == "evolution":
                            llm = make_llm_client(mode="evolution")
                        else:
                            llm = self.llm
                        
                        answer = llm.answer(query, pack["context"])

                        refs = pack.get("refs", [])
                        retrieved = pack.get("retrieved", [])

                        row: Dict[str, Any] = {
                            "query_group": group_name,
                            "query": query,
                            "expected_source": expected,
                            "pipeline": {"chunking": chunking, "representation": repr_},
                            "k": k,
                            "retrieval_mode": retrieval_mode,
                            "timeaware": timeaware,
                            "references": refs,
                            "answer": answer,
                        }
                        
                        # Add mode-specific metadata
                        if retrieval_mode == "evolution":
                            row["ranges"] = pack.get("ranges")
                            row["debug"] = pack.get("debug")
                            row["window_months"] = pack.get("window_months")
                        elif retrieval_mode == "timeaware":
                            row["time_info"] = pack.get("time_info")
                            row["plan"] = pack.get("plan")
                            row["debug"] = pack.get("debug")
                        
                        # Add chunk IDs and scores (handle evolution structure)
                        if isinstance(refs, dict):  # Evolution mode
                            row["retrieved_chunk_ids"] = {
                                "early": [format_ref_id(r) for r in refs.get("early", [])],
                                "late": [format_ref_id(r) for r in refs.get("late", [])],
                            }
                            row["retrieved"] = pack.get("retrieved")
                        else:  # Standard/timeaware mode
                            row["retrieved_chunk_ids"] = [format_ref_id(r) for r in refs]
                            row["retrieved_chunk_id_scores"] = [
                                {"id": format_ref_id(refs[i]), "score": retrieved[i].get("score")}
                                for i in range(min(len(refs), len(retrieved)))
                            ]

                        results.append(row)

        return results


# Legacy function for backward compatibility
def run_batch_queries(
    retriever: RAGRetriever,
    llm: LLMClient,
    queries_by_group: Dict[str, List[Dict[str, Any]]],
    pipelines: List[Tuple[str, str]],
    ks: List[int],
    timeaware: bool,
    log_progress: bool = True,
    enable_evolution: bool = True,
    window_months: int = 8,
) -> List[Dict[str, Any]]:
    """
    Legacy wrapper for BatchQueryRunner.
    
    Args:
        retriever: RAG retriever instance
        llm: LLM client
        queries_by_group: Dictionary mapping groups to query lists
        pipelines: List of (chunking, representation) tuples
        ks: List of k values
        timeaware: If True, uses time-aware retrieval
        log_progress: If True, prints progress
        enable_evolution: If True, detect and use evolution retrieval
        window_months: Window size for evolution queries
    
    Returns:
        List of result dictionaries
    """
    runner = BatchQueryRunner(
        retriever=retriever,
        llm=llm,
        enable_evolution=enable_evolution,
        window_months=window_months,
    )
    return runner.run(
        queries_by_group=queries_by_group,
        pipelines=pipelines,
        ks=ks,
        timeaware=timeaware,
        log_progress=log_progress,
    )

    return results
