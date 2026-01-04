"""Single query runner for RAG system.

Executes a single query across multiple pipeline configurations and k values,
with automatic routing to evolution/timeaware/baseline retrieval.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from RAG_retriever.RAG_retriever import RAGRetriever
from RAG_retriever.utils import format_ref_id
from LLM.LLM_client import LLMClient
from LLM.runners.base_runner import BaseRunner
from LLM.LLM_factory import make_llm_client


class SingleQueryRunner(BaseRunner):
    """Runner for executing a single query with automatic retrieval routing."""
    
    def run(
        self,
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
            query: The query string to process
            pipelines: List of (chunking, representation) tuples
            ks: List of k values (number of chunks to retrieve)
            timeaware: If True, uses time-aware retrieval (unless evolution detected)
            print_console: If True, prints progress and results to console
            query_group: Group/bucket name for this query (default "single")
        
        Returns:
            List of result dictionaries, one per pipeline/k combination
        """
        results: List[Dict[str, Any]] = []

        for chunking, repr_ in pipelines:
            for k in ks:
                if print_console:
                    print(f"[{query_group}] {chunking}/{repr_} | k={k}")
                    print(f"Query: {query}")

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

                if print_console:
                    print(f"\nRetrieval Mode: {retrieval_mode}")
                    print("\nAnswer:")
                    print(answer.strip())
                    print("\nTop refs:")
                    
                    # Handle evolution refs (early/late structure)
                    if retrieval_mode == "evolution" and isinstance(refs, dict):
                        print("  EARLY:")
                        for ref in refs.get("early", [])[:5]:
                            print(f"  - {format_ref_id(ref)}")
                        print("  LATE:")
                        for ref in refs.get("late", [])[:5]:
                            print(f"  - {format_ref_id(ref)}")
                    else:
                        for i, ref in enumerate(refs[:10]):
                            score = retrieved[i].get("score") if i < len(retrieved) else None
                            rid = format_ref_id(ref)
                            print(f"- {rid}" + (f" | score={score}" if score is not None else ""))
                    print("\n" + "-" * 60 + "\n")

                row: Dict[str, Any] = {
                    "query_group": query_group,
                    "query": query,
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
def run_single_query(
    retriever: RAGRetriever,
    llm: LLMClient,
    query: str,
    pipelines: List[Tuple[str, str]],
    ks: List[int],
    timeaware: bool,
    print_console: bool = True,
    query_group: str = "single",
    enable_evolution: bool = True,
    window_months: int = 8,
) -> List[Dict[str, Any]]:
    """
    Legacy wrapper for SingleQueryRunner.
    
    Args:
        retriever: RAG retriever instance
        llm: LLM client
        query: Query string
        pipelines: List of (chunking, representation) tuples
        ks: List of k values
        timeaware: If True, uses time-aware retrieval
        print_console: If True, prints results to console
        query_group: Group name for query
        enable_evolution: If True, detect and use evolution retrieval
        window_months: Window size for evolution queries
    
    Returns:
        List of result dictionaries
    """
    runner = SingleQueryRunner(
        retriever=retriever,
        llm=llm,
        enable_evolution=enable_evolution,
        window_months=window_months,
    )
    return runner.run(
        query=query,
        pipelines=pipelines,
        ks=ks,
        timeaware=timeaware,
        print_console=print_console,
        query_group=query_group,
    )
