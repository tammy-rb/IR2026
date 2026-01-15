"""
Test script for RAG_retriever with different strategies and representations.

Tests:
- Hard filtering (time-aware with explicit ranges)
- Soft decay (time-aware with recency scoring)
- Evolution (early vs late window comparison)

Representations:
- BM25 (sparse lexical)
- Dense (Qdrant vector search)

Chunking:
- Fixed (for consistent testing)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG_retriever.RAG_retriever import RAGRetriever
from RAG_retriever.prefilter.chuncks_selector import ChunkFilter


def print_separator(title: str) -> None:
    """Print a formatted section separator."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_chunks(chunks: list, limit: int = 5) -> None:
    """Print chunk information (doc_id, timestamp, score)."""
    for i, item in enumerate(chunks[:limit], 1):
        chunk = item["chunk"]
        score = item["score"]
        doc_id = chunk.get("doc_id", "N/A")
        doc_ts = chunk.get("doc_timestamp", "N/A")
        doc_date = chunk.get("doc_date_iso", "N/A")
        print(f"  {i}. doc_id={doc_id} | date={doc_date} | ts={doc_ts} | score={score:.4f}")


def test_hard_filtering(retriever: RAGRetriever, representation: str) -> None:
    """Test hard filtering strategy."""
    print_separator(f"HARD FILTERING - {representation.upper()}")
    
    # Query with explicit time reference
    query = "What were the debates about climate change in June 2023?"
    
    # Hard filter: June 2023 (timestamps)
    # June 1, 2023 = 1685577600
    # June 30, 2023 = 1688169599
    
    result = retriever.get_topk_timeaware(
        query=query,
        chunking="fixed",
        representation=representation,
        k=5,
    )
    
    print(f"\nQuery: {query}")
    print(f"Strategy: {result['plan']['strategy']}")
    print(f"Time range: {result['plan']['start_ts']} to {result['plan']['end_ts']}")
    print(f"\nTop 5 chunks:")
    print_chunks(result["retrieved"], limit=5)


def test_soft_decay(retriever: RAGRetriever, representation: str) -> None:
    """Test soft decay reranking strategy."""
    print_separator(f"SOFT DECAY - {representation.upper()}")
    
    # Query with recency preference
    query = "recent discussions on healthcare reform"
    
    result = retriever.get_topk_timeaware(
        query=query,
        chunking="fixed",
        representation=representation,
        k=5,
    )
    
    print(f"\nQuery: {query}")
    print(f"Strategy: {result['plan']['strategy']}")
    print(f"Reference timestamp: {result['plan']['ref_ts']}")
    print(f"Alpha (recency weight): {result['plan']['alpha']}")
    print(f"\nTop 5 chunks:")
    print_chunks(result["retrieved"], limit=5)


def test_evolution(retriever: RAGRetriever, representation: str) -> None:
    """Test evolution retrieval (early vs late windows)."""
    print_separator(f"EVOLUTION RETRIEVAL - {representation.upper()}")
    
    query = "What changed in healthcare policy debates between 2023 and 2025?"
    
    result = retriever.get_topk_evolution(
        query=query,
        chunking="fixed",
        representation=representation,
        k=3,
        window_months=6,
    )
    
    print(f"\nQuery: {query}")
    print(f"Window: {result['window_months']} months")
    
    print(f"\nEarly window ({result['ranges']['early']['start_iso']} to {result['ranges']['early']['end_iso']}):")
    print_chunks(result["retrieved"]["early"], limit=3)
    
    print(f"\nLate window ({result['ranges']['late']['start_iso']} to {result['ranges']['late']['end_iso']}):")
    print_chunks(result["retrieved"]["late"], limit=3)
    
    print(f"\nDebug: early_found={result['debug']['early_found']}, late_found={result['debug']['late_found']}")


def test_filtered_retrieval(retriever: RAGRetriever, representation: str) -> None:
    """Test metadata-based filtering."""
    print_separator(f"FILTERED RETRIEVAL - {representation.upper()}")
    
    query = "healthcare legislation"
    
    # Filter: US corpus only, specific time range
    flt = ChunkFilter(
        time_min_ts=1672531200,  # Jan 1, 2023
        time_max_ts=1704067199,  # Dec 31, 2023
        require_timestamp=True,
        corpora={"us"},
    )
    
    result = retriever.get_topk_filtered(
        query=query,
        chunking="fixed",
        representation=representation,
        k=5,
        flt=flt,
    )
    
    print(f"\nQuery: {query}")
    print(f"Filter: corpus=US, year=2023")
    print(f"\nTop 5 chunks:")
    print_chunks(result["retrieved"], limit=5)


def main() -> None:
    """Run all tests for both BM25 and Dense representations."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  RAG RETRIEVER TEST SUITE".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Initialize retriever (loads all pipelines)
    print("\nInitializing RAG Retriever...")
    retriever = RAGRetriever()
    print("✓ Retriever initialized successfully")
    
    # Test both representations
    for representation in ["bm25", "dense"]:
        print("\n\n" + "▓" * 80)
        print(f"▓  TESTING: {representation.upper()} REPRESENTATION (FIXED CHUNKING)".ljust(78) + "▓")
        print("▓" * 80)
        
        try:
            # Test 1: Hard filtering
            test_hard_filtering(retriever, representation)
            
            # Test 2: Soft decay
            test_soft_decay(retriever, representation)
            
            # Test 3: Evolution
            test_evolution(retriever, representation)
            
            # Test 4: Filtered retrieval
            test_filtered_retrieval(retriever, representation)
            
        except Exception as e:
            print(f"\n❌ ERROR in {representation}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  ALL TESTS COMPLETED".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    main()
