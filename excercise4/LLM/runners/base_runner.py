"""Base runner interface for RAG query processing.

Defines the common interface for query runners that can automatically route
queries to appropriate retrieval strategies (evolution vs. standard/timeaware).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from RAG_retriever.RAG_retriever import RAGRetriever
from LLM.LLM_client import LLMClient
from LLM.utils.evolution_query_detector import detect_evolution_bool


class BaseRunner(ABC):
    """
    Abstract base class for RAG query runners.
    
    Provides common interface for running queries with automatic routing
    to appropriate retrieval strategies based on query type detection.
    """
    
    def __init__(
        self,
        retriever: RAGRetriever,
        llm: LLMClient,
        enable_evolution: bool = True,
        window_months: int = 8,
    ):
        """
        Initialize runner with retriever and LLM.
        
        Args:
            retriever: RAG retriever instance
            llm: LLM client instance
            enable_evolution: If True, detect and route evolution queries
            window_months: Window size for evolution queries (default 8)
        """
        self.retriever = retriever
        self.llm = llm
        self.enable_evolution = enable_evolution
        self.window_months = window_months
    
    def route_query(
        self,
        query: str,
        chunking: str,
        representation: str,
        k: int,
        timeaware: bool,
    ) -> Dict[str, Any]:
        """
        Route a query to the appropriate retrieval pipeline.
        
        Decision logic:
        1. If enable_evolution and query is detected as evolution -> get_topk_evolution
        2. Elif timeaware -> get_topk_timeaware
        3. Else -> get_topk (baseline)
        
        Args:
            query: Query string
            chunking: "fixed" or "semantic"
            representation: "bm25" or "dense"
            k: Number of chunks to retrieve
            timeaware: Whether to use time-aware retrieval (ignored if evolution)
            
        Returns:
            Dictionary with retrieval results and metadata including:
            - retrieval_mode: "evolution", "timeaware", or "baseline"
            - context: formatted context for LLM
            - refs: list of chunk references
            - retrieved: list of retrieved chunks with scores
            - Additional mode-specific fields
        """
        is_evolution = self.enable_evolution and detect_evolution_bool(query)
        
        if is_evolution:
            pack = self.retriever.get_topk_evolution(
                query=query,
                chunking=chunking,
                representation=representation,
                k=k,
                window_months=self.window_months,
            )
            pack["retrieval_mode"] = "evolution"
            
        elif timeaware:
            pack = self.retriever.get_topk_timeaware(
                query=query,
                chunking=chunking,
                representation=representation,
                k=k,
            )
            pack["retrieval_mode"] = "timeaware"
            
        else:
            pack = self.retriever.get_topk(
                query=query,
                chunking=chunking,
                representation=representation,
                k=k,
            )
            pack["retrieval_mode"] = "baseline"
        
        return pack
    
    @abstractmethod
    def run(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the runner's main logic.
        
        Must be implemented by subclasses.
        
        Returns:
            List of result dictionaries
        """
        pass
