"""LLM Runners package for RAG query execution.

Provides runners for executing RAG queries with automatic routing to appropriate
retrieval strategies (evolution, timeaware, baseline).
"""
from __future__ import annotations

from LLM.runners.base_runner import BaseRunner
from LLM.runners.single_runner import SingleQueryRunner, run_single_query
from LLM.runners.batch_runner import BatchQueryRunner, run_batch_queries

__all__ = [
    "BaseRunner",
    "SingleQueryRunner",
    "BatchQueryRunner",
    "run_single_query",
    "run_batch_queries",
]
