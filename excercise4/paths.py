# exercise4/paths.py
from __future__ import annotations

from pathlib import Path

# ============================================================
# Project root (exercise4)
# ============================================================
EXERCISE4_DIR = Path(__file__).resolve().parent

# ============================================================
# Stage directories
# ============================================================
STAGE1_BASELINE_DIR = EXERCISE4_DIR / "stage1_baseline"
STAGE2_INDEXING_DIR = EXERCISE4_DIR / "stage2_indexing"
STAGE3_RETRIEVAL_DIR = EXERCISE4_DIR / "stage3_retrieval"

# ============================================================
# Repository root (IR2026)
# ============================================================
PROJECT_ROOT = EXERCISE4_DIR.parent

# ============================================================
# Data directories
# ============================================================
EXERCISE2_DIR = PROJECT_ROOT / "excercise2"
EXERCISE2_DATA_DIR = EXERCISE2_DIR / "data"

EXERCISE3_DIR = PROJECT_ROOT / "excercise3"

# ============================================================
# Exercise 4 inputs
# ============================================================
QUERIES_DIR = EXERCISE4_DIR / "queries"
TEMPORAL_QUERIES_JSON = QUERIES_DIR / "temporal_queries.json"

# ============================================================
# Exercise 4 outputs
# ============================================================
OUTPUTS_DIR = EXERCISE4_DIR / "outputs"
CHUNKS_DIR = OUTPUTS_DIR / "chunks"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
EMBEDDERS_DIR = EXERCISE4_DIR / "embedders"

BM25_DIR = EMBEDDINGS_DIR / "bm25"
OPENAI_DIR = EMBEDDINGS_DIR / "openai"
STAGE1_DIR = OUTPUTS_DIR / "stage1_baseline_runs"
STAGE3_COMPARISON_DIR = OUTPUTS_DIR / "rag_runs" / "stage3_temporal_analysis"
STAGE3_SUMMARY_DIR = OUTPUTS_DIR / "rag_runs" / "stage3_summaries"

# Precomputed index/artifact directories
BM25_FIXED_DIR = BM25_DIR / "fixed"
BM25_SEM_DIR = BM25_DIR / "semantic"

FAISS_FIXED_DIR = OPENAI_DIR / "fixed_faiss"
FAISS_SEM_DIR = OPENAI_DIR / "semantic_faiss"

# Chunk manifests
CHUNKS_FIXED_JSONL = CHUNKS_DIR / "chunks_fixed.jsonl"
CHUNKS_SEM_JSONL = CHUNKS_DIR / "chunks_semantic.jsonl"

# reports
REPORTS_DIR = OUTPUTS_DIR / "reports"
TIME_HIST_DIR = REPORTS_DIR / "time_histograms"

# ============================================================
# Convenience
# ============================================================
def ensure_dirs() -> None:
    for p in [
        OUTPUTS_DIR,
        CHUNKS_DIR,
        BM25_DIR,
        EMBEDDINGS_DIR,
        REPORTS_DIR,
        TIME_HIST_DIR,
        STAGE1_DIR,
        STAGE3_COMPARISON_DIR,
        STAGE3_SUMMARY_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
