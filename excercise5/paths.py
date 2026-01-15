# exercise5/paths.py
from __future__ import annotations

from pathlib import Path

# ============================================================
# Project root (exercise5)
# ============================================================
EXERCISE5_DIR = Path(__file__).resolve().parent

# ============================================================
# Repository root (IR2026)
# ============================================================
PROJECT_ROOT = EXERCISE5_DIR.parent

# ============================================================
# Data sources - Raw corpus directories
# ============================================================
# Legacy data from excercise2
EXCERCISE2_DATA_DIR = PROJECT_ROOT / "excercise2" / "data"
BRITISH_PARLIAMENT_DIR = EXCERCISE2_DATA_DIR / "british_parliament_debates"
US_CONGRESS_DIR = EXCERCISE2_DATA_DIR / "US_congress_debates"

# Additional data sources (add new corpora here)
# BBC_DATA_DIR = PROJECT_ROOT / "data" / "bbc_debates"

# All active corpus directories (add new sources to this list)
CORPUS_DIRS = [
    BRITISH_PARLIAMENT_DIR,
    US_CONGRESS_DIR,
    # BBC_DATA_DIR,  # uncomment when available
]

# ============================================================
# Exercise 5 outputs
# ============================================================
OUTPUTS_DIR = EXERCISE5_DIR / "outputs"
CHUNKS_DIR = OUTPUTS_DIR / "chunks"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
BM25_DIR = EMBEDDINGS_DIR / "bm25"
BM25_FIXED_DIR = BM25_DIR / "fixed"
BM25_SEM_DIR = BM25_DIR / "semantic"
OPENAI_DIR = EMBEDDINGS_DIR / "openai"
OPENAI_QDRANT_DIR = EMBEDDINGS_DIR / "openai_qdrant"
OPENAI_QDRANT_FIXED_DIR = OPENAI_QDRANT_DIR / "fixed"
OPENAI_QDRANT_SEMANTIC_DIR = OPENAI_QDRANT_DIR / "semantic"

# Chunking methods
CHUNKS_FIXED_JSONL = CHUNKS_DIR / "chunks_fixed.jsonl"
CHUNKS_SEM_JSONL = CHUNKS_DIR / "chunks_semantic.jsonl"
CHUNKS_SEMANTIC_JSONL = CHUNKS_SEM_JSONL  # Alias for consistency

# ============================================================
# Models
# ============================================================
MODELS_DIR = EXERCISE5_DIR / "models"

# ============================================================
# Convenience
# ============================================================
def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for p in [
        OUTPUTS_DIR,
        CHUNKS_DIR,
        EMBEDDINGS_DIR,
        BM25_DIR,
        BM25_FIXED_DIR,
        BM25_SEM_DIR,
        OPENAI_QDRANT_DIR,
        OPENAI_QDRANT_FIXED_DIR,
        OPENAI_QDRANT_SEMANTIC_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
