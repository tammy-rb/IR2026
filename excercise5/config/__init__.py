"""
config/__init__.py

Centralized configuration for Exercise 5.
Import all config values from this module for easy access.
"""

from config.qdrant_config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_FIXED,
    QDRANT_COLLECTION_SEMANTIC,
    VECTOR_DISTANCE,
)

from config.openai_config import (
    OPENAI_EMBED_MODEL,
    OPENAI_EMBED_DIMENSIONS,
)

from config.embeddings_config import (
    FIXED_DOCS_BATCH_SIZE,
    FIXED_EMBED_BATCH_SIZE,
    SEMANTIC_DOCS_BATCH_SIZE,
    SEMANTIC_EMBED_BATCH_SIZE,
)

__all__ = [
    # Qdrant
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_COLLECTION_FIXED",
    "QDRANT_COLLECTION_SEMANTIC",
    "VECTOR_DISTANCE",
    # OpenAI
    "OPENAI_EMBED_MODEL",
    "OPENAI_EMBED_DIMENSIONS",
    # Embeddings
    "FIXED_DOCS_BATCH_SIZE",
    "FIXED_EMBED_BATCH_SIZE",
    "SEMANTIC_DOCS_BATCH_SIZE",
    "SEMANTIC_EMBED_BATCH_SIZE",
]
