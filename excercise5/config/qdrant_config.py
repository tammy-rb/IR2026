"""
config/qdrant_config.py

Qdrant vector database configuration.
"""

# Qdrant connection settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Collection names for different chunking strategies
QDRANT_COLLECTION_FIXED = "chunks_openai_fixed_large"
QDRANT_COLLECTION_SEMANTIC = "chunks_openai_semantic_large"

# Distance metric for vector similarity
VECTOR_DISTANCE = "cosine"  # "cosine" | "dot" | "euclid"
