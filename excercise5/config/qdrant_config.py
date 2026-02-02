"""
config/qdrant_config.py

Qdrant vector database configuration.
"""

# Qdrant connection settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Collection names for semantic chunking by corpus
QDRANT_COLLECTION_BRITISH_PARLIAMENT = "british_parliament"
QDRANT_COLLECTION_US_CONGRESS = "us_congress"

# Distance metric for vector similarity
VECTOR_DISTANCE = "cosine"  # "cosine" | "dot" | "euclid"
