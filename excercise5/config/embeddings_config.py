"""
config/embeddings_config.py

Embedding pipeline batch sizes and parameters.
"""

# Fixed chunking batch sizes
FIXED_DOCS_BATCH_SIZE = 32
FIXED_EMBED_BATCH_SIZE = 8

# Semantic chunking batch sizes
SEMANTIC_DOCS_BATCH_SIZE = 16
SEMANTIC_EMBED_BATCH_SIZE = 4
