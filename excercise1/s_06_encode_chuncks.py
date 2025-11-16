"""Sentence Vector Builder using Transformer Models.

This module provides utilities for building document embeddings using
Sentence Transformers (SimCSE, SBERT, etc.). It handles document chunking,
encoding, and aggregation to create document-level vector representations.
"""

import os
import glob
import json
import numpy as np

from sentence_transformers import SentenceTransformer


# ==============================================================================
# Encoding and Embedding Generation
# ==============================================================================

def encode_chunked_documents(model, all_doc_chunks, batch_size=16, max_chunks_per_call=64):
    """Encode chunked documents into fixed-size vectors.
    
    Processes each document's chunks in batches to avoid memory issues, then
    aggregates chunk embeddings into a single document vector using mean pooling.
    
    Memory-efficient strategy:
    1. Process each document independently.
    2. For large documents, encode chunks in groups of max_chunks_per_call.
    3. Accumulate chunk embeddings incrementally.
    4. Average all chunk embeddings to create document vector.
    
    Args:
        model: SentenceTransformer model for encoding.
        all_doc_chunks: List of lists containing document chunks.
        batch_size: Number of chunks to encode simultaneously.
        max_chunks_per_call: Maximum chunks to process per encoding call.
        
    Returns:
        np.ndarray: Document embeddings matrix of shape (n_docs, embedding_dim).
    """
    print("Encoding chuncked documents...")
    doc_embeddings = []
    emb_dim = model.get_sentence_embedding_dimension()

    for doc_idx, chunks in enumerate(all_doc_chunks):
        # Handle empty documents with zero vectors
        if not chunks:
            doc_embeddings.append(np.zeros(emb_dim, dtype=np.float32))
            continue

        doc_sum = np.zeros(emb_dim, dtype=np.float32)
        total_chunks = 0

        # Process this document's chunks in manageable batches
        for start in range(0, len(chunks), max_chunks_per_call):
            sub_chunks = chunks[start:start + max_chunks_per_call]

            sub_embs = model.encode(
                sub_chunks,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Accumulate embeddings: shape (len(sub_chunks), emb_dim)
            doc_sum += sub_embs.sum(axis=0)
            total_chunks += sub_embs.shape[0]

        # Average all chunk embeddings to get final document vector
        doc_vec = doc_sum / max(total_chunks, 1)
        doc_embeddings.append(doc_vec)

        if (doc_idx + 1) % 10 == 0:
            print(f" Encoded {doc_idx + 1} / {len(all_doc_chunks)} documents")

    doc_embeddings = np.vstack(doc_embeddings)
    print(f"Finished encoding. Document embeddings shape: {doc_embeddings.shape}")
    return doc_embeddings


# ==============================================================================
# File I/O and Persistence
# ==============================================================================

def save_embeddings(doc_embeddings, filenames, out_dir, model_name, prefix="embeddings"):
    """Save document embeddings and metadata to disk.
    
    Creates three files in the output directory:
    - {prefix}.npy: NumPy array of embeddings (n_docs, embedding_dim).
    - filenames.json: List mapping row indices to document filenames.
    - model_name.txt: Identifier of the transformer model used.
    
    Args:
        doc_embeddings: NumPy array of document vectors.
        filenames: List of document filenames.
        out_dir: Output directory path.
        model_name: Name of the SentenceTransformer model.
        prefix: Prefix for the embeddings filename.
    """
    os.makedirs(out_dir, exist_ok=True)

    emb_path = os.path.join(out_dir, f"{prefix}.npy")
    np.save(emb_path, doc_embeddings)

    fn_path = os.path.join(out_dir, "filenames.json")
    with open(fn_path, "w", encoding="utf-8") as f:
        json.dump(filenames, f)

    model_path = os.path.join(out_dir, "model_name.txt")
    with open(model_path, "w", encoding="utf-8") as f:
        f.write(model_name)

    print(f"Saved embeddings to: {emb_path}")
    print(f"Saved filenames to: {fn_path}")
    print(f"Saved model name to: {model_path}")
    print(f"All outputs saved in: {out_dir}")


# ==============================================================================
# High-Level Pipeline – Load Chunks & Build Embeddings
# ==============================================================================

def load_chunks_and_filenames(chunks_dir="docs_chuncks"):
    """Load document chunks from individual chunk files.
    
    Loads all *.chunks.json files from the chunks directory and reads each
    document's chunks separately.
    
    Args:
        chunks_dir: Directory containing the chunk files.
        
    Returns:
        tuple: (all_doc_chunks, filenames) where:
            - all_doc_chunks: List of lists containing document chunks.
            - filenames: List of original document filenames (without .chunks.json).
    """
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "*.chunks.json")))
    
    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}")
        return [], []
    
    all_doc_chunks = []
    filenames = []
    
    for chunk_path in chunk_files:
        # Extract original filename (remove .chunks.json suffix)
        base = os.path.basename(chunk_path)
        original_filename = base.replace(".chunks.json", "")
        filenames.append(original_filename)
        
        # Load chunks for this document
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_doc_chunks.append(chunks)
    
    print(f"Loaded {len(all_doc_chunks)} documents with chunks from {chunks_dir}")
    return all_doc_chunks, filenames


def build_embeddings_from_saved_chunks(
    chunks_dir,
    model_name,
    out_dir,
    batch_size=16,
    prefix="embeddings",
    max_chunks_per_call=64,
):
    """Complete pipeline for building document embeddings from pre-saved chunks.
    
    This is a generic pipeline that works with any SentenceTransformer model
    (SimCSE, SBERT, etc.). It handles the full workflow from precomputed
    chunks to saving final embeddings.
    
    Pipeline steps:
    1. Load precomputed document chunks and filenames.
    2. Load the specified transformer model.
    3. Encode chunks and aggregate to document vectors.
    4. Save embeddings, filenames, and model metadata.
    
    Args:
        chunks_dir: Directory containing chunk files (e.g., 'docs_chuncks').
        model_name: HuggingFace model identifier
                    (e.g., 'princeton-nlp/unsup-simcse-bert-base-uncased').
        out_dir: Output directory for saving results.
        batch_size: Number of chunks to encode per batch.
        prefix: Filename prefix for embeddings.
        max_chunks_per_call: Maximum chunks to process per encoding call.
    """
    print("=== Starting embedding pipeline from precomputed chunks ===")

    # Step 1: Load chunks and filenames
    all_doc_chunks, filenames = load_chunks_and_filenames(chunks_dir=chunks_dir)
    if not all_doc_chunks:
        print("No document chunks found. Aborting.")
        return

    # Step 2: Load transformer model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Step 3: Encode chunks and aggregate to document vectors
    doc_embeddings = encode_chunked_documents(
        model,
        all_doc_chunks,
        batch_size=batch_size,
        max_chunks_per_call=max_chunks_per_call,
    )

    # Step 4: Save embeddings and metadata
    save_embeddings(
        doc_embeddings,
        filenames,
        out_dir=out_dir,
        model_name=model_name,
        prefix=prefix,
    )

    print("🎉 Embedding pipeline from chunks completed successfully!")
