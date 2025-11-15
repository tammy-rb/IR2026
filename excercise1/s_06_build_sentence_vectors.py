"""Sentence Vector Builder using Transformer Models.

This module provides utilities for building document embeddings using
Sentence Transformers (SimCSE, SBERT, etc.). It handles document chunking,
encoding, and aggregation to create document-level vector representations.
"""

import os
import glob
import json
import numpy as np

import spacy
from lxml import etree
from sentence_transformers import SentenceTransformer

# Load spaCy model with only sentence segmentation enabled
nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])


# ==============================================================================
# Document Loaders
# ==============================================================================

def load_raw_xml_documents(raw_dir="docs", pattern="*.xml"):
    """Load and extract text from XML documents.
    
    Parses XML files and extracts all text nodes, combining them into a single
    string per document. This is a legacy function - prefer using pre-extracted
    text files for better performance.
    
    Args:
        raw_dir: Directory containing XML files.
        pattern: Glob pattern for matching XML files.
        
    Returns:
        tuple: (texts, filenames) where:
            - texts: List of document text strings.
            - filenames: List of corresponding XML filenames.
    """
    texts = []
    filenames = []

    for path in sorted(glob.glob(os.path.join(raw_dir, pattern))):
        base = os.path.basename(path)
        filenames.append(base)

        tree = etree.parse(path)
        all_texts = tree.xpath("//text()")
        doc_text = " ".join(t.strip() for t in all_texts if t.strip())
        texts.append(doc_text)

    print(f"Loaded {len(texts)} raw XML documents from {raw_dir}")
    return texts, filenames


# ==============================================================================
# Text Chunking Utilities
# ==============================================================================

def split_document_to_sentences(text):
    """Split text into sentences using spaCy's sentence segmentation.
    
    Args:
        text: Input text to segment.
        
    Returns:
        List of sentence strings, with whitespace stripped.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def chunk_long_sentence(sentence, max_tokens=256):
    """Split a long sentence into smaller chunks.
    
    Divides sentences that exceed max_tokens by splitting on word boundaries.
    This prevents memory issues when encoding very long sentences.
    
    Args:
        sentence: The sentence to split.
        max_tokens: Maximum number of words per chunk.
        
    Returns:
        List of text chunks, each containing at most max_tokens words.
    """
    words = sentence.split()
    chunks = []

    for i in range(0, len(words), max_tokens):
        part = " ".join(words[i:i + max_tokens])
        chunks.append(part)

    return chunks


def split_document_into_chunks(text, max_tokens=512):
    """Split document into semantically coherent chunks.
    
    Groups consecutive sentences together into chunks up to max_tokens,
    creating larger, more meaningful units for encoding. This approach
    preserves context better than encoding individual sentences.
    
    Strategy:
        1. Split text into sentences using spaCy.
        2. Group consecutive sentences until reaching max_tokens.
        3. Split oversized sentences into smaller chunks.
        4. Start new chunk when adding next sentence would exceed limit.
    
    Args:
        text: Input document text.
        max_tokens: Maximum number of words per chunk.
        
    Returns:
        List of text chunks, each containing grouped sentences.
    """
    sentences = split_document_to_sentences(text)

    if not sentences:
        return [text.strip()] if text.strip() else []

    final_chunks = []
    current_chunk = []
    current_word_count = 0

    for sent in sentences:
        sent_word_count = len(sent.split())

        # If single sentence is too long, split it separately
        if sent_word_count > max_tokens:
            # First, save any accumulated chunk
            if current_chunk:
                final_chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            # Split the long sentence
            sub_chunks = chunk_long_sentence(sent, max_tokens=max_tokens)
            final_chunks.extend(sub_chunks)
        else:
            # Check if adding this sentence would exceed max_tokens
            if current_word_count + sent_word_count > max_tokens and current_chunk:
                # Save current chunk and start a new one
                final_chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_word_count = sent_word_count
            else:
                # Add sentence to current chunk
                current_chunk.append(sent)
                current_word_count += sent_word_count

    # Don't forget the last chunk
    if current_chunk:
        final_chunks.append(" ".join(current_chunk))

    return final_chunks


def build_chunks_for_all_docs(texts, max_tokens=512):
    """Chunk all documents in the corpus.
    
    Applies the chunking strategy to each document and provides statistics
    about the resulting chunks.
    
    Args:
        texts: List of document texts.
        max_tokens: Maximum number of words per chunk.
        
    Returns:
        List of lists, where each inner list contains the chunks for one document.
    """
    all_doc_chunks = []
    total_chunks = 0

    for text in texts:
        chunks = split_document_into_chunks(text, max_tokens=max_tokens)
        all_doc_chunks.append(chunks)
        total_chunks += len(chunks)

    print(f"Built chunks for {len(all_doc_chunks)} documents")
    print(f"Total chunks: {total_chunks} (avg {total_chunks/len(texts):.1f} per doc)")
    return all_doc_chunks


# ==============================================================================
# Encoding and Embedding Generation
# ==============================================================================

def encode_chunked_documents(model, all_doc_chunks, batch_size=16, max_chunks_per_call=64):
    """Encode chunked documents into fixed-size vectors.
    
    Processes each document's chunks in batches to avoid memory issues,
    then aggregates chunk embeddings into a single document vector using
    mean pooling.
    
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
    print("Encoding documents with chunked strategy (memory friendly)...")

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
            print(f"  Encoded {doc_idx + 1} / {len(all_doc_chunks)} documents")

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
# High-Level Pipeline Functions
# ==============================================================================

def build_chunked_embeddings_for_xml(
    raw_dir,
    model_name,
    out_dir,
    xml_pattern="*.xml",
    chunk_max_tokens=512,
    batch_size=16,
    prefix="embeddings",
    max_chunks_per_call=64,
):
    """Complete pipeline for building document embeddings from XML files.
    
    This is a generic pipeline that works with any SentenceTransformer model
    (SimCSE, SBERT, etc.). It handles the full workflow from XML parsing to
    saving final embeddings.
    
    Pipeline steps:
        1. Load and parse XML documents.
        2. Chunk documents into manageable pieces.
        3. Load the specified transformer model.
        4. Encode chunks and aggregate to document vectors.
        5. Save embeddings, filenames, and model metadata.
    
    Args:
        raw_dir: Directory containing XML files.
        model_name: HuggingFace model identifier (e.g., 'princeton-nlp/unsup-simcse-bert-base-uncased').
        out_dir: Output directory for saving results.
        xml_pattern: Glob pattern for matching XML files.
        chunk_max_tokens: Maximum words per chunk.
        batch_size: Number of chunks to encode per batch.
        prefix: Filename prefix for embeddings.
        max_chunks_per_call: Maximum chunks to process per encoding call.
    """
    print("=== Starting chunked embedding pipeline on XML documents ===")

    # Step 1: Load raw XML documents
    texts, filenames = load_raw_xml_documents(raw_dir=raw_dir, pattern=xml_pattern)
    if not texts:
        print("No XML documents found. Aborting.")
        return

    # Step 2: Build chunks for all documents
    all_doc_chunks = build_chunks_for_all_docs(texts, max_tokens=chunk_max_tokens)

    # Step 3: Load transformer model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Step 4: Encode chunks and aggregate to document vectors
    doc_embeddings = encode_chunked_documents(
        model,
        all_doc_chunks,
        batch_size=batch_size,
        max_chunks_per_call=max_chunks_per_call,
    )

    # Step 5: Save embeddings and metadata
    save_embeddings(
        doc_embeddings,
        filenames,
        out_dir=out_dir,
        model_name=model_name,
        prefix=prefix,
    )

    print("🎉 Chunked embedding pipeline completed successfully!")
