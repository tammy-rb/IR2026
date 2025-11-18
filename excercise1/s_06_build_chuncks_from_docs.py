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

RAW_DIR = "docs"

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
    creating larger, more meaningful units for encoding.
    This approach preserves context better than encoding individual sentences.
    
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
                current_chunk.append(sent)
                current_word_count += sent_word_count

    # last chunk
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
# High-Level Pipeline – Build & Save Chunks
# ==============================================================================

def build_and_save_chunks_for_xml(
    raw_dir,
    out_dir="docs_chuncks",
    xml_pattern="*.xml",
    chunk_max_tokens=512,
):
    """Complete pipeline for building and saving document chunks from XML files.
    
    Each document's chunks are saved in a separate JSON file with the same
    base name as the original document (e.g., debates2023-06-28d.xml.chunks.json).
    
    Pipeline steps:
    1. Load and parse XML documents.
    2. Chunk each document into manageable pieces.
    3. Save each document's chunks to its own file in out_dir.
    
    Args:
        raw_dir: Directory containing XML files.
        out_dir: Output directory for saving chunk files.
        xml_pattern: Glob pattern for matching XML files.
        chunk_max_tokens: Maximum words per chunk.
    """
    print("=== Starting chunked building pipeline on XML documents ===")

    # Step 1: Load raw XML documents
    texts, filenames = load_raw_xml_documents(raw_dir=raw_dir, pattern=xml_pattern)
    if not texts:
        print("No XML documents found. Aborting.")
        return

    # Step 2: Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Step 3: Process and save each document's chunks individually
    total_chunks = 0
    for i, (text, filename) in enumerate(zip(texts, filenames)):
        # Chunk the document
        chunks = split_document_into_chunks(text, max_tokens=chunk_max_tokens)
        total_chunks += len(chunks)
        
        # Create chunk filename (e.g., debates2023-06-28d.xml.chunks.json)
        chunk_filename = f"{filename}.chunks.json"
        chunk_path = os.path.join(out_dir, chunk_filename)
        
        # Save chunks for this document
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} / {len(texts)} documents")

    print(f"\nProcessed {len(texts)} documents")
    print(f"Total chunks: {total_chunks} (avg {total_chunks/len(texts):.1f} per doc)")
    print(f"Saved individual chunk files in: {out_dir}")
    print("🎉 Chunked building pipeline completed successfully!")

if __name__ == "__main__":
    build_and_save_chunks_for_xml(
        raw_dir=RAW_DIR,
        out_dir="docs_chuncks",
        xml_pattern="*.xml",
        chunk_max_tokens=256,
    )
