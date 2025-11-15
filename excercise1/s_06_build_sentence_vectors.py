import os
import glob
import json
import numpy as np

import spacy
from lxml import etree
from sentence_transformers import SentenceTransformer


# Global spaCy object for sentence splitting
nlp = spacy.load("en_core_web_sm")


# ====================== XML Loader ======================

def load_raw_xml_documents(raw_dir="docs", pattern="*.xml"):
    """
    Load all XML documents from raw_dir and return:
      - texts: list of full text per document
      - filenames: list of corresponding filenames

    Each document is represented as one long string
    containing all text nodes extracted from the XML.
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


# ====================== Chunking helpers ======================

def split_document_to_sentences(text):
    """
    Use spaCy to split a document into sentences.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def chunk_long_sentence(sentence, max_tokens=256):
    """
    If a single sentence is longer than max_tokens (approx. words),
    split it into smaller chunks.
    """
    words = sentence.split()
    chunks = []

    for i in range(0, len(words), max_tokens):
        part = " ".join(words[i:i + max_tokens])
        chunks.append(part)

    return chunks


def split_document_into_chunks(text, max_tokens=256):
    """
    Split a long document into smaller chunks suitable for
    BERT-based models (SimCSE, SBERT, etc.):

      1. Use spaCy to split into sentences.
      2. If a sentence is too long, split it further.
    """
    final_chunks = []
    sentences = split_document_to_sentences(text)

    for sent in sentences:
        word_count = len(sent.split())
        if word_count > max_tokens:
            sub_chunks = chunk_long_sentence(sent, max_tokens=max_tokens)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(sent)

    # Fallback: if no chunks but text is non-empty, keep the whole text
    if not final_chunks and text.strip():
        final_chunks = [text.strip()]

    return final_chunks


def build_chunks_for_all_docs(texts, max_tokens=256):
    """
    Build chunk lists for each document in the corpus.

    Returns:
      all_doc_chunks: list of list of strings (chunks per document)
    """
    all_doc_chunks = []
    for text in texts:
        chunks = split_document_into_chunks(text, max_tokens=max_tokens)
        all_doc_chunks.append(chunks)

    print(f"Built chunks for {len(all_doc_chunks)} documents")
    return all_doc_chunks


# ====================== Encoding (chunked) ======================

def encode_chunked_documents(model, all_doc_chunks, batch_size=32, max_chunks_per_call=256):
    """
    Encode documents that were split into chunks.

    For each document:
      - encode its chunks with the SentenceTransformer model
      - process chunks in smaller groups (max_chunks_per_call)
      - accumulate a running sum of all chunk embeddings
      - average them into one document vector

    This is memory-efficient and avoids OOM issues with very large documents.

    Returns:
      doc_embeddings: np.ndarray of shape (n_docs, embedding_dim)
    """
    print("Encoding documents with chunked strategy (memory friendly)...")

    doc_embeddings = []
    emb_dim = model.get_sentence_embedding_dimension()

    for doc_idx, chunks in enumerate(all_doc_chunks):
        # Empty document → zero vector
        if not chunks:
            doc_embeddings.append(np.zeros(emb_dim, dtype=np.float32))
            continue

        doc_sum = np.zeros(emb_dim, dtype=np.float32)
        total_chunks = 0

        # Process chunks for THIS doc in smaller pieces
        for start in range(0, len(chunks), max_chunks_per_call):
            sub_chunks = chunks[start:start + max_chunks_per_call]

            sub_embs = model.encode(
                sub_chunks,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # sub_embs shape: (len(sub_chunks), emb_dim)

            doc_sum += sub_embs.sum(axis=0)
            total_chunks += sub_embs.shape[0]

        # Mean pooling over all chunk embeddings for this document
        doc_vec = doc_sum / max(total_chunks, 1)
        doc_embeddings.append(doc_vec)

        if (doc_idx + 1) % 10 == 0:
            print(f"  Encoded {doc_idx + 1} / {len(all_doc_chunks)} documents")

    doc_embeddings = np.vstack(doc_embeddings)
    print(f"Finished encoding. Document embeddings shape: {doc_embeddings.shape}")
    return doc_embeddings


# ====================== Save helpers ======================

def save_embeddings(doc_embeddings, filenames, out_dir, model_name, prefix="embeddings"):
    """
    Save embeddings matrix, filenames mapping, and model name.

    Files:
      - {prefix}.npy       : embeddings matrix (n_docs, embedding_dim)
      - filenames.json     : list of filenames
      - model_name.txt     : the model identifier used
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


# ====================== Generic pipeline ======================

def build_chunked_embeddings_for_xml(
    raw_dir,
    model_name,
    out_dir,
    xml_pattern="*.xml",
    chunk_max_tokens=256,
    batch_size=32,
    prefix="embeddings",
    max_chunks_per_call=256,
):
    """
    Generic pipeline for:
      - loading raw XML debates,
      - chunking them,
      - encoding with a SentenceTransformer model,
      - saving document-level embeddings.

    This works for BOTH SimCSE and SBERT, only the model_name and out_dir change.
    """
    print("=== Starting chunked embedding pipeline on XML documents ===")

    # 1) Load raw XML docs
    texts, filenames = load_raw_xml_documents(raw_dir=raw_dir, pattern=xml_pattern)
    if not texts:
        print("No XML documents found. Aborting.")
        return

    # 2) Build chunks per document
    all_doc_chunks = build_chunks_for_all_docs(texts, max_tokens=chunk_max_tokens)

    # 3) Load SentenceTransformer model (SimCSE / SBERT / etc.)
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # 4) Encode chunks and aggregate to document vectors (memory-friendly)
    doc_embeddings = encode_chunked_documents(
        model,
        all_doc_chunks,
        batch_size=batch_size,
        max_chunks_per_call=max_chunks_per_call,
    )

    # 5) Save everything
    save_embeddings(
        doc_embeddings,
        filenames,
        out_dir=out_dir,
        model_name=model_name,
        prefix=prefix,
    )

    print("🎉 Chunked embedding pipeline completed successfully!")
