import os
import glob
import json
import numpy as np

from lxml import etree
from sentence_transformers import SentenceTransformer


# ============ Directories (adjust to your project) ============

RAW_DIR = "docs"  # folder with original XML files (e.g. debates2023-06-28d.xml, ...)
SIMCSE_OUT_DIR = "vectors/simcse_raw"  # where to save SimCSE document vectors

# You can change the model to any unsupervised SimCSE model you prefer
SIMCSE_MODEL_NAME = "princeton-nlp/unsup-simcse-bert-base-uncased"


# ============ Step 1: Load raw XML documents ============

def load_raw_xml_documents(raw_dir=RAW_DIR, pattern="*.xml"):
    """
    Load all XML documents from raw_dir and return:
    - texts: list of raw text extracted from each XML
    - filenames: corresponding list of XML filenames

    We keep the original file name so we can later map
    each embedding row back to the source file.
    """
    texts = []
    filenames = []

    for path in sorted(glob.glob(os.path.join(raw_dir, pattern))):
        base = os.path.basename(path)
        filenames.append(base)

        # Parse the XML and extract all text nodes
        tree = etree.parse(path)
        # This collects all text content from the XML
        all_texts = tree.xpath("//text()")

        # Join them into one large string for this document
        doc_text = " ".join(t.strip() for t in all_texts if t.strip())
        texts.append(doc_text)

    print(f"Loaded {len(texts)} raw XML documents from {raw_dir}")
    return texts, filenames


# ============ Step 2: Split documents into chunks/sentences ============

def simple_chunk_document(text, max_length=256):
    """
    Very simple document chunking:
    - Split by periods into rough "sentences".
    - Group several sentences together until we reach ~max_length tokens.
    - This is not perfect sentence segmentation, but it's simple and cheap.

    Reason:
    SimCSE (BERT-based) works best on reasonably short texts, not super-long documents.
    We then aggregate sentence/chunk embeddings to get a document-level vector.
    """
    # Rough sentence split
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    chunks = []
    current_chunk = []

    # Count tokens approximately by splitting on spaces
    current_len = 0
    for sent in sentences:
        sent_len = len(sent.split())
        # If adding this sentence would exceed max_length, start a new chunk
        if current_len + sent_len > max_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = sent_len
        else:
            current_chunk.append(sent)
            current_len += sent_len

    # Add the last chunk if any
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # If the document was empty or extremely short, ensure at least one chunk
    if not chunks and text.strip():
        chunks = [text.strip()]

    return chunks


def build_chunks_for_all_docs(texts, max_length=256):
    """
    Build a list of chunks (short texts) for each document.

    Returns:
    - all_doc_chunks: list of list of strings; each inner list holds chunks of one document
    """
    all_doc_chunks = []
    for i, text in enumerate(texts):
        chunks = simple_chunk_document(text, max_length=max_length)
        all_doc_chunks.append(chunks)
    print(f"Built chunks for {len(all_doc_chunks)} documents")
    return all_doc_chunks


# ============ Step 3: Encode chunks with SimCSE ============

def encode_doc_chunks(model, all_doc_chunks, batch_size=32):
    """
    Encode all document chunks using a SimCSE model.

    Args:
        model: a SentenceTransformer SimCSE model
        all_doc_chunks: list of list of strings (chunks per document)
        batch_size: encoding batch size

    Returns:
        doc_embeddings: np.ndarray of shape (n_docs, embedding_dim)
    """
    print("Encoding documents with SimCSE...")

    doc_embeddings = []
    for doc_idx, chunks in enumerate(all_doc_chunks):
        if not chunks:
            # If document has no chunks (empty), create a zero vector
            emb_dim = model.get_sentence_embedding_dimension()
            doc_embeddings.append(np.zeros(emb_dim, dtype=np.float32))
            continue

        # Encode all chunks of this document
        chunk_embeddings = model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Aggregate chunk embeddings to get a single document vector
        # Here: simple average, but you can also try max-pooling, etc.
        doc_vec = np.mean(chunk_embeddings, axis=0)
        doc_embeddings.append(doc_vec)

        if (doc_idx + 1) % 50 == 0:
            print(f"  Encoded {doc_idx + 1} / {len(all_doc_chunks)} documents")

    doc_embeddings = np.vstack(doc_embeddings)
    print(f"Finished encoding. Document embeddings shape: {doc_embeddings.shape}")
    return doc_embeddings


# ============ Step 4: Save results ============

def save_simcse_results(doc_embeddings, filenames, out_dir, model_name):
    """
    Save SimCSE document embeddings, filenames mapping, and the model name.

    - doc_embeddings.npy: numpy array of shape (n_docs, embedding_dim)
    - filenames.json: list of filenames in the same order as doc_embeddings rows
    - model_name.txt: text file storing the SimCSE model identifier
    """
    os.makedirs(out_dir, exist_ok=True)

    emb_path = os.path.join(out_dir, "doc_embeddings.npy")
    np.save(emb_path, doc_embeddings)

    fn_path = os.path.join(out_dir, "filenames.json")
    with open(fn_path, "w", encoding="utf-8") as f:
        json.dump(filenames, f)

    model_path = os.path.join(out_dir, "model_name.txt")
    with open(model_path, "w", encoding="utf-8") as f:
        f.write(model_name)

    print(f"Saved document embeddings to: {emb_path}")
    print(f"Saved filenames mapping to: {fn_path}")
    print(f"Saved model name to: {model_path}")
    print(f"All SimCSE outputs saved in: {out_dir}")


# ============ Step 5: Full pipeline function ============

def build_simcse_vectors_for_raw_xml():
    """
    Full SimCSE pipeline:

    1) Load raw XML documents (original debates files).
    2) Build text chunks (pseudo-sentences / paragraphs) per document.
    3) Load an unsupervised SimCSE model.
    4) Encode all chunks and aggregate to document-level embeddings.
    5) Save embeddings, filenames, and model name.
    """
    print("=== SimCSE document vector construction (on RAW XML) ===")

    # 1) Load raw XML
    texts, filenames = load_raw_xml_documents()

    if not texts:
        print("No raw XML documents found, aborting.")
        return

    # 2) Split documents into chunks
    all_doc_chunks = build_chunks_for_all_docs(texts, max_length=256)

    # 3) Load SimCSE model
    print(f"Loading SimCSE model: {SIMCSE_MODEL_NAME}")
    model = SentenceTransformer(SIMCSE_MODEL_NAME)

    # 4) Encode and aggregate
    doc_embeddings = encode_doc_chunks(model, all_doc_chunks, batch_size=32)

    # 5) Save everything
    save_simcse_results(doc_embeddings, filenames, SIMCSE_OUT_DIR, SIMCSE_MODEL_NAME)

    print("🎉 SimCSE document vectors for raw XML were built and saved successfully!")


if __name__ == "__main__":
    build_simcse_vectors_for_raw_xml()
