import os
import glob
import json
import numpy as np
import spacy

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# === Directories (adjust if your folder names are different) ===
LEMMA_DIR = "lemmas"          # directory with lemma files (xxx.lemma.txt)
CLEAN_DIR = "clean_docs"      # directory with cleaned text files (xxx.clean.txt)

VECTORS_W2V_LEMMAS_DIR = "vectors/word2vec_lemmas"
VECTORS_W2V_CLEAN_DIR = "vectors/word2vec_words"

# Load spaCy English model (make sure you ran: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# ====================== Document loading ======================

def load_documents(doc_dir, file_pattern):
    """
    Load all documents from the specified directory matching the pattern.

    Args:
        doc_dir (str): Directory containing document files
        file_pattern (str): File pattern to match
                           (e.g., "*.lemma.txt" or "*.clean.txt")

    Returns:
        tuple: (docs, filenames)
            docs       - list of document texts
            filenames  - list of corresponding filenames
    """
    docs = []
    filenames = []

    for path in sorted(glob.glob(os.path.join(doc_dir, file_pattern))):
        filenames.append(os.path.basename(path))
        with open(path, encoding="utf-8") as f:
            docs.append(f.read())

    print(f"Loaded {len(docs)} documents from {doc_dir}")
    return docs, filenames


def load_lemmatized_documents(lemma_dir=LEMMA_DIR):
    """Load all lemmatized documents (*.lemma.txt)."""
    return load_documents(lemma_dir, "*.lemma.txt")


def load_clean_documents(clean_dir=CLEAN_DIR):
    """Load all clean documents (*.clean.txt)."""
    return load_documents(clean_dir, "*.clean.txt")


# ====================== Tokenization & cleaning for Word2Vec ======================

def tokenize_for_embeddings(text):
    """
    Tokenize text for Word2Vec / Gensim using spaCy.

    Requirements from the assignment:
    - No punctuation
    - No numbers, digits, dates
    - Remove English stop-words
    - Everything in lowercase
    """
    doc = nlp(text)
    tokens = []

    for token in doc:
        # Skip punctuation, spaces, numbers, symbols
        if token.is_punct or token.is_space or token.like_num:
            continue

        tok = token.text.lower()

        # Skip stopwords
        if tok in ENGLISH_STOP_WORDS:
            continue

        tokens.append(tok)

    return tokens


# ====================== Train Word2Vec model ======================

def train_word2vec(tokenized_docs,
                   vector_size=100,
                   window=5,
                   min_count=5,
                   workers=4,
                   sg=1,
                   epochs=10):
    """
    Train a Word2Vec model on the given tokenized documents.

    Args:
        tokenized_docs (list of list of str): tokenized documents
        vector_size (int): dimensionality of word vectors
        window (int): context window size
        min_count (int): ignore words with total frequency < min_count
        workers (int): number of worker threads
        sg (int): 1 for Skip-gram, 0 for CBOW
        epochs (int): number of training epochs

    Returns:
        gensim.models.Word2Vec: trained model
    """
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg
    )
    # Extra training pass (optional, but keeps your previous code behaviour)
    model.train(tokenized_docs, total_examples=len(tokenized_docs), epochs=epochs)
    print(f"Finished training Word2Vec. Vocab size: {len(model.wv)}")
    return model


# ====================== Build document vectors ======================

def document_vector(model, tokens):
    """
    Build a document vector by averaging word vectors of tokens in the document.

    If no token from the document appears in the model vocabulary,
    return an all-zero vector.

    Args:
        model: trained Word2Vec model
        tokens (list of str): tokens of a document

    Returns:
        np.ndarray of shape (vector_size,)
    """
    # Keep only tokens that are in the model vocabulary
    vectors = [model.wv[t] for t in tokens if t in model.wv]

    if not vectors:
        # If there are no words from the document in the model, return zero vector
        return np.zeros(model.vector_size, dtype=np.float32)

    # Average the word vectors
    return np.mean(vectors, axis=0)


def build_document_vectors(model, tokenized_docs):
    """
    Build document vectors for all documents.

    Args:
        model: trained Word2Vec model
        tokenized_docs (list of list of str): tokenized documents

    Returns:
        np.ndarray of shape (n_docs, vector_size)
    """
    print("Building document vectors...")
    doc_vecs = np.vstack([
        document_vector(model, tokens)
        for tokens in tokenized_docs
    ])
    print(f"Document vectors shape: {doc_vecs.shape}")
    return doc_vecs


# ====================== Save results ======================

def save_word2vec_results(model, doc_vectors, filenames, out_dir):
    """
    Save Word2Vec KeyedVectors (wv), document vectors, and filenames.

    Args:
        model: trained Word2Vec model
        doc_vectors (np.ndarray): document vectors array
        filenames (list): list of document filenames
        out_dir (str): directory to save all data
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save only the KeyedVectors (wv), not the full model
    kv_path = os.path.join(out_dir, "word2vec.kv")
    model.wv.save(kv_path)

    # Save document vectors
    vecs_path = os.path.join(out_dir, "doc_vectors.npy")
    np.save(vecs_path, doc_vectors)

    # Save filenames (mapping between matrix row index and file name)
    filenames_path = os.path.join(out_dir, "filenames.json")
    with open(filenames_path, "w", encoding="utf-8") as f:
        json.dump(filenames, f)

    print(f"Saved Word2Vec KeyedVectors to: {kv_path}")
    print(f"Saved document vectors to: {vecs_path}")
    print(f"Saved filenames to: {filenames_path}")
    print(f"All Word2Vec-based outputs saved in: {out_dir}")


# ====================== Full pipeline for one dataset ======================

def build_w2v_for_dataset(doc_loader_func, out_dir, dataset_name):
    """
    Build Word2Vec document vectors for a specific dataset.

    Steps:
    1. Load documents
    2. Tokenize and clean for embeddings (spaCy)
    3. Train Word2Vec model
    4. Build document vectors (average of word vectors)
    5. Save KeyedVectors, doc vectors and filenames
    """
    print(f"\n=== Processing {dataset_name} dataset (Word2Vec) ===")

    # 1) Load documents
    docs, filenames = doc_loader_func()
    if not docs:
        print(f"No documents found for {dataset_name}!")
        return

    # 2) Tokenization and cleaning
    tokenized_docs = [tokenize_for_embeddings(text) for text in docs]
    print(f"Tokenized {len(tokenized_docs)} documents for {dataset_name}")

    # 3) Train Word2Vec model
    model = train_word2vec(tokenized_docs)

    # 4) Build document vectors
    doc_vectors = build_document_vectors(model, tokenized_docs)

    # 5) Save everything (wv + doc vectors + filenames)
    save_word2vec_results(model, doc_vectors, filenames, out_dir)

    print(f"{dataset_name} Word2Vec document vectors built and saved successfully!")


# ====================== main ======================

def build_word2vec_vectors():
    """
    Main function to build Word2Vec-based document vectors
    for both:
    1. Lemmatized documents  -> vectors/word2vec_lemmas/
    2. Clean documents       -> vectors/word2vec_words/
    """
    print("Starting Word2Vec document vector construction for all datasets...")

    # 1. Lemmas
    build_w2v_for_dataset(
        load_lemmatized_documents,
        VECTORS_W2V_LEMMAS_DIR,
        "Lemmatized"
    )

    # 2. Clean words
    build_w2v_for_dataset(
        load_clean_documents,
        VECTORS_W2V_CLEAN_DIR,
        "Clean"
    )

    print("\n🎉 All Word2Vec document vectors have been built and saved successfully!")


if __name__ == "__main__":
    build_word2vec_vectors()
