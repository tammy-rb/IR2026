import os
import glob
import json
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import numpy as np

LEMMA_DIR = "lemmas"
VECTORS_DIR = "vectors"


def load_lemmatized_documents(lemma_dir=LEMMA_DIR):
    """
    Load all lemmatized documents from the specified directory.
    
    Args:
        lemma_dir (str): Directory containing lemmatized .txt files
        
    Returns:
        tuple: (docs, filenames) where docs is a list of document texts
               and filenames is a list of corresponding filenames
    """
    docs = []  # each entry is the text of a document
    filenames = []  # corresponding filenames
    
    for path in sorted(glob.glob(os.path.join(lemma_dir, "*.lemma.txt"))):
        filenames.append(os.path.basename(path)) 
        with open(path, encoding="utf-8") as f:
            docs.append(f.read())
    
    print(f"Loaded {len(docs)} documents from {lemma_dir}")
    return docs, filenames


def build_tf_matrix(docs, min_df=5, max_df=0.8):
    """
    Build TF (Term Frequency) matrix from documents using CountVectorizer.
    
    Args:
        docs (list): List of document texts
        min_df (int): Minimum document frequency for terms
        max_df (float): Maximum document frequency for terms (as ratio)
        
    Returns:
        tuple: (tf_matrix, vectorizer) where tf_matrix is sparse TF matrix
               and vectorizer is the fitted CountVectorizer
    """
    vectorizer = CountVectorizer(
        input="content",
        analyzer="word",
        stop_words="english",
        min_df=min_df,
        max_df=max_df
    )
    
    # learn vocabulary and create TF sparse matrix (csr format)
    tf_matrix = vectorizer.fit_transform(docs)
    
    print(f"Built TF matrix with shape: {tf_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return tf_matrix, vectorizer


def save_tf_results(tf_matrix, vectorizer, filenames, vectors_dir=VECTORS_DIR):
    """
    Save TF matrix, filenames, and vocabulary to files.
    
    Args:
        tf_matrix: Sparse TF matrix
        vectorizer: Fitted CountVectorizer with vocabulary
        filenames (list): List of document filenames
        vectors_dir (str): Directory to save results
    """
    os.makedirs(vectors_dir, exist_ok=True)
    
    # save sparse TF matrix in .npz format 
    sparse.save_npz(os.path.join(vectors_dir, "sparse_TF_matrix.npz"), tf_matrix)
    
    # save filenames
    with open(os.path.join(vectors_dir, "filenames.json"), "w") as f:
        json.dump(filenames, f)
    
    # save vocabulary, mapping term -> column index
    with open(os.path.join(vectors_dir, "vocabulary.json"), "w") as f:
        json.dump(vectorizer.vocabulary_, f)
    
    print(f"Saved TF matrix and metadata to {vectors_dir}")

def bm25_weight(tf_csr, k1=1.6, b=0.75):
    """
    Calculate BM25 weights from TF matrix using Okapi BM25 formula.
    
    Args:
        tf_csr: Sparse TF matrix in CSR format
        k1 (float): BM25 parameter k1 (term frequency saturation)
        b (float): BM25 parameter b (document length normalization)
        
    Returns:
        sparse.csr_matrix: BM25 weighted sparse matrix
    """
    tf_csr = tf_csr.tocsr().astype(float)
    N, n_terms = tf_csr.shape

    # Document frequency for each term
    df = np.bincount(tf_csr.indices, minlength=n_terms)

    # BM25 IDF calculation
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

    # Document length (sum of tf in each document)
    dl = np.asarray(tf_csr.sum(axis=1)).ravel()
    avgdl = dl.mean()

    rows, cols = tf_csr.nonzero()
    data = tf_csr.data
    new_data = np.empty_like(data)

    for i in range(len(data)):
        row = rows[i]
        col = cols[i]
        freq = data[i]
        denom = freq + k1 * (1 - b + b * dl[row] / avgdl)
        new_data[i] = idf[col] * (freq * (k1 + 1)) / denom

    bm25_matrix = sparse.csr_matrix((new_data, (rows, cols)), shape=tf_csr.shape)
    
    print(f"Built BM25 matrix with shape: {bm25_matrix.shape}")
    print(f"BM25 parameters: k1={k1}, b={b}")
    
    return bm25_matrix


def save_bm25_matrix(bm25_matrix, vectors_dir=VECTORS_DIR):
    """
    Save BM25 matrix to file.
    
    Args:
        bm25_matrix: Sparse BM25 matrix
        vectors_dir (str): Directory to save the matrix
    """
    os.makedirs(vectors_dir, exist_ok=True)
    sparse.save_npz(os.path.join(vectors_dir, "bm25_okapi.npz"), bm25_matrix)
    print(f"Saved BM25 matrix to {vectors_dir}")


def build_sparse_vectors():
    """
    Main function to build both TF and BM25 sparse matrices.
    
    This function orchestrates the entire process:
    1. Load lemmatized documents
    2. Build TF matrix
    3. Save TF results
    4. Calculate BM25 weights
    5. Save BM25 matrix
    """
    print("Starting sparse vector construction...")
    
    # Load documents
    docs, filenames = load_lemmatized_documents()
    
    if not docs:
        print("No documents found! Make sure lemmatized files exist in the lemmas directory.")
        return
    
    # Build TF matrix
    tf_matrix, vectorizer = build_tf_matrix(docs)
    
    # Save TF results
    save_tf_results(tf_matrix, vectorizer, filenames)
    
    # Calculate and save BM25 matrix
    bm25_matrix = bm25_weight(tf_matrix)
    save_bm25_matrix(bm25_matrix)
    
    print("TF and BM25 sparse matrices have been built and saved successfully!")


if __name__ == "__main__":
    build_sparse_vectors()