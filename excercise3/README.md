# Exercise 3: 

tammy please notice after the top k- maybe add 
functionality to return by threshold. not by k.
so if there no matching- will not give unecessary context (confusion)

### stage 01: Document Chunking

## Overview

This stage implements two document chunking strategies for parliamentary debates:
- **Fixed Chunking**: Size-based segmentation with sentence overlap
- **Semantic Chunking**: Embedding-based segmentation that respects meaning boundaries

## Semantic Chunking (Embedding-Based Segmentation)

### Method

In this project we implemented semantic chunking, an embedding-based segmentation method that aims to split long documents into coherent chunks according to meaning, rather than by a fixed size alone.

The process works as follows:

1. **Sentence Splitting**: Each document is split into full sentences while preserving their original character offsets
2. **Embedding Generation**: Each sentence is encoded into a dense semantic vector using a Sentence-Transformers (SBERT) model (`all-MiniLM-L6-v2`), which captures the contextual meaning in a high-dimensional embedding space
3. **Similarity Computation**: For every pair of adjacent sentences, we compute their cosine similarity
4. **Boundary Detection**: A semantic boundary is detected whenever the similarity between two consecutive sentences falls below a predefined threshold, indicating a meaningful topic or discourse shift
5. **Chunk Formation**: Chunks are formed by grouping consecutive sentences until a semantic boundary is reached, while also enforcing constraints:
   - Chunks contain only full sentences
   - Maximum 660 words per chunk (unless a single sentence is longer)
   - Minimum number of sentences per chunk to avoid very small chunks

### Similarity Threshold Selection

The similarity threshold was set to **0.62** based on empirical considerations and prior observations of cosine similarity distributions produced by SBERT embeddings.

**Rationale**:
- Values **above 0.62** typically indicate strong semantic continuity (e.g., elaboration or clarification of the same idea)
- Values **below 0.6** often correspond to topic changes, speaker shifts, or transitions between different arguments
- This threshold provides a balanced trade-off:
  - Conservative enough to prevent over-fragmentation into overly small chunks
  - Sensitive enough to capture genuine semantic shifts in parliamentary debates, which often evolve gradually rather than abruptly

This threshold supports the creation of chunks that are both semantically coherent and well-suited for downstream tasks such as retrieval, clustering, and classification.

## Configuration

### Fixed Chunking
- Max words per chunk: 660
- Overlap: 3 sentences between consecutive chunks

### Semantic Chunking
- Max words per chunk: 660
- Similarity threshold: 0.62
- Min sentences per chunk: 4
- Overlap: 0 sentences (optional)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

## Output

The script generates two JSONL files in `outputs/chunks/`:
- `chunks_fixed.jsonl`: Fixed-size chunks with sentence overlap
- `chunks_semantic.jsonl`: Semantically coherent chunks

Each chunk contains:
- Document ID and source path
- Corpus label (british/us)
- Chunking method
- Character offsets (start_char, end_char) (for retrieval)
- Chunk text
- Word count

### Note on Storage Format

Each chunk stores character-level offsets (`start_char`, `end_char`) pointing to its location in the original document. This pointer-based representation enables efficient reconstruction of the chunk text directly from the source file without re-running sentence splitting or chunking. 

The actual chunk text is also stored for convenience, debugging, and direct indexing in retrieval systems; however, it is optional and can be omitted in memory- or storage-constrained settings where only offsets are required.

## Usage

```bash
python s_01_chuncking.py
```

### Stage 02: Vector Representations and Retrieval Indexes (BM25 + Dense Embeddings)

## Overview

After producing chunks with both fixed-size and semantic chunking, we build two types of retrieval indexes:

- **Sparse retrieval (BM25 / Okapi)**: Lexical matching based on term frequency and inverse document frequency
- **Dense retrieval (OpenAI embeddings + FAISS)**: Semantic matching using vector embeddings and approximate nearest neighbor search

This allows us to compare sparse vs. dense retrieval under the same chunking setup (fixed vs. semantic), and later combine them into a RAG pipeline.

## Sparse Retrieval with BM25 (Okapi)

### Method

We implement BM25 by first building a term-frequency (TF) matrix over the chunk texts using `CountVectorizer`. This results in a sparse matrix of shape `(N_chunks × V_vocab)`. We then convert this TF matrix into an Okapi BM25-weighted matrix using the standard BM25 formula with parameters k and b.

At query time, we vectorize the query into the same vocabulary space and compute BM25 scores via a sparse dot product:

score(chunk) = BM25(chunk) · TF(query)

### Configuration

- **BM25_K1** = 1.6
- **BM25_B** = 0.75
- **Vectorizer settings**:
  - `stop_words` = "english"
  - `min_df` = 2
  - `max_df` = 0.85

### Output Artifacts

For each chunking strategy (fixed / semantic) we save:

- `bm25_okapi.npz`: Sparse BM25 matrix
- `vocabulary.json`: Token → column index mapping
- `meta.json`: Metadata per chunk (file path, offsets, etc.)
- `vectorizer_config.json`: BM25 + vectorizer parameters for reproducibility

## Dense Retrieval with OpenAI Embeddings + FAISS

### Method

To support semantic retrieval, we embed each chunk using OpenAI's embedding model:

- **Model**: `text-embedding-3-large`

Each chunk is represented as a dense vector, and we build a FAISS index over these vectors for fast similarity search. During retrieval, the user query is embedded using the same model, and FAISS returns the top-K nearest chunks.

### Output Artifacts

For each chunking strategy we save a FAISS index directory (e.g., `fixed_faiss/`, `semantic_faiss/`) which includes:

- The FAISS index file(s)
- A docstore mapping vector IDs to chunk metadata (including source file and offsets)

### Note on Memory Error and the Fix

#### What happened

When embedding a large collection of chunks (≈ 87k), the initial implementation attempted to create the FAISS index by calling:

```python
FAISS.from_documents(all_docs, embeddings)
```

This approach loads the entire dataset into memory and triggers embedding requests on very large batches. For large models such as `text-embedding-3-large`, each embedding response is high-dimensional, and sending too many texts per request can produce extremely large HTTP responses. This caused a `MemoryError` during response reading / buffering and resulted in a connection failure.

#### How we solved it

We replaced the "all-at-once" approach with a streaming + batching pipeline:

1. **Stream JSONL from disk**: Process one line at a time instead of loading all chunks into RAM
2. **Batch documents**: e.g., 256 chunks per batch
3. **Limit embedding request size**: Using `OpenAIEmbeddings(chunk_size=32)` so each API call handles a small number of texts
4. **Incrementally add vectors**: To the FAISS index using `vectorstore.add_documents(...)`
5. **Checkpoint saves**: Every N batches so progress is not lost

This reduces peak memory usage and avoids oversized API responses, making the embedding process stable for large corpora.

### Semantic Retrieval with FAISS and OpenAI Embeddings

#### Overview

To enable semantic (meaning-based) retrieval, we represent each document chunk as a dense vector using OpenAI's `text-embedding-3-large` model and index these vectors with FAISS. 

Unlike lexical methods such as BM25, this approach allows retrieval based on **semantic similarity** rather than exact word overlap, enabling the system to identify chunks that are conceptually relevant even when different wording is used.

#### Why FAISS + OpenAI Embeddings?

- **FAISS**: Provides efficient nearest-neighbor search in high-dimensional vector spaces and allows us to retrieve the top-K most semantically similar chunks for a given query
- **OpenAI Embeddings**: Selected because our RAG pipeline is implemented within the LangChain framework, which offers mature, well-supported integration with both OpenAI embedding models and FAISS vector stores
- **Benefits**: This choice ensures a stable, scalable, and easily extensible semantic retrieval component that integrates seamlessly with downstream LLM-based answer generation