# Exercise 3: 

tammy please notice after the top k- maybe add 
functionality to return by threshold. not by k.
so if there no matching- will not give unecessary context (confusion)

### stage 01: Document Chunking

## Overview

In the context of the assignment, our group was randomly assigned the **semantic chunking** method as the primary chunking strategy to study in depth. Instead of cutting documents purely by size, semantic chunking tries to respect natural meaning boundaries in the debates: it groups together sentences that “belong to the same idea” and separates chunks when the topic, argument, or speaker changes. In the following section we describe this method in detail and later compare it empirically to a simpler fixed-size chunking baseline.

This stage implements two document chunking strategies for parliamentary debates:
- **Fixed Chunking**: Size-based segmentation with sentence overlap
- **Semantic Chunking**: Embedding-based segmentation that respects meaning boundaries

## Semantic Chunking (Embedding-Based Segmentation)

### Method

In this project we implemented semantic chunking, an embedding-based segmentation method that aims to split long documents into coherent chunks according to meaning, rather than by a fixed size alone.

Intuitively, the goal is to keep together sentences that a human reader would naturally read as a single unit of argument or discussion, and to cut the text only when the debate “moves on” to a different point. Instead of saying “every 660 words is a new chunk”, we let sentence-level embeddings and cosine similarity decide where the topic really changes. This is especially important in parliamentary debates, where speakers often develop the same argument over several sentences and only then switch to a new issue (for example, from fuel prices to foreign policy). Semantic chunking tries to capture these shifts automatically so that each chunk is both internally coherent and useful for downstream retrieval.

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

python s_01_chuncking.py

### Stage 02: Vector Representations and Retrieval Indexes (BM25 + Dense Embeddings)

## Overview

After producing chunks with both fixed-size and semantic chunking, we build two types of retrieval indexes:

* **Sparse retrieval (BM25 / Okapi)**: Lexical matching based on term frequency and inverse document frequency
* **Dense retrieval (OpenAI embeddings + FAISS)**: Semantic matching using vector embeddings and approximate nearest neighbor search

This allows us to compare sparse vs. dense retrieval under the same chunking setup (fixed vs. semantic), and later combine them into a RAG pipeline.

## Sparse Retrieval with BM25 (Okapi)

### Method

We implement BM25 by first building a term-frequency (TF) matrix over the chunk texts using `CountVectorizer`. This results in a sparse matrix of shape `(N_chunks × V_vocab)`. We then convert this TF matrix into an Okapi BM25-weighted matrix using the standard BM25 formula with parameters k and b.

At query time, we vectorize the query into the same vocabulary space and compute BM25 scores via a sparse dot product:

score(chunk) = BM25(chunk) · TF(query)

### Configuration

* **BM25_K1** = 1.6
* **BM25_B** = 0.75
* **Vectorizer settings**:

  * `stop_words` = "english"
  * `min_df` = 2
  * `max_df` = 0.85

### Output Artifacts

For each chunking strategy (fixed / semantic) we save:

* `bm25_okapi.npz`: Sparse BM25 matrix
* `vocabulary.json`: Token → column index mapping
* `meta.json`: Metadata per chunk (file path, offsets, etc.)
* `vectorizer_config.json`: BM25 + vectorizer parameters for reproducibility

## Dense Retrieval with OpenAI Embeddings + FAISS

### Method

To support semantic retrieval, we embed each chunk using OpenAI's embedding model:

* **Model**: `text-embedding-3-large`

Each chunk is represented as a dense vector, and we build a FAISS index over these vectors for fast similarity search. During retrieval, the user query is embedded using the same model, and FAISS returns the top-K nearest chunks.

### Output Artifacts

For each chunking strategy we save a FAISS index directory (e.g., `fixed_faiss/`, `semantic_faiss/`) which includes:

* The FAISS index file(s)
* A docstore mapping vector IDs to chunk metadata (including source file and offsets)

### Note on Memory Error and the Fix

#### What happened

When embedding a large collection of chunks (≈ 87k), the initial implementation attempted to create the FAISS index by calling:

FAISS.from_documents(all_docs, embeddings)


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

Concretely, our dense representation is based on the **`text-embedding-3-large`** model from OpenAI. This model was chosen because it is specifically optimized for semantic search and clustering, provides high-dimensional embeddings that work well on long political texts, and is directly supported in LangChain’s FAISS integration. Compared to older embedding models, it offers better cross-sentence semantic coherence and multilingual robustness, which are desirable properties when working with complex parliamentary debates and queries that sometimes paraphrase the original wording.

#### Why FAISS + OpenAI Embeddings?

* **FAISS**: Provides efficient nearest-neighbor search in high-dimensional vector spaces and allows us to retrieve the top-K most semantically similar chunks for a given query
* **OpenAI Embeddings**: Selected because our RAG pipeline is implemented within the LangChain framework, which offers mature, well-supported integration with both OpenAI embedding models and FAISS vector stores
* **Benefits**: This choice ensures a stable, scalable, and easily extensible semantic retrieval component that integrates seamlessly with downstream LLM-based answer generation


### Stage 03: RAG Pipeline and LLM

#### LLM Choice and RAG Setup

For the answer-generation step we used an OpenAI GPT-4–class chat model (via the Chat Completions API). We chose this model because:

* It supports a sufficiently large context window to accept multiple retrieved chunks together with the user query.
* It is well suited for instruction-following and question-answering tasks, producing relatively focused, well-structured answers.
* It integrates smoothly with LangChain, which we already used for embeddings and vector stores, so the RAG pipeline could be implemented with minimal “glue code”.

In all experiments the RAG pipeline followed the same pattern:

1. **Query encoding**
   The user query (factual or conceptual) is sent either to the BM25 index (sparse retrieval) or to the FAISS index (dense retrieval via `text-embedding-3-large`), depending on the pipeline under test.

2. **Top-K retrieval**
   The retrieval layer returns the top-K most relevant chunks for that configuration:

   * Fixed chunking + BM25 (`fixed_bm25`)
   * Fixed chunking + dense embeddings (`fixed_dense`)
   * Semantic chunking + BM25 (`semantic_bm25`)
   * Semantic chunking + dense embeddings (`semantic_dense`)

3. **LLM answering**
   The LLM receives:

   * The original user query.
   * The retrieved chunks (as contextual “sources”).
   * A prompt that instructs it to answer *only* based on the provided chunks and, whenever possible, to cite sources using square brackets (e.g. `[source 1]`).

By keeping the LLM fixed and varying only the retrieval configuration (chunking strategy, representation type, and K), we can attribute differences in answer quality to the retrieval layer rather than to the generative model itself.

#### Choice of K (Top-K Chunks)

To study the trade-off between precision and recall in the retrieval stage, we ran each pipeline with several values of K:

K ∈ {1, 2, 3, 5, 10, 15, 20}


The intuition is that:

* Small K (e.g., 1–3) forces the retriever to be very selective: it returns only one or a few chunks, which should ideally be “the” correct location in the corpus.
* Large K (e.g., 10–20) increases the chance that relevant chunks are included somewhere in the list, but also introduces more noise (irrelevant chunks) into the context shown to the LLM.

The aggregate results reflect this trade-off clearly:

* For **K = 2**, we obtained relatively high average file precision (≈ **0.39**) but only moderate recall (≈ **0.56**).
* As **K** grows, recall increases monotonically while precision decreases. For example, at **K = 20** the average file recall reaches ≈ **0.84**, but precision drops to ≈ **0.08**.

In practice, this suggests that:

* **Small K (1–3)** is appropriate for narrow factual questions where we expect a single short answer and want to minimize noise.
* **Medium K (5–10)** gives a good balance between finding the right files and avoiding too much irrelevant context.
* **Large K (15–20)** is more suitable for broad conceptual questions, where the answer may be spread across multiple parts of a debate and we are willing to tolerate more noise.


### Stage 04: Evaluation and Analysis

#### Evaluation Setup

To analyse the behaviour of the different RAG pipelines we used the script `s_04_analyze_results.py`. This script loads all JSON results from `outputs/rag_runs/`, computes file-level precision and recall for each query (based on the expected source files), aggregates metrics across different conditions (chunking strategy, representation type, K value, and query type), and writes a human-readable summary as well as machine-readable JSON files (`metrics.json`, `aggregates.json`, `comparisons.json`, `examples.json`).

* **File-level precision**: among the files actually retrieved by the pipeline, how many belong to the expected gold list for that query.
* **File-level recall**: among the expected gold files for that query, how many were retrieved at all.
* **Answer-level features**: answer length and whether the LLM output contains citations (detected heuristically via `[]`).

These aggregates can be visualised using bar plots or histograms (e.g., in a Jupyter notebook with Seaborn or matplotlib) to show how precision and recall change as a function of K, chunking strategy, and representation type.

#### Fixed vs. Semantic Chunking

We first compare fixed-size chunking to the semantic chunking method that we implemented:

* **Fixed chunking**

  * Average file precision: ≈ **0.25**
  * Average file recall: ≈ **0.65**

* **Semantic chunking**

  * Average file precision: ≈ **0.21**
  * Average file recall: ≈ **0.73**

In other words:

* Fixed chunks yield **slightly higher precision**: fewer retrieved chunks come from completely wrong files.
* Semantic chunks yield **higher recall**: for a given K, the retriever is more likely to hit at least one correct file.

This matches the intuition behind the methods. Semantic chunks tend to be longer and follow meaning shifts more closely, so when the system lands on the right region it often captures multiple relevant sentences at once, improving recall. However, these larger chunks may also include surrounding, non-essential text, which explains the modest drop in precision compared to the “tighter” fixed-size chunks.

#### BM25 vs Dense Embeddings

Next, we compare sparse BM25 retrieval with dense retrieval based on OpenAI embeddings:

* **BM25 (sparse)**

  * Average file precision: ≈ **0.25**
  * Average file recall: ≈ **0.76**

* **Dense embeddings (FAISS + `text-embedding-3-large`)**

  * Average file precision: ≈ **0.21**
  * Average file recall: ≈ **0.61**

Surprisingly, in this particular setup BM25 **outperforms** dense retrieval on both precision and recall. A plausible explanation is that the queries from the assignment are often phrased very similarly to the original debate text (e.g., specific program names, exact fuel prices, explicit references to particular plans). In such cases the lexical overlap between query and document is high, which is exactly where BM25 excels. Dense embeddings are more advantageous when there is significant paraphrasing or when the query expresses a concept that is not literally spelled out in the same words in the corpus; here, this advantage is less pronounced.

It is also possible that some dense-retrieval hyper-parameters (e.g., FAISS index type, vector normalisation, or K for re-ranking) were not fully optimised, leaving room for future improvement on the dense side.

#### Factual vs Conceptual Questions

The assignment distinguishes between **factual** queries (seeking a concrete fact) and **conceptual** queries (asking about arguments, positions, or moral evaluations). We therefore split our analysis by `query_type` and by representation:

* For **BM25**:

  * Factual queries: file precision ≈ **0.34**
  * Conceptual queries: file precision ≈ **0.16**

* For **dense embeddings**:

  * Factual queries: file precision ≈ **0.23**
  * Conceptual queries: file precision ≈ **0.19**

The results show that BM25 is particularly strong on factual questions, where the target information is typically expressed with almost the same wording as in the query (numbers, names, dates, explicit mentions). For conceptual questions, precision drops significantly for both methods, but the gap between factual and conceptual is smaller on the dense side, suggesting that embeddings are somewhat better at handling paraphrase and high-level descriptions.

#### Did We Retrieve the Relevant Files and Chunks?

At the global level, the overall average file recall is ≈ **0.69**, meaning that in the majority of runs at least some of the expected files were retrieved. According to the analysis:

* **Perfect retrievals** (all expected files retrieved): **198** runs.
* **Complete failures** (no expected file retrieved): **90** runs.

This confirms that the system is far from perfect: while many queries are handled very well, there is still a non-trivial fraction where the retriever completely misses the gold documents.

We did not manually annotate relevance at the *chunk* level (the field `relevant_chunks` in `RetrievalMetrics` remains 0), but we can reason about chunk behaviour indirectly via K:

* For **small K** (1–3), precision is relatively high, but recall is limited. This suggests that the first chunk or two often come from the correct file, but additional relevant chunks from the same file might not be retrieved.
* For **large K** (10–20), file recall becomes very high (up to ≈ **0.84**), so it is likely that most relevant chunks are present somewhere in the retrieved list. However, precision is low (down to ≈ **0.08**), which means that many retrieved chunks are only weakly relevant or completely off-topic.

In practical terms:

* **Not all relevant chunks are retrieved** when K is small.
* **Not all retrieved chunks are relevant** when K is large.

A manual spot-check of several factual queries (e.g., those about fuel prices and specific policy proposals) showed that when the correct file was retrieved, the LLM’s answer was usually faithful to the retrieved text, including numbers and names. In failure cases where no correct file was returned, the LLM tended to either give a very generic answer or hallucinate plausible-sounding details that did not appear in the corpus.

#### Insights and Possible Improvements

From the comparisons above we draw several conclusions:

1. **Chunking strategy matters**: semantic chunking improves recall but slightly hurts precision relative to fixed-size chunking. This suggests that a hybrid strategy (e.g., semantic boundaries combined with a maximum size) could be worth exploring.

2. **BM25 is very competitive for this task**: because many queries are lexically close to the source text, BM25 performs surprisingly well, even better than dense embeddings in our current configuration. In a different domain with more paraphrase, we might expect the opposite.

3. **Factual vs conceptual queries behave differently**: factual questions benefit strongly from BM25, while conceptual questions remain challenging for both methods, with dense embeddings showing a small advantage.

4. **Top-K alone is not ideal**: currently the retriever always returns K chunks, even if all similarity scores are very low. As hinted in the exercise comment at the top of this file, a **threshold-based** stopping criterion (e.g., “only return chunks whose score is above τ”) could prevent us from passing completely irrelevant context to the LLM and reduce confusion in low-similarity cases.

Overall, the experiments highlight the importance of tuning the retrieval layer (chunking, representation, K, and potentially score thresholds) to the specific query distribution and corpus characteristics, rather than assuming that a single “default” configuration will work optimally in all scenarios.
