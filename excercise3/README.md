# Exercise 3: Document Chunking, Vector Representations, and RAG Pipeline
---

## Table of Contents
1. [Stage 01: Document Chunking](#stage-01-document-chunking)
2. [Stage 02: Vector Representations and Retrieval Indexes](#stage-02-vector-representations-and-retrieval-indexes)
3. [Stage 03: RAG Pipeline and LLM](#stage-03-rag-pipeline-and-llm)
4. [Retrieval-Augmented Generation Analysis](#retrieval-augmented-generation-analysis)
5. [Conclusions & Insights](#conclusions--insights)

---

## Stage 01: Document Chunking

### Overview

In the context of the assignment, our group was randomly assigned the **semantic chunking** method as the primary chunking strategy to study in depth. Instead of cutting documents purely by size, semantic chunking tries to respect natural meaning boundaries in the debates: it groups together sentences that "belong to the same idea" and separates chunks when the topic, argument, or speaker changes.

This stage implements two document chunking strategies for parliamentary debates:

* **Fixed Chunking**: Size-based segmentation with sentence overlap
* **Semantic Chunking**: Embedding-based segmentation that respects meaning boundaries

---

### Semantic Chunking (Embedding-Based Segmentation)

#### Method

In this project we implemented semantic chunking, an embedding-based segmentation method that aims to split long documents into coherent chunks according to meaning, rather than by a fixed size alone.

**Intuition**: The goal is to keep together sentences that a human reader would naturally read as a single unit of argument or discussion, and to cut the text only when the debate "moves on" to a different point. Instead of saying "every 660 words is a new chunk", we let sentence-level embeddings and cosine similarity decide where the topic really changes.

**Why this matters**: This is especially important in parliamentary debates, where speakers often develop the same argument over several sentences and only then switch to a new issue (e.g., from fuel prices to foreign policy). Semantic chunking tries to capture these shifts automatically so that each chunk is both internally coherent and useful for downstream retrieval.

#### The Process

1. **Sentence Splitting**: Each document is split into full sentences while preserving their original character offsets
2. **Embedding Generation**: Each sentence is encoded into a dense semantic vector using a Sentence-Transformers (SBERT) model (`all-MiniLM-L6-v2`), which captures the contextual meaning in a high-dimensional embedding space
3. **Similarity Computation**: For every pair of adjacent sentences, we compute their cosine similarity
4. **Boundary Detection**: A semantic boundary is detected whenever the similarity between two consecutive sentences falls below a predefined threshold, indicating a meaningful topic or discourse shift
5. **Chunk Formation**: Chunks are formed by grouping consecutive sentences until a semantic boundary is reached, while also enforcing constraints:
   * Chunks contain only full sentences
   * Maximum 660 words per chunk (unless a single sentence is longer)
   * Minimum number of sentences per chunk to avoid very small chunks

#### Similarity Threshold Selection

**Threshold**: `0.62`

**Rationale**:

We chose 0.62 because it is a commonly used mid-range cosine-similarity cutoff for Sentence-Transformer embeddings, and it matched our data by separating clear discourse shifts without fragmenting coherent argument spans

* Values **above 0.62** → Strong semantic continuity (e.g., elaboration or clarification of the same idea)
* Values **below 0.6** → Topic changes, speaker shifts, or transitions between different arguments
* This threshold provides a balanced trade-off:
  * Conservative enough to prevent over-fragmentation into overly small chunks
  * Sensitive enough to capture genuine semantic shifts in parliamentary debates, which often evolve gradually rather than abruptly

This threshold supports the creation of chunks that are both semantically coherent and well-suited for downstream tasks such as retrieval, clustering, and classification.

---

### Configuration

#### Fixed Chunking
* **Max words per chunk**: 660
* **Overlap**: 3 sentences between consecutive chunks

#### Semantic Chunking
* **Max words per chunk**: 660
* **Similarity threshold**: 0.62
* **Min sentences per chunk**: 4
* **Overlap**: 0 sentences (optional)
* **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`

---

### Output

The script generates two JSONL files in `outputs/chunks/`:

* `chunks_fixed.jsonl`: Fixed-size chunks with sentence overlap
* `chunks_semantic.jsonl`: Semantically coherent chunks

**Each chunk contains**:

* Document ID and source path
* Corpus label (british/us)
* Chunking method
* Character offsets (`start_char`, `end_char`) for retrieval
* Chunk text
* Word count

#### Note on Storage Format

Each chunk stores character-level offsets (`start_char`, `end_char`) pointing to its location in the original document. This pointer-based representation enables efficient reconstruction of the chunk text directly from the source file without re-running sentence splitting or chunking.

The actual chunk text is also stored for convenience, debugging, and direct indexing in retrieval systems; however, it is optional and can be omitted in memory- or storage-constrained settings where only offsets are required.

---

### Usage

```bash
python s_01_chuncking.py
```

---

## Stage 02: Vector Representations and Retrieval Indexes

### Overview

After producing chunks with both fixed-size and semantic chunking, we build two types of retrieval indexes:

* **Sparse retrieval (BM25 / Okapi)**: Lexical matching based on term frequency and inverse document frequency
* **Dense retrieval (OpenAI embeddings + FAISS)**: Semantic matching using vector embeddings and approximate nearest neighbor search

This allows us to compare sparse vs. dense retrieval under the same chunking setup (fixed vs. semantic), and later combine them into a RAG pipeline.

---

### Sparse Retrieval with BM25 (Okapi)

#### Method

We implement BM25 by first building a term-frequency (TF) matrix over the chunk texts using `CountVectorizer`. This results in a sparse matrix of shape `(N_chunks × V_vocab)`. We then convert this TF matrix into an Okapi BM25-weighted matrix using the standard BM25 formula with parameters k and b.

**At query time**: We vectorize the query into the same vocabulary space and compute BM25 scores via a sparse dot product:

```
score(chunk) = BM25(chunk) · TF(query)
```

The BM25 score reflects how well a chunk matches the query in terms of exact term overlap, weighted by term importance and normalized for chunk length, making it effective for lexical retrieval but limited in capturing semantic similarity.

#### Configuration

* **BM25_K1** = 1.6
* **BM25_B** = 0.75
* **Vectorizer settings**:
  * `stop_words` = "english"
  * `min_df` = 2
  * `max_df` = 0.85

#### Output Artifacts

For each chunking strategy (fixed / semantic) we save:

* `bm25_okapi.npz`: Sparse BM25 matrix
* `vocabulary.json`: Token → column index mapping
* `meta.json`: Metadata per chunk (file path, offsets, etc.)
* `vectorizer_config.json`: BM25 + vectorizer parameters for reproducibility

---

### Dense Retrieval with OpenAI Embeddings + FAISS

#### Method

To support semantic retrieval, we embed each chunk using OpenAI's embedding model:

* **Model**: `text-embedding-3-large`

Each chunk is represented as a dense vector, and we build a FAISS index over these vectors for fast similarity search. During retrieval, the user query is embedded using the same model, and FAISS returns the top-K nearest chunks.

#### Output Artifacts

For each chunking strategy we save a FAISS index directory (e.g., `fixed_faiss/`, `semantic_faiss/`) which includes:

* The FAISS index file(s)
* A docstore mapping vector IDs to chunk metadata (including source file and offsets)

---

### Note on Memory Error and the Fix

#### ⚠️ What Happened

When embedding a large collection of chunks (≈ 87k), the initial implementation attempted to create the FAISS index by calling:

```python
FAISS.from_documents(all_docs, embeddings)
```

This approach loads the entire dataset into memory and triggers embedding requests on very large batches. For large models such as `text-embedding-3-large`, each embedding response is high-dimensional, and sending too many texts per request can produce extremely large HTTP responses. This caused a **MemoryError** during response reading/buffering and resulted in a connection failure.

#### ✅ How We Solved It

We replaced the "all-at-once" approach with a streaming + batching pipeline:

1. **Stream JSONL from disk**: Process one line at a time instead of loading all chunks into RAM
2. **Batch documents**: e.g., 256 chunks per batch
3. **Limit embedding request size**: Using `OpenAIEmbeddings(chunk_size=32)` so each API call handles a small number of texts
4. **Incrementally add vectors**: To the FAISS index using `vectorstore.add_documents(...)`
5. **Checkpoint saves**: Every N batches so progress is not lost

**Result**: This reduces peak memory usage and avoids oversized API responses, making the embedding process stable for large corpora.

Why Batching Improves Reliability

Instead of attempting to embed and index all chunks in a single operation, we process the corpus incrementally in batches. This design ensures that a failure in one embedding request affects only a small subset of chunks rather than the entire corpus. Each successfully processed batch is immediately added to the FAISS index and can be checkpointed to disk.

As a result, the pipeline becomes more reliable and fault-tolerant: partial progress is preserved, memory usage remains bounded, and transient API or network failures do not require restarting the entire embedding process. This approach is therefore much better suited for large corpora with tens of thousands of chunks, where all-at-once indexing is both memory-intensive and fragile.

---

### Semantic Retrieval with FAISS and OpenAI Embeddings

#### Overview

To enable semantic (meaning-based) retrieval, we represent each document chunk as a dense vector using OpenAI's `text-embedding-3-large` model and index these vectors with FAISS.

Unlike lexical methods such as BM25, this approach allows retrieval based on **semantic similarity** rather than exact word overlap, enabling the system to identify chunks that are conceptually relevant even when different wording is used.

#### Why This Model?

Our dense representation is based on the **`text-embedding-3-large`** model from OpenAI. This model was chosen because:

* **Optimized for semantic search**: Specifically designed for search and clustering tasks
* **High-dimensional embeddings**: Works well on long political texts
* **LangChain integration**: Directly supported in LangChain's FAISS integration
* **Better coherence**: Compared to older embedding models, it offers better cross-sentence semantic coherence and multilingual robustness

#### Why FAISS + OpenAI Embeddings?

* **FAISS**: Provides efficient nearest-neighbor search in high-dimensional vector spaces and allows us to retrieve the top-K most semantically similar chunks for a given query
* **OpenAI Embeddings**: Selected because our RAG pipeline is implemented within the LangChain framework, which offers mature, well-supported integration with both OpenAI embedding models and FAISS vector stores
* **Benefits**: This choice ensures a stable, scalable, and easily extensible semantic retrieval component that integrates seamlessly with downstream LLM-based answer generation

---

## Stage 03: RAG Pipeline and LLM

### LLM Choice and RAG Setup

For the answer-generation step, we used the **OpenAI gpt-4o-mini** chat model via the Chat Completions API.

#### Why This Model?

* ✅ Supports a sufficiently large context window to accept multiple retrieved chunks together with the user query
* ✅ Well suited for instruction-following and question-answering tasks, producing relatively focused, well-structured answers
* ✅ Integrates smoothly with LangChain, which we already used for embeddings and vector stores

Importantly, the LLM was kept fixed across all experiments. This ensures that observed differences in answer quality are attributable to the retrieval configuration (chunking strategy, vector representation, and Top-K) rather than to changes in the language model itself.

---

### RAG Pipeline Flow

In all experiments the RAG pipeline followed the same pattern:

#### 1. Query Encoding

The user query (factual or conceptual) is sent either to:
* The **BM25 index** (sparse retrieval), or
* The **FAISS index** (dense retrieval via `text-embedding-3-large`)

...depending on the pipeline under test.

#### 2. Top-K Retrieval

The retrieval layer returns the top-K most relevant chunks for that configuration:

* `fixed_bm25`: Fixed chunking + BM25
* `fixed_dense`: Fixed chunking + dense embeddings
* `semantic_bm25`: Semantic chunking + BM25
* `semantic_dense`: Semantic chunking + dense embeddings

#### Retrieval Scoring Mechanism

The ranking of chunks differs by retrieval method:

* **BM25 (Sparse Retrieval)**:  
  Chunks are ranked via a dot product between the query term-frequency (TF) vector and the BM25-weighted chunk vectors. This captures lexical overlap, with inverse document frequency (IDF) weighting and length normalization.

* **Dense Retrieval (FAISS + Embeddings)**:  
  Both the query and chunks are embedded into a dense vector space. Chunks are ranked by cosine similarity between the query embedding and the chunk embeddings, favoring semantically similar content even when exact wording differs.

In both cases, the **Top-K chunks with the highest scores** are selected and passed to the LLM as contextual evidence.

#### 3. LLM Answering

The LLM receives:

* The original user query
* The retrieved chunks (as contextual "sources")
* A prompt that instructs it to answer *only* based on the provided chunks and, whenever possible, to cite sources using square brackets (e.g. `[source 1]`)

**Key Point**: By keeping the LLM fixed and varying only the retrieval configuration (chunking strategy, representation type, and K), we can attribute differences in answer quality to the retrieval layer rather than to the generative model itself.

---

## Retrieval-Augmented Generation Analysis

### 1. Experimental Setup and Execution

All experiments were executed using:

```bash
python RAG_llm_runner.py --queries_json queries/given_queries.json --k1 3 --k2 5 --k3 10
python RAG_llm_runner.py --queries_json queries/queries.json       --k1 3 --k2 5 --k3 10 
```

**What was evaluated**:
* Four retrieval pipelines
* Three Top-K values (K ∈ {3, 5, 10})
* Outputs include the LLM's final answers and retrieval references

---

### 2. Evaluation Methodology (Two-Phase Strategy)
We apply a consistent two-phase evaluation methodology.

#### Phase A — Answer-Level Evaluation (Pre-Chunk Inspection)

In the first phase, we evaluate the system purely from an end-user perspective, based only on the LLM’s final answers, without inspecting the retrieved chunks.

This analysis provides a **high-level comparison of pipelines** and allows us to identify which retrieval configuration produces the most accurate and stable answers.

We evaluate:

* ❓ Whether the LLM is able to answer the query at all
* ✅ The correctness, completeness, and clarity of the answer
* 📊 Answer stability as the Top-K value increases
* 🔍 Sensitivity of each pipeline to increased K (i.e., robustness to retrieval noise)

Based on this phase, we determine the **best-performing pipeline at the answer level**.

---

#### Phase B — Evidence-Level Evaluation (Chunk-Level Inspection)

Only after identifying the best-performing pipeline in Phase A do we examine the retrieved chunks themselves.

The goal of this phase is to verify that the **retrieval layer provides the actual evidence** needed to support the LLM’s answers and to explain the qualitative behavior observed at the answer level.

To enable a focused yet informative analysis, **two representative queries were selected from each query set**:
* One **factual** query
* One **conceptual** query

These queries were analyzed in depth to capture different information needs and to study how retrieval quality and noise evolve as the Top-K value increases.

For each selected query:
* All retrieved chunks were manually labeled as **Directly Relevant**, **Supporting**, or **Irrelevant**
* The relevance distribution was examined for K ∈ {3, 5, 10}
* The impact of increasing K on evidence quality, redundancy, and noise was analyzed

This evidence-level inspection allows us to:
* Assess whether relevant evidence is retrieved at low K
* Understand how irrelevant context accumulates as K grows
* Explain why certain pipelines produce more stable and better-grounded answers
* Identify **practical K values** for factual versus conceptual queries

> **Important Note**  
> The optimal K values derived from this analysis are based on a limited number of representative queries. While this does not guarantee global optimality across all possible queries, the selected cases provide meaningful insight into retrieval behavior. A broader evaluation would require substantially more manual labeling and computational resources; therefore, this analysis is treated as a **qualitative but informative approximation** suitable for comparative evaluation.

**Chunk Relevance Labels**:

* **Directly Relevant**: Contains the exact information or core argument required to answer the query
* **Supporting**: Provides partial or contextual support for the answer
* **Irrelevant**: Unrelated or off-topic content

---
## Global Quantitative Summary (Both Query Sets)

Before analyzing each query set in depth (Phase A/B), we first report **global quantitative signals** aggregated by pipeline and K.  
These plots provide a compact view of answerability and retrieval noise, and help motivate the qualitative analyses that follow.

### 1) Fallback Rate (Answerability)

We compute the **fallback rate**: the percentage of queries for which the LLM returned the enforced fallback response (i.e., it could not answer using the retrieved context).  
Lower is better, but this metric reflects **answerability**, not necessarily **correctness**.

#### Fallback heatmap — Given Queries
![Fallback rate heatmap (given queries)](outputs/plots/fallback_rate_heatmap_20251223_070456.png)

#### Fallback heatmap — Second Query Set
![Fallback rate heatmap (second query set)](outputs/plots/fallback_rate_heatmap_20251223_070445.png)

**Interpretation.** Dense retrieval (FAISS + embeddings) shows near-zero fallback across K, indicating that semantically similar evidence is usually retrieved.  
However, a low fallback rate does not guarantee high-quality answers: larger fixed-size chunks can reduce fallback simply because each retrieved chunk contains more text, which may increase redundancy and answer dilution as K grows.  
Therefore, we use fallback rate as a supporting metric alongside qualitative answer evaluation (accuracy, focus, and stability across K).

### 2) Why We Chose Semantic Chunking + Dense Embeddings

The final pipeline choice was driven first and foremost by **manual answer-level analysis (Phase A)** across both query sets.  
We compared the *final LLM answers* produced by all four pipelines under K ∈ {3, 5, 10} and evaluated:

- ✅ **Correctness and completeness** of the answer (factual accuracy or a clear main argument)
- ✅ **Grounding** (the answer is supported by the retrieved context rather than generic summarization)
- ✅ **Stability across K** (answers remain consistent when more retrieved chunks are added)

Across both query sets, **Semantic Chunking + Dense Embeddings** produced the most **correct, well-grounded, and stable** answers for both factual and conceptual queries.

**Why this pipeline wins qualitatively:**
- **Semantic chunking** reduces mixed-topic chunks and creates coherent evidence units aligned with discourse boundaries.
- **Dense embeddings** retrieve semantically relevant evidence even when query phrasing differs from the debate text (robust to paraphrase and concept-level questions).

In contrast:
- **BM25 pipelines** often failed when relevant evidence was phrased differently than the query (lexical mismatch), leading to fallback or incomplete answers.
- **Fixed chunking + dense** often retrieved relevant content, but large fixed chunks and higher K introduced redundancy and unrelated subtopics, increasing the risk of answer dilution as K grows.

### 3) Evidence Quality vs. Noise (Chunk Labels): Validating the Choice and Selecting K

After selecting **Semantic Chunking + Dense Embeddings** as best from manual answer-level inspection, we validated this choice by inspecting retrieval evidence directly.  
To quantify retrieval quality beyond answerability, we label retrieved chunks as:
- **Directly Relevant**
- **Supporting**
- **Irrelevant**

> Note: The final pipeline selection is based on combined evidence from (1) qualitative Phase A/B analyses per query set and (2) these quantitative plots, which help explain and validate the observed behavior.

> Quantitative summary plots for both query sets are reported above.  
> Here we focus on qualitative answer-level behavior for this specific query set.

---

## Part I — First Query Set ("given_queries.json")

### 3. RAG Results Analysis – Given Queries (Pre-Chunk Inspection)

**Files reviewed**:
* `given_queries.json`
* `rag_given_queries_4pipelines_k3-5-10_20251222_170721.json`

> **Note**: In this phase, we evaluate only the *final LLM answers*. Retrieved chunks are **not** inspected yet. In particular, we observe whether the LLM provides a grounded answer or returns the enforced fallback response (“I don't know based on the retrieved chunks.”).

---

### Pipeline-Level Results (Answer-Level)

#### 1) Fixed Chunking + BM25 ❌

**Observed behavior**:
* The LLM frequently fails on factual precision tasks and often responds with  
  *“I don't know based on the retrieved chunks.”*
* Increasing K does not recover missing evidence; instead, it tends to increase noise and instability in the answers.

**Interpretation**:
Fixed-size chunks combined with lexical-only retrieval often fail to surface the exact evidence span needed for factual queries. When the correct information is not retrieved, the LLM correctly refuses to answer, indicating weak retrieval performance rather than generation failure.

**Conclusion**: Poor for factual extraction; not improved by higher K.

---

#### 2) Semantic Chunking + BM25 ⚠️

**Observed behavior**:
* Semantic chunking improves local coherence of retrieved context.
* However, the LLM still fails on queries that require semantic matching rather than exact term overlap, again producing fallback answers in several cases.

**Interpretation**:
While semantic chunking reduces mixed-topic chunks, it cannot compensate for the lexical limitations of BM25. If query phrasing does not closely match the text, relevant chunks may still not be retrieved.

**Conclusion**: Better coherence than fixed BM25, but semantic chunking alone cannot overcome lexical-only retrieval.

---

#### 3) Fixed Chunking + Dense Embeddings ⚠️

**Observed behavior**:
* Often produces correct answers already at K = 3, indicating strong recall.
* As K increases, answers become less precise and may include additional or loosely related information.

**Interpretation**:
Dense embeddings successfully retrieve semantically relevant content even when wording differs. However, fixed-size chunks are relatively large and may contain multiple subtopics. At higher K, this leads to redundancy and contextual dilution, which can affect answer precision.

**Conclusion**: Good recall; precision degrades as K grows.

---

#### 4) Semantic Chunking + Dense Embeddings ✅ **BEST**

**Observed behavior**:
* Produces the most correct, stable, and well-grounded answers across both factual and conceptual queries.
* K = 3 typically yields concise and accurate answers.
* K = 10 often introduces unnecessary context, leading to answer dilution.

**Interpretation**:
This pipeline benefits from both semantically coherent chunk boundaries and semantic retrieval. Relevant evidence is retrieved early in the ranking, and the provided context aligns closely with the answer required by the query.

**Conclusion**: Best overall pipeline; K should remain small unless additional nuance is required.

---

### 🏆 Best Configuration (Global – Given Queries)

**Pipeline**: Semantic Chunking + Dense Embeddings  
**Best K**: K = 3 (especially for factual queries)

At the answer level, this configuration provides the strongest balance between correctness, stability, and resistance to retrieval noise.

### 4. Chunk-Level Analysis – Given Queries (Best Pipeline)

This section presents an in-depth, evidence-level analysis for the **best-performing pipeline identified in Phase A**:  
**Semantic Chunking + Dense Embeddings**.

For each query type, we examine:
- The relevance of retrieved chunks
- The LLM’s final answer at different K values
- How retrieval noise affects answer precision and stability

---

#### 4.1 Factual Query: Prime Minister Defense Budget Speech Dates

**Query**:  
*"On what dates did the British Prime Minister deliver his speech on the defense budget?"*

**Pipeline**: Semantic Chunking + Dense Embeddings  
**K values analyzed**: 3, 5, 10

##### Chunk Distribution

| K  | Direct | Supporting | Irrelevant | Total |
|----|--------|------------|------------|-------|
| 3  | 1      | 1          | 1          | 3     |
| 5  | 1      | 1          | 3          | 5     |
| 10 | 1      | 1          | 8          | 10    |

![alt text](outputs/plots/semantic_dense_given_factual_stacked.png)

##### LLM Answers and Analysis

- **K = 3**  
  The LLM produces a concise, grounded answer:
  > *“The British Prime Minister delivered his speech on the defense budget on 30 October and in January at Lancaster House.”*  
  This answer is directly supported by the single **Directly Relevant** chunk and remains focused and precise.

- **K = 5**  
  The LLM gives essentially the **same answer**, despite additional retrieved context:
  > *“The British Prime Minister delivered his speech on the defense budget on 30 October and in January.”*  
  No new factual evidence is added; extra chunks are mostly irrelevant and do not improve answer quality.

- **K = 10**  
  The answer begins to show **signs of dilution**:
  > *“…on 30 October and in February, specifically at Lancaster House in January…”*  
  Here, the LLM blends temporal references from irrelevant or weakly related chunks, even though the number of directly relevant chunks has not increased.

##### Conclusion (Factual)

- ✅ The number of **Directly Relevant** chunks remains constant across all K values
- ❌ Increasing K introduces substantial noise without adding new factual evidence
- 🎯 **K = 3** provides the best balance of precision, stability, and interpretability

This confirms that **factual queries are precision-critical** and benefit from minimal, high-quality context.

---

#### 4.2 Conceptual Query: Immigration Bill – Main Argument

**Query**:  
*"What was the main argument regarding the immigration bill that was presented?"*

**Pipeline**: Semantic Chunking + Dense Embeddings  
**K values analyzed**: 3, 5

##### Chunk Distribution

| K | Direct | Supporting | Irrelevant | Total |
|---|--------|------------|------------|-------|
| 3 | 2      | 1          | 0          | 3     |
| 5 | 2      | 2          | 1          | 5     |

![alt text](outputs/plots/semantic_dense_given_conceptual_stacked.png)

##### LLM Answers and Analysis

- **K = 3**  
  The LLM provides a coherent summary of the core argument:
  > *“The main argument was that devolving immigration powers to the Scottish Government could support rural and island communities through economic migration, though concerns were raised about effectiveness and accountability.”*  
  This answer is supported by **two Directly Relevant** chunks and one Supporting chunk, which together contain both the proposal and its critique.

- **K = 5**  
  The LLM produces a **slightly richer explanation**, adding nuance:
  > *“…while migration can be beneficial, critics argued the bill lacked clarity, security, and accountability, and could create more uncertainty than solutions.”*  
  The additional Supporting chunk reinforces the critique without substantially increasing noise.

##### Conclusion (Conceptual)

- ✅ Conceptual questions benefit from **multiple reinforcing chunks**
- ✅ **K = 3** already contains sufficient evidence for a complete answer
- ✅ **K = 5** can add nuance and justification with only minor noise increase

Unlike factual queries, conceptual questions tolerate a modest increase in K because synthesis—not exact extraction—is required.

---

### Summary of Chunk-Level Findings (Given Queries)

- **Factual queries**:
  - Depend on a single “gold” chunk
  - Increasing K mainly adds irrelevant context
  - **Best K: 3**

- **Conceptual queries**:
  - Benefit from multiple aligned chunks
  - Moderate K increases can improve explanatory depth
  - **Best K: 3–5**

These observations explain the answer-level behavior seen in Phase A and motivate the task-dependent K recommendations used in the final system configuration.

---

## Part II — Second Query Set ("queries.json")

### 5. RAG Results Analysis – Second Query Set (Pre-Chunk Inspection)

**Files reviewed**:
* `queries.json`
* `rag_queries_4pipelines_k3-5-10_20251222_165745.json`

> **Note**: As before, chunk content is not inspected at this stage.

---

#### 1) Fixed Chunking + BM25 ❌

**Observed behavior**:

- ⚠️ **Factual queries** are answered correctly only when the relevant information appears verbatim in one of the retrieved chunks  
  *Example*: For the fuel price query, the LLM correctly reports  
  > “143.43p per litre for unleaded petrol and 145.6p per litre for diesel”  
  but this correctness persists unchanged across K = 3, 5, and 10, indicating that additional context does not contribute new evidence.

- ❌ Increasing K frequently introduces **redundant or unrelated parliamentary material**, without improving factual completeness.

- ❌ **Conceptual queries** often result in vague summaries, such as  
  > “concerns were raised” or “issues were discussed,”  
  without clearly identifying a central argument or causal reasoning.

**Conclusion**:  
This pipeline is highly sensitive to lexical overlap, offers limited abstraction, and shows weak stability as K increases. It performs poorly for conceptual understanding and does not benefit meaningfully from higher K.

---

#### 2) Semantic Chunking + BM25 ⚠️

**Observed behavior**:

- ⚠️ Semantic chunking improves **local coherence**, leading to slightly more focused answers compared to fixed chunking.

- ❌ However, answers still fail when **semantic inference or synthesis** is required.  
  For example, in queries asking *why* a policy is considered problematic, the LLM often omits key reasoning steps or policy implications.

- ❌ Some factual answers are **partially correct but incomplete**, suggesting that lexical retrieval alone remains a bottleneck even with improved chunk boundaries.

**Conclusion**:  
Semantic chunking alone cannot compensate for BM25’s lexical limitations. While coherence improves marginally, missing values and shallow reasoning remain common.

---

#### 3) Fixed Chunking + Dense Embeddings ⚠️

**Observed behavior**:

- ✅ Many **factual answers are already correct at K = 3**, indicating strong recall from dense retrieval.

- ⚠️ As K increases, answers tend to become **longer and less focused**, occasionally mixing relevant facts with loosely related contextual information.

- ⚠️ Large fixed chunks amplify this effect: additional retrieved chunks often repeat background discussion rather than adding new evidence.

**Conclusion**:  
This pipeline achieves good recall but suffers from declining precision as K grows. The lack of semantic chunk boundaries makes it vulnerable to redundancy and answer drift.

---

#### 4) Semantic Chunking + Dense Embeddings ✅ **BEST**

**Observed behavior**:

- ✅ Produces the **most grounded and consistent answers** across both factual and conceptual queries.

- ✅ At K = 3, answers are typically concise, accurate, and directly supported by retrieved evidence.  
  In multiple factual queries, the LLM provides exact values or names without unnecessary elaboration.

- ✅ For conceptual queries, the LLM successfully identifies a **clear main argument** and supporting rationale, rather than generic summaries.

- ⚠️ At K = 10, minor dilution appears: answers remain correct but may include less relevant contextual detail.

**Conclusion**:  
This pipeline provides the best trade-off between recall and noise control. Semantic chunking ensures coherent evidence units, while dense embeddings enable semantic matching beyond exact wording.

---

### 🏆 Best Configuration (Global)

**Pipeline**: Semantic Chunking + Dense Embeddings  
**Recommended K**:
- **K = 3** for factual queries (highest precision, minimal noise)
- **K = 3–5** for conceptual queries (allows reinforcement and nuance without significant degradation)

These conclusions are consistently supported by answer-level behavior across all evaluated queries.

---
### 6. Chunk-Level Analysis – Second Query Set (Best Pipeline)

This section analyzes the **LLM answers together with the retrieved evidence**, focusing on the best-performing pipeline identified earlier: **Semantic Chunking + Dense Embeddings**.  
For each query, we examine how the answer quality behaves as K increases and explicitly cite representative LLM outputs.

---

#### 6.1 Factual Query: Fuel Prices (Exact Per-Liter Values)

**Query**  
*“According to the debate on fuel prices, what were the exact per-liter prices of unleaded petrol and diesel as of Monday, 26 June?”*

**Pipeline**: Semantic Chunking + Dense Embeddings  
**K values**: 3, 5, 10

**Representative LLM Answers**

* **K = 3**  
  > “As of Monday, 26 June, the price of unleaded petrol was **143.43p per litre**, and diesel was **145.6p per litre**.”

* **K = 5**  
  > “As of Monday, 26 June, the price of unleaded petrol was **143.43p per litre**, and diesel was **145.6p per litre**.”

* **K = 10**  
  > “As of Monday, 26 June, the price of unleaded petrol was **143.43p per litre**, and diesel was **145.6p per litre**.”

Across all K values, the **numerical answer remains identical**, indicating that the LLM consistently relies on the same core evidence.

**Chunk Distribution**

| K  | Direct | Supporting | Irrelevant | Total |
|----|--------|------------|------------|-------|
| 3  | 1      | 1          | 1          | 3     |
| 5  | 1      | 2          | 2          | 5     |
| 10 | 1      | 2          | 7          | 10    |

![alt text](outputs/plots/semantic_dense_second_factual_stacked.png)

**Analysis (Factual)**

* ✅ The correct answer is already fully determined by a single “gold” chunk
* ➖ Increasing K does **not** add new factual content
* ❌ Larger K substantially increases irrelevant context, without improving correctness
* 🎯 **K = 3** provides the cleanest evidence-to-answer mapping

---

#### 6.2 Conceptual Query: Rwanda Plan (“Stop the Boats”) – Deterrence Criticism

**Query**  
*“What is the main criticism raised in the debate about the Rwanda plan in the context of ‘stop the boats’, and why is it argued that the plan would not act as a deterrent?”*

**Pipeline**: Semantic Chunking + Dense Embeddings  
**K values**: 3, 5

**Representative LLM Answers**

* **K = 3**  
  > “The main criticism is that the Rwanda plan would **not deter crossings** because it would affect only a very small number of migrants and would be **too slow and limited in scale** to change behavior.”

* **K = 5**  
  > “Critics argue that the Rwanda plan would **fail as a deterrent** since only a handful of people would ever be relocated, the process would be slow, and smugglers and migrants would **continue to see the risk as worthwhile**.”

The K=5 answer expands on the same argument with **additional justification**, but does not contradict or replace the core claim identified at K=3.

**Chunk Distribution**

| K | Direct | Supporting | Irrelevant | Total |
|---|--------|------------|------------|-------|
| 3 | 2      | 1          | 0          | 3     |
| 5 | 2      | 2          | 1          | 5     |

![alt text](outputs/plots/semantic_dense_second_conceptual_stacked.png)

**Analysis (Conceptual)**

* ✅ Core argument is already present and coherent at K = 3
* ✅ Additional chunks at K = 5 reinforce and enrich the explanation
* ⚠️ Small amount of noise appears, but does not disrupt the answer
* 🎯 **K = 3–5** offers the best balance between completeness and focus

---

### 7. Consolidated Evidence-Based Conclusions

**Best Pipeline**  
✅ **Semantic Chunking + Dense Embeddings**

**Best K (Task-Dependent)**

* **Factual queries** → **K = 3**  
  * One decisive chunk determines the answer  
  * Larger K adds noise without benefit

* **Conceptual queries** → **K = 3–5**  
  * Multiple reinforcing chunks improve reasoning  
  * Moderate K increases nuance while maintaining stability

Overall, the cited LLM answers demonstrate that **answer stability and grounding are primarily driven by retrieval quality and chunk coherence**, not by large K values.

---

#### 💡 Practical Recommendation

For accurate, stable, and interpretable RAG behavior:

> **Use Semantic Chunking + Dense Embeddings with K = 3**,  
> increasing to K = 5 only when conceptual synthesis requires additional supporting context.

---

## Conclusions & Insights (Answering the Assignment Questions)

### 1. Chunking Method (Assigned by Lottery): Semantic Chunking

In this project, our primary chunking strategy (assigned by lottery) was **semantic chunking**.

**What is it?**: Unlike fixed-size chunking, semantic chunking attempts to preserve meaningful discourse units by grouping consecutive sentences that are semantically coherent, and placing a boundary when the topic or argument shifts.

**Why it matters**: In parliamentary debates, this is especially important: speakers often build a single argument over multiple sentences and only later move to a different policy issue.

**Our results**:
* ✅ The pipeline **Semantic Chunking + Dense Embeddings** produced the most stable and best-grounded answers
* ✅ Chunk-level labeling showed that relevant argument or evidence tends to appear as a coherent unit already at low K
  * Example: Directly Relevant = 2 and Supporting = 1 for conceptual queries at K=3
* ❌ In contrast, fixed chunking frequently bundled multiple topics into one chunk, increasing the chance that retrieval returns partially related context that encourages answer drift

---
2. LLM Choice and Rationale

Model: OpenAI gpt-4o-mini

The LLM used in this project is an OpenAI-based model. We selected gpt-4o-mini as it is a well-established and sufficiently capable model for retrieval-augmented question answering, while remaining lightweight and efficient.

The choice was mainly motivated by practical and experimental considerations:

The model is fully integrated and easy to use within the LangChain framework, which was used throughout the project for both embeddings and RAG orchestration.

We had prior experience working with this model, which allowed us to focus on the retrieval and chunking experiments rather than on LLM configuration or prompt tuning.

The model provides reliable and consistent behavior when answering based on retrieved context, making it suitable for controlled comparisons between different retrieval pipelines.

Importantly, the LLM was kept fixed across all experiments. This ensures that observed differences in answer quality are a result of changes in the retrieval strategy (chunking method, vector representation, and Top-K) rather than differences in the language model itself.

---

### 3. Vector Representation Choice (BM25 vs Dense Embeddings)

We evaluated two fundamentally different representations:

#### BM25 (Sparse Lexical Retrieval)
* ✅ Strong when query terms overlap explicitly with the text
* ❌ Limited in semantic matching

#### Dense Embeddings (OpenAI + FAISS)
* ✅ Captures semantic similarity
* ✅ Retrieves relevant evidence even when wording differs

**Result**: **Dense embeddings were consistently superior** for our RAG setting.

**Example**: In the first query set, BM25 pipelines repeatedly failed to answer the factual query ("Prime Minister defense budget speech dates"), producing "I don't know based on the retrieved chunks," across all K values, while **dense pipelines produced correct answers already at low K**.

**Interpretation**: Lexical overlap alone was insufficient to locate the needed evidence, whereas semantic retrieval succeeded.

---

### 4. Why We Chose K = 3, 5, 10

We selected **K = 3, 5, 10** to explicitly study the classic **precision–recall trade-off** in RAG:

* **Small K (K=3)**: Tests whether the retriever can surface a minimal set of high-quality evidence (high precision)
* **Medium K (K=5)**: Tests whether moderate context expansion improves completeness without excessive noise
* **Larger K (K=10)**: Stresses the system under increased context and evaluates robustness to retrieval noise

**Key Finding**: **Increasing K does not guarantee better answers**.

**Evidence**:

For **factual queries**, Directly Relevant evidence remained constant while irrelevant chunks grew sharply:

| Query Type | K=3 | K=5 | K=10 |
|------------|-----|-----|------|
| Fuel Prices (Irrelevant) | 1 | 2 | 7 |

For **conceptual queries**, K=5 sometimes adds nuance, but K=10 generally increases risk of dilution.

---

### 5. Fixed-Size vs Semantic Chunking: Which Is Better?

**Answer**: **Semantic chunking outperformed fixed-size chunking**, especially when combined with dense retrieval.

**Evidence**:

**Answer-level**:
* ✅ Semantic + dense produced the most correct and stable answers across K

**Chunk-level**:
* ✅ Semantic + dense yielded higher proportions of Directly Relevant and Supporting chunks at low K
* ✅ Less irrelevant noise than fixed-size setups

**Why fixed chunking underperforms**:
* ❌ Fixed chunking can still work reasonably for dense retrieval, but it more often introduces extra unrelated context inside each chunk (because the chunk boundary is not aligned with argument structure)
* ❌ This increases the chance of answer drift as K grows

---

### 6. BM25 vs Dense Embeddings: Which Is Better?

**Answer**: **Dense embeddings were clearly better** for our tasks.

**Why BM25 often failed**:

* ❌ Paraphrased queries
* ❌ Queries referencing concepts rather than exact phrases
* ❌ Cases where the "right" chunk did not share enough surface vocabulary with the query

**Why dense embeddings succeeded**:

* ✅ Semantic similarity is robust to wording differences
* ✅ Dense pipelines often produced correct answers already at K=3
* ❌ BM25 pipelines either failed completely or produced incomplete answers

---

### 7. Did BM25 Perform Better on Factual Questions?

**Answer**: **No, BM25 did not perform better on factual questions** in our experiments.

**Why not?**:

* While BM25 can sometimes be strong for fact retrieval in settings where the query contains unique keywords that appear verbatim in the relevant text...
* ❌ Our factual questions often required retrieving evidence expressed differently than the query wording
* ❌ This mismatch caused BM25 to return topically adjacent but non-answering chunks

**In contrast**:

* ✅ Dense embeddings + semantic chunking consistently retrieved at least one "gold" chunk for factual questions at low K
* ✅ This allowed correct grounded answers

---

### 8. Were Relevant Chunks/Files Retrieved? Were All Retrieved Chunks Relevant?

Our chunk-level labeling provides evidence for both questions:

#### ✅ Were relevant chunks retrieved?

**Yes** — under Semantic Chunking + Dense Embeddings, relevant evidence was retrieved at low K for both factual and conceptual queries.

**Examples**:

* **Fuel Prices (factual)**: Direct evidence appears already at K=3 (Direct=1)
* **Rwanda Plan (conceptual)**: Multiple Directly Relevant chunks appear already at K=3 (Direct=2)

---

#### ⚠️ Were all relevant chunks retrieved?

**Not necessarily**, and this is expected.

**Why?**: Retrieval is ranked, and the goal of RAG is not to retrieve all relevant chunks, but to retrieve enough evidence to answer correctly.

* For **factual queries**: Often one gold chunk is sufficient
* For **conceptual queries**: Several reinforcing chunks help, but retrieving every relevant chunk becomes less important than preserving coherence

---

#### ❌ Were all retrieved chunks relevant?

**No**. Irrelevant chunks increase as K grows.

**Evidence**:

| Query | K=3 (Irrelevant) | K=5 (Irrelevant) | K=10 (Irrelevant) |
|-------|------------------|------------------|-------------------|
| Fuel Prices | 1 | 2 | 7 |
| Rwanda Plan | 0 | 1 | — |

**Impact**: This noise growth explains why larger K values can degrade answer quality: the LLM may generalize, merge topics, or dilute a precise conclusion when unnecessary context is added.

---

### 9. Alignment Between Answers and Retrieved Evidence

**When retrieval quality is high** (semantic + dense, low K):

* ✅ The LLM answers are strongly aligned with the retrieved evidence
* ✅ Factual answers are grounded in the gold chunk
* ✅ Conceptual answers synthesize consistent arguments repeated across multiple relevant chunks

**When misalignment occurs**:

* ❌ Mainly in high-noise settings (especially at K=10)
* ❌ Irrelevant chunks can introduce competing narratives or unrelated facts
* ❌ This encourages answer drift or overgeneralization

---

### 10. Why Do We See These Differences?

The observed differences are explained by an interaction of:

1. **Representation type** (lexical BM25 vs semantic embeddings)
2. **Chunk boundaries** (fixed vs discourse-aligned semantic chunking)
3. **Query type** (factual vs conceptual)
4. **Context size** (K controlling noise injection)

**Factual queries** are **precision-critical**: They typically depend on a single correct evidence span.

**Conceptual queries** are **synthesis-heavy**: They benefit from multiple supporting spans but tolerate moderate K increases.

---

## 🎯 Final Recommendations

### Best Overall Configuration

**✅ Semantic Chunking + Dense Embeddings with K = 3**

---

### When to Increase K

* **Factual queries**: Keep K = 3
* **Conceptual queries**: K = 3–5 acceptable
* **Avoid K > 10** unless specifically needed for broad exploratory queries

---

### 💡 Practical Improvement Suggestion: Threshold-Based Retrieval

A clear insight from our results is that **fixed Top-K retrieval can inject unnecessary noise when relevance is weak**.

#### The Problem

When K is fixed (e.g., K=10), the system *always* returns exactly 10 chunks, even when:
* Only 1–2 chunks are actually relevant
* The remaining chunks have low similarity scores
* Adding those chunks introduces noise and degrades answer quality

#### Proposed Solution

**Replace fixed Top-K with adaptive threshold-based retrieval**:

```python
return_by_threshold(similarity ≥ τ) instead of always returning K chunks
```

#### How It Works

* **High-confidence retrieval**: If multiple chunks exceed the similarity threshold τ, return all of them
* **Low-confidence retrieval**: If no chunk crosses the threshold, return fewer chunks (or none)
* **Honest failure mode**: When evidence is insufficient, the LLM responds with "not enough evidence" rather than hallucinating from irrelevant context

**Impact**: This would directly address the observed failure mode where K=10 adds many irrelevant chunks without adding new relevant evidence.

---

