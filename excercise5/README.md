# Exercise 5: Topic Modeling over News and Parliamentary Debates

This README documents **Step 1 of the project pipeline**: data cleaning, chunking, embedding, and exploratory topic modeling using **BERTopic** over precomputed embeddings stored in **Qdrant**. The goal of this step is not final topic production, but **understanding the structure of the data**, identifying noise, and converging on a configuration that yields meaningful, balanced topics before downstream RAG and temporal analysis.

---

## Pipeline Overview

1. **Corpus Cleaning**

   * News corpora (BBC / NBC)
   * Parliamentary corpora (US Congress, UK Parliament)
2. **Chunking Strategy** (news vs. debates)
3. **Embedding** (Sentence-BERT, stored in Qdrant)
4. **Exploratory BERTopic Training**

   * Fit on existing vectors
   * Diagnose clustering behavior
   * Reduce to 20 topics only after validation

---

## 1. Corpus Cleaning

### 1.1 BBC / NBC News Cleaning

Raw crawled news data contains a significant amount of **non-article content** that must be removed before embedding and topic modeling. Without cleaning, this material dominates the vector space and degrades both clustering and retrieval quality.

**Main issues addressed:**

* **NBC cookie-consent boilerplate**
  Many NBC pages are truncated by a long *"This Cookie Notice"* section, sometimes leaving only a headline.

  **Action:**
  When a cookie notice is detected, the file is truncated **in-place**, keeping only the text that appears before the notice.

* **BBC technical / error pages**
  Includes geo-blocking messages ("outside the UK"), closed live pages, and index/topic pages.

* **Non-news content**
  Podcasts, radio shows, TV program listings, and episode pages.

* **Duplicate articles (same day)**
  Identical content published multiple times under different filenames on the same date.

**Filtering rules:**

* Files are labeled as `VALID`, `JUNK`, or `NON_NEWS`.
* Minimum length thresholds (applied *after* cleaning):

  * BBC: 80 words
  * NBC: 30 words
* Duplicate detection is **same-day only**, using normalized-content hashing.

Short headline-only stubs are intentionally discarded: they provide insufficient semantic signal and behave like noise in embedding-based methods.

---

### 1.2 US Congress Cleaning (Critical for Topic Modeling)

For the US Congressional Record, we apply **additional structural cleaning** beyond basic text normalization.

**Key decision:**
All headers and technical material **before the actual speech text** are removed.

**Why this matters:**

Congressional documents share highly repetitive headers (e.g., *Congressional Record*, *Extensions of Remarks*, page numbers, metadata blocks). During early BERTopic runs, these headers became a **dominant semantic signal**, causing HDBSCAN to collapse most documents into a **single mega-cluster**.

This happens because:

* The header text is nearly identical across documents
* It appears at the beginning of every file
* It overwhelms the actual semantic differences between speeches

**Action:**

* Strip all content before the first actual speech body
* Ensure embeddings reflect **what is being discussed**, not document structure

This change alone prevents clustering from degenerating into a single dominant topic.

---

## 2. Chunking Strategy

### 2.1 News Articles (BBC / NBC)

* **No chunking applied**
* Each article file is treated as **one document → one vector**

**Rationale:**

* News articles are typically single-topic
* Internal coherence is high
* Chunking would fragment context without clear benefit

---

### 2.2 Parliamentary Debates

Parliamentary data is structurally different and requires chunking.

#### Initial Observation

* A single debate or speech often contains **multiple themes**
* Treating an entire speech as one vector mixes unrelated policy areas

#### Semantic Chunking

We use **semantic chunking** to split speeches based on topic change.

However, analysis revealed an important nuance:

* The semantic chunker correctly identified **rhetorical paragraph boundaries**
* But these boundaries did **not always correspond to analytically meaningful topic shifts**

In practice:

* A coherent parliamentary speech could be split into ~14 very short chunks
* These chunks reflected stylistic transitions, not substantive policy changes

#### Final Decision: Semantic Chunking + Controlled Merging

* We keep the semantic chunker **as-is** (threshold unchanged)
* We **merge consecutive chunks** up to a maximum word limit

**Goal:**
Each final chunk should represent **one coherent topic**, not rhetorical fragments.

---

## 3. Embedding and Vector Storage

### Embedding Model

* **Sentence-BERT: all-MiniLM-L6**

### Why we generate embeddings ourselves

Instead of letting BERTopic embed the text internally, we precompute embeddings and store them in Qdrant.

**Reasons:**

1. **Reusability**
   Embeddings can be reused for:

   * RAG retrieval
   * Cluster centroids
   * ANN search
   * Temporal or evolution analysis

2. **Speed**
   BERTopic runs significantly faster when embeddings are precomputed, enabling rapid iteration and parameter tuning.

3. **Diagnostics & Control**
   We can inspect vectors, clusters, and metadata independently of BERTopic.

---

## 4. Qdrant Collections

We work with **four embedding collections** in Qdrant:

* `bbc_news_chunks`
* `nbc_news_chunks`
* `us_congress_chunks`
* `uk_parliament_chunks`

Each point contains:

* embedding vector
* text chunk
* metadata (source, date, corpus, offsets)

---

## 5. BERTopic Configuration (Exploratory Phase)

### Why override BERTopic defaults

Default BERTopic parameters are generic and not well-suited for:

* long-form political text
* precomputed embeddings
* mixed granularities (news vs debates)

We explicitly control UMAP and HDBSCAN to adapt clustering behavior to our data.

---

### Dimensionality Reduction (UMAP)

```text
n_neighbors = 15
n_components = 5
min_dist = 0.0
metric = cosine
```

**Rationale:**

* Preserve global topic structure
* Allow tight clusters for recurring themes
* Cosine distance matches embedding geometry

---

### Clustering (HDBSCAN)

```text
min_cluster_size = 30
min_samples = None
metric = euclidean
cluster_selection_method = eom
prediction_data = True
```

**Rationale:**

* Enforce topic robustness (no tiny or unstable topics)
* Allow natural outliers instead of forcing weak assignments
* Enable post-hoc probability inspection

---

## 6. Fit → Diagnose → Reduce (Not Reduce-First)

We **do not** reduce to 20 topics immediately.

### Process

1. Run `fit_transform()` with **no forced topic count**
2. Save full diagnostics:

   * number of discovered clusters
   * outlier rate
   * cluster size distribution
   * probability statistics
3. Inspect results
4. Only then call `reduce_topics(nr_topics=20)`

This allows us to identify problems **before** collapsing structure.

---

## 7. What “Successful” BERTopic Means for Us

Before reduction, we expect:

* ~30–40 natural clusters
* Balanced topic sizes (no mega-topic)
* Meaningful, interpretable labels
* Outlier rate that is noticeable but not dominant

This range provides enough structure so that reducing to 20 topics is a **controlled compression**, not an artificial invention.

---

## 8. Empirical Findings & Iterations

### 8.1 Document Granularity

* **News articles and US Congress speeches**: single-topic → one document
* **UK parliamentary speeches**: multi-topic → semantic chunking + merging

---

### 8.2 Speaker Name Bias (US Congress)

Early runs revealed two very large clusters dominated by **speaker names**.

**Cause:**

* Speeches often begin with patterns like `Mr. X`, `Ms. Y`, `Mr. Speaker`
* Names are unique and dominate token importance
* HDBSCAN clusters by *who* is speaking instead of *what* is discussed

**Fix:**

* Remove speaker names from the text body
* Add speaker-related terms to **custom stopwords**

This preserves semantic meaning while removing clustering noise.

---

### 8.3 UK Parliament Example Diagnostics

Example run (5,000 documents):

* **Before reduction**:

  * 30 topics (excluding outliers)
  * Min topic size: 32
  * No tiny or junk topics
  * Outlier rate: ~26%

* **After reduction to 20**:

  * Balanced topic sizes
  * Improved topic probability confidence
  * Stable semantic labels

**Conclusion:**
Topics were semantically strong and balanced, but the outlier rate suggested further HDBSCAN tuning was needed.

---

## 9. Key Takeaways

* Cleaning is **not optional** for topic modeling on political text
* Headers and boilerplate can completely dominate clustering
* Precomputed embeddings enable speed, reuse, and deep diagnostics
* Topic reduction should be the **last step**, not the first
* “Good” topic models are diagnosed empirically, not assumed

This step establishes a clean, interpretable semantic space that downstream RAG and temporal analysis can rely on.

---

*Last updated: February 2026*
