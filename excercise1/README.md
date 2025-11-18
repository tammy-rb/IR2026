# British Parliament Debates — Information Retrieval Pipeline

### name: Tammy Rabinovitz
### ID: 214667875

## Stage 1 — Downloading XML Corpus

### Goal:
Download parliamentary debate transcripts from **TheyWorkForYou.com**.

### Method:
- Scrape index page and download XML files starting from `debates2023-06-28d.xml`
- Skip existing files to resume interrupted downloads

### Output:
- Raw XML files in `docs/` — structured debate transcripts with metadata

---

## Stage 2 — XML Cleaning and Tokenization

### Goal:
Extract clean text from XML documents and tokenize into words for NLP processing.

### Method:
- **Text extraction:** Used `lxml` XPath `//text()` to extract all textual content, stripping XML tags and structure
- **Tokenization:** Used `spaCy en_core_web_sm` (disabled parser/NER/textcat for speed)
- **No filtering:** Kept all words, numbers, and punctuation for maximum flexibility

### Reasoning:
- **spaCy chosen** for linguistic accuracy (handles contractions, punctuation, Unicode correctly).
- **Disabled components** for 5-10x faster processing with large corpus

### Output:
- `clean_docs/*.clean.txt` — space-separated tokens (words + punctuation)

---

## Stage 3 — Lemmatization

### Goal:
Reduce words to base forms (lemmas) to normalize morphological variations and improve retrieval effectiveness.

### Method:
- **Tool:** Used `spaCy en_core_web_sm` lemmatizer (disabled parser/NER/textcat)
- **Process:** Extract `.lemma_` attribute, lowercase all tokens
- **Special handling:** Replace `-PRON-` placeholder with original pronoun (preserves "I" vs "they" distinctions)

### Reasoning:
- **Reduces vocabulary** 30-40% by conflating variants ("running"/"ran"/"runs" → "run")

### Output:
- `lemmas/*.lemma.txt` — space-separated lemmas (lowercase)

---

## Stage 4 — Building Sparse Vectors: TF & BM25

### Goal:
Create sparse vector representations of documents using both **Term Frequency (TF)** and **Okapi BM25** weighting schemes for two datasets:
1. Lemmatized documents (from `lemmas/`)
2. Clean tokenized documents (from `clean_docs/`)

### Method:

#### 4.1 TF Matrix Construction
- Used `sklearn.feature_extraction.text.CountVectorizer` to build Term Frequency matrices
- **Parameters chosen:**
  - `min_df=3`: Terms must appear in at least 3 documents (removes rare/noisy terms)
  - `max_df=0.8`: Terms appearing in >80% of documents are excluded (removes overly common terms)
  - `stop_words="english"`: Filters common English stopwords
  - `analyzer="word"`: Word-level tokenization

#### 4.2 BM25 Weighting
- Implemented **Okapi BM25** formula to weight the TF matrix:
  ```
  BM25(t,d) = IDF(t) × [tf(t,d) × (k1 + 1)] / [tf(t,d) + k1 × (1 - b + b × |d|/avgdl)]
  ```
- **Parameters chosen:**
  - `k1=1.6`: Controls term frequency saturation (higher values allow more weight for repeated terms)
  - `b=0.75`: Controls document length normalization (standard value balancing length effects)

#### 4.3 Dual Dataset Processing
Built separate vector spaces for:
- **Lemmatized docs** → `vectors/BM25_lemmas/`
- **Clean docs** → `vectors/BM25_words/`

### Output:

For each dataset (lemmas and words):
- `sparse_TF_matrix.npz`: Sparse CSR matrix of term frequencies
- `bm25_okapi.npz`: Sparse CSR matrix with BM25 weights
- `vocabulary.json`: Term → column index mapping
- `filenames.json`: Document filenames for row indices

---

## Stage 5 — Building Dense Vectors: Word2Vec

### Goal:
Create **dense vector representations** of documents using word embeddings trained with **Word2Vec**, applied to both lemmatized and clean datasets.

### Method:

#### 5.1 Text Preprocessing for Embeddings
Used `spaCy` tokenization with strict filtering:
- **Removed:**
  - Punctuation and whitespace
  - Numbers and numeric-like tokens (`like_num`)
  - Any token containing digits
  - English stopwords (from `sklearn`)
- **Normalized:** All tokens to lowercase

#### 5.2 Word2Vec Training
- Used **Gensim's Word2Vec** implementation
- **Architecture:** Skip-gram (`sg=1`)
- **Parameters:**
  - `vector_size=100`: 100-dimensional word embeddings
  - `window=5`: Context window of 5 words on each side
  - `min_count=5`: Words appearing <5 times are ignored
  - `epochs=10`: Training iterations over the corpus
  - `workers=4`: Parallel processing threads

#### 5.3 Document Vector Construction
- Each document represented as **average of its word vectors**
- Formula: `doc_vec = mean([word2vec(w) for w in doc_tokens if w in vocab])`
- Out-of-vocabulary tokens are skipped
- Empty documents → zero vector

#### 5.4 Dual Dataset Processing
Built embeddings for:
- **Lemmatized docs** → `vectors/word2vec_lemmas/`
- **Clean docs** → `vectors/word2vec_words/`

### Reasoning:

**Why Word2Vec?**
- **Semantic representation:** Captures word meanings and relationships 
- **Dense vectors:** More compact than sparse methods (100 dims vs thousands)
- **Context-aware:** Learns from word co-occurrence patterns

**Why Skip-gram over CBOW?**
- Better performance on smaller datasets
- More effective at capturing rare word semantics
- Standard choice for IR applications

**Why averaging for document vectors?**
- **Simple and effective:** Averaging is a proven baseline for document representation
- **Fast:** No additional training required (unlike Doc2Vec)

### Output:

For each dataset (lemmas and words):
- `word2vec.kv`: Trained word embeddings (KeyedVectors format)
- `doc_vectors.npy`: NumPy array of document vectors (shape: n_docs × 100)
- `filenames.json`: Document filename mapping

---

## Stage 6 — Transformer-Based Document Embeddings (SimCSE & SBERT)

### Goal:
Create dense document representations using pre-trained **Sentence Transformer models** with chunking. Both models are BERT-based and generate dynamic contextual vectors for each chunk. They use BERT's transformer layers fine-tuned on specific tasks, then apply mean pooling (instead of the CLS token) to aggregate word-level contextual embeddings into chunk-level vectors.

### Method:

#### 6.1 Document Chunking (`s_06_build_chuncks_from_docs.py`)
- **Strategy:** Group consecutive sentences into chunks of max 256 words using spaCy segmentation
- **Preserves context:** Sentence grouping maintains semantic coherence vs arbitrary splitting
- **Output:** Individual JSON files per document in `docs_chuncks/` (e.g., `debates2023-06-28d.xml.chunks.json`)

#### 6.2 Encoding Pipeline (`s_06_encode_chuncks.py`)
- **Generic framework:** Works with any SentenceTransformer model
- **Memory-efficient:** Process documents individually, encode chunks in batches (`batch_size=16`)
- **Aggregation:** Mean pooling across chunk embeddings to create final document vector

#### 6.3 SimCSE Embeddings (`s_06-1_build_SimCSE_vectors.py`)
- **Model:** `princeton-nlp/unsup-simcse-bert-base-uncased`
- **Method:** Contrastive learning (same sentence + dropout = positive pairs)
- **Output:** 768-dim embeddings in `vectors/simcse_raw/`

#### 6.4 SBERT Embeddings (`s_06-2_build_SBERT_vectors.py`)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (most popular SBERT model)
- **Method:** Multi-task training on NLI + semantic similarity + QA
- **Output:** 384-dim embeddings in `vectors/SBERT_origin/`

### Output:
For each model:
- `{model}_doc_embeddings.npy`: Document vectors
- `filenames.json`: Document mapping
- `model_name.txt`: Model identifier

---

## Stage 7 — BM25 Clustering & Feature Importance Analysis

### Goal:
Discover document clusters and identify the most discriminative terms using unsupervised clustering and feature selection methods on BM25 sparse vectors.

### Method:

#### 7.1 Document Clustering (`s_07-1_cluster_bm25.py`)

**Clustering Approach:**
- **Algorithm:** MiniBatchKMeans with `k=5` clusters
- **Input:** BM25 sparse matrices (from `vectors/BM25_lemmas/` and `vectors/BM25_words/`)
- **Purpose:** Create pseudo-labels for feature importance analysis

**Why MiniBatchKMeans?**
- Memory-efficient for large sparse matrices
- Faster than standard KMeans with minimal accuracy loss
- Well-suited for high-dimensional text data

**Parameters:**
- `n_clusters=5`: Groups documents into 5 thematic clusters
- `batch_size=256`: Process documents in batches
- `max_iter=100`: Clustering iterations

#### 7.2 Feature Importance Analysis (`s_07-2_feature_importance_bm25.py`)

**Metrics Computed:**

1. **Information Gain (Mutual Information):**
   - Measures how much knowing a term reduces uncertainty about cluster membership
   - `MI(term, cluster) = H(cluster) - H(cluster | term)`
   - Higher values → term is more informative for distinguishing clusters

2. **Chi-Squared (χ²) Score:**
   - Statistical test for independence between term presence and cluster assignment
   - Measures deviation from expected distribution if term and cluster were independent
   - Higher values → stronger association between term and specific clusters

**Process:**
1. Load BM25 matrices and precomputed cluster labels
2. Convert BM25 to binary (presence/absence) for proper statistical analysis
3. Compute Information Gain using `sklearn.feature_selection.mutual_info_classif`
4. Compute Chi-Squared scores using `sklearn.feature_selection.chi2`
5. Rank terms by each metric

**Why Binary Conversion?**
- Information Gain and Chi-Squared assume categorical/binary features
- Binary representation focuses on term presence (more interpretable for feature selection)

### Output:

For each dataset (lemmas and words):
- `cluster_labels_k5.npy`: Cluster assignments for each document
- `feature_importance_lemmas.xlsx`: Excel file with 2 sheets:
  - **InformationGain**: Terms ranked by mutual information
  - **ChiSquared**: Terms ranked by chi-squared scores
- `feature_importance_words.xlsx`: Same structure for clean words dataset

### Top 20 Most Informative Terms (by Information Gain):

#### Lemmatized Documents:
1. regard (0.514)
2. appear (0.513)
3. miss (0.504)
4. outline (0.501)
5. damage (0.500)
6. sort (0.498)
7. sorry (0.495)
8. extremely (0.495)
9. afraid (0.494)
10. lack (0.493)
11. date (0.493)
12. direct (0.491)
13. especially (0.491)
14. conduct (0.491)
15. push (0.490)
16. likely (0.490)
17. assurance (0.488)
18. strike (0.486)
19. acknowledge (0.486)
20. operate (0.486)

#### Clean Words Documents:
1. sides (0.552)
2. regard (0.544)
3. adjourn (0.533)
4. afraid (0.529)
5. takes (0.528)
6. direct (0.526)
7. sorry (0.526)
8. discussions (0.526)
9. extremely (0.524)
10. ways (0.524)
11. especially (0.523)
12. concern (0.523)
13. remarks (0.523)
14. adjourned (0.522)
15. mind (0.522)
16. requires (0.522)
17. effect (0.521)
18. ability (0.521)
19. appreciate (0.521)
20. shared (0.521)

**Interpretation:**
- These terms have the highest mutual information with cluster assignments
- They are the most discriminative features for distinguishing between document clusters
- Terms like "regard," "sorry," "afraid," "direct," and "especially" appear in both lists, indicating strong clustering power across both preprocessing approaches
- Clean words show slightly higher Information Gain values, suggesting more distinct term distributions

---


