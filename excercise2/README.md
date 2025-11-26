# Exercise 2: Document Clustering of Congressional Debates

## Overview

This project performs clustering analysis on two corpora of parliamentary debates:
- **US Congress Debates**: [`data/US_congress_debates`](data/US_congress_debates)
- **British Parliament Debates**: [`data/british_parliament_debates`](data/british_parliament_debates)

The goal is to apply various clustering algorithms to identify patterns and group documents based on their textual content.

---

## Project Structure

```
.
├── data/                          # Raw debate documents
├── clean_docs/                    # Cleaned documents
├── lemmas/                        # Lemmatized documents
├── vectors/                       # BM25 vectorized representations
│   ├── BM25_clean/
│   └── BM25_lemmas/
├── clusters/                      # Clustering results
│   ├── kmeans/
│   ├── dbscan/
│   ├── hdbscan/
│   └── gmm/
├── s_01_clean_and_lemmatize.py
├── s_02_build_sparse_vectors_tf_bm25.py
├── s_03_clustering_utils.py
├── s_03_k-mean_clustering.py
├── s_03_DBSCAN_clustering.py
├── s_03_HDBSCAN_clustering.py
├── s_03_GMM_clustering.py
├── s_04_evaluate_clustering.py
└── s_05_visualize_clusters.py
```

---

## Usage

Run the pipeline stages sequentially:

```bash
# Stage 1: Preprocessing
python s_01_clean_and_lemmatize.py

# Stage 2: Vectorization
python s_02_build_sparse_vectors_tf_bm25.py

# Stage 3: Clustering
python s_03_k-mean_clustering.py
python s_03_DBSCAN_clustering.py
python s_03_HDBSCAN_clustering.py
python s_03_GMM_clustering.py

# Stage 4: Evaluation
python s_04_evaluate_clustering.py

# Stage 5: Visualization
python s_05_visualize_clusters.py
```

## Pipeline Stages

### Stage 1: Text Preprocessing

Documents are cleaned and lemmatized to prepare them for vectorization.

**Outputs:**
- Cleaned documents: [`clean_docs/`](clean_docs)
- Lemmatized documents: [`lemmas/`](lemmas)

### Stage 2: Vectorization

BM25 sparse vectors are constructed for both corpora using a unified vocabulary. Each document is represented as a vector and assigned a ground truth label (US or UK).

**Outputs:**
- Vocabulary: [`vectors/BM25_clean/vocabulary.json`](vectors/BM25_clean/vocabulary.json)
- BM25 matrix: `vectors/BM25_clean/bm25_okapi.npz`
- Filenames: [`vectors/BM25_clean/filenames.json`](vectors/BM25_clean/filenames.json)
- Ground truth labels: [`vectors/BM25_clean/labels.json`](vectors/BM25_clean/labels.json)

### Stage 3: Clustering

Multiple clustering algorithms are applied to the BM25 lemmatized vectors. Each method generates predicted cluster labels and metadata stored in the `clusters/` directory.

---

## Clustering Methods

### Method 1: MiniBatchKMeans

**Algorithm:** `MiniBatchKMeans` from scikit-learn

MiniBatchKMeans is a scalable variant of K-Means that processes data in mini-batches, making it efficient for large corpora. It iteratively updates cluster centroids based on subsets of the data.

**Parameters:**
- `n_clusters = 2` — Target number of clusters (US vs UK)
- `batch_size = 256` — Number of documents per batch
- `max_iter = 100` — Maximum number of iterations

**Outputs:**
- Cluster labels: [`clusters/kmeans/cluster_labels.npy`](clusters/kmeans/cluster_labels.npy)
- Metadata: [`clusters/kmeans/clustering_meta.json`](clusters/kmeans/clustering_meta.json)

---

### Method 2: DBSCAN

**Algorithm:** `DBSCAN` from scikit-learn

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies clusters based on density. Unlike K-Means, it doesn't require specifying the number of clusters and can detect outliers as noise points.

**Key Concept:**
- Points with sufficient neighbors within a radius (`eps`) form dense regions (clusters)
- Points in low-density regions are classified as noise

**Parameter Selection:**

*`min_samples = 5`*
- Minimum number of neighbors required for a point to be considered a core point
- Value chosen based on high-dimensional sparse data characteristics (lower values prevent excessive noise classification)

*`eps` (neighborhood radius)*
- Determined automatically using **k-distance heuristic method**:

**K-Distance Heuristic Process:**
1. **Compute Pairwise Distances**: Calculate cosine distance between all document pairs in the BM25 vector space
2. **Find k-Nearest Neighbors**: For each document, identify the k-th nearest neighbor (where k = `min_samples`)
3. **Extract k-Distances**: Record the distance to the k-th neighbor for every document
4. **Apply Quantile Threshold**: Take the 80th percentile of all k-distances as the `eps` value

**Rationale:**
- Documents in dense regions have smaller k-distances (neighbors are close)
- Documents in sparse regions or near cluster boundaries have larger k-distances
- The 80th percentile approximates the "elbow point" in a k-distance plot, separating dense clusters from noise
- This approach avoids manual `eps` tuning by automatically adapting to the data's density distribution

**Noise Handling:**
- Initial noise points (label = -1) are recorded in metadata
- Post-processing step: each noise point is reassigned to its nearest cluster based on cosine distance to cluster centroids
- This ensures all documents receive a cluster assignment while preserving information about original noise classification

**Outputs:**
- Cluster labels: [`clusters/dbscan/cluster_labels.npy`](clusters/dbscan/cluster_labels.npy)
- Metadata: [`clusters/dbscan/clustering_meta.json`](clusters/dbscan/clustering_meta.json)
  - Contains: algorithm parameters, number of clusters found, noise point statistics, heuristic details

---

### Method 3: HDBSCAN

**Algorithm:** `HDBSCAN` from the hdbscan library

HDBSCAN (Hierarchical DBSCAN) extends DBSCAN by varying the density threshold (eps) across different regions. It builds a hierarchy of clusters at different density levels and selects the most stable clusters.

**Key Advantages:**
- Adapts to varying density across the data space
- Automatically determines optimal density thresholds
- No need to specify `eps` manually

**How It Works:**
Unlike DBSCAN which uses a fixed `eps` radius, HDBSCAN:
1. Tests multiple distance thresholds to build a **cluster hierarchy**
2. Identifies clusters that remain stable across different density levels
3. Selects the most persistent clusters from the hierarchy

This approach handles datasets with **varying cluster densities** more effectively than standard DBSCAN.

**Parameter Selection:**

*`min_cluster_size = 30`*
- Minimum number of documents for a valid cluster
- Set to ~3-5% of dataset size to ensure meaningful groupings
- Prevents overly fragmented clustering

*`min_samples = 8`*
- Controls clustering conservativeness (higher values create denser clusters with more noise points)
- Balances between capturing true clusters and avoiding excessive noise classification
- Set to require reasonable local density without being overly strict

*`cluster_selection_method = 'eom'`* (Excess of Mass)
- Selects stable, large clusters from the hierarchy
- Alternative `'leaf'` would produce many small clusters
- EOM is preferred for identifying the two main groups (US vs UK)

**Noise Handling:**
- Points that don't belong to any stable cluster are initially labeled as noise (label = -1)
- Post-processing: noise points are reassigned to their nearest cluster based on cosine distance
- Original noise count is preserved in metadata for analysis

**Outputs:**
- Cluster labels: [`clusters/hdbscan/cluster_labels.npy`](clusters/hdbscan/cluster_labels.npy)
- Metadata: [`clusters/hdbscan/clustering_meta.json`](clusters/hdbscan/clustering_meta.json)
  - Contains: parameters, cluster counts, noise statistics, selection rationale

---

### Method 4: GMM (Gaussian Mixture Model)

**Algorithm:** `GaussianMixture` from scikit-learn

GMM is a probabilistic clustering method that models the data as a mixture of Gaussian distributions. Unlike hard clustering (K-Means), GMM provides soft cluster assignments with probabilities.

**Key Concept:**
- Assumes data is generated from a mixture of Gaussian distributions
- Each cluster is represented by a Gaussian with mean and covariance
- Documents can belong to multiple clusters with different probabilities

**Parameters:**
- `n_components = 2` — Number of Gaussian components (US vs UK)
- `covariance_type = 'tied'` — Shared covariance across clusters (reduces parameters for high-dimensional sparse data)
- `max_iter = 100` — Maximum EM algorithm iterations

**Outputs:**
- Cluster labels: [`clusters/gmm/cluster_labels.npy`](clusters/gmm/cluster_labels.npy)
- Metadata: [`clusters/gmm/clustering_meta.json`](clusters/gmm/clustering_meta.json)

---

## Stage 4: Evaluation

### Evaluation Methodology

Clustering results from Stage 3 are evaluated against ground truth labels (US/UK) from Stage 2 using standard classification metrics.

**Libraries Used:**
- `sklearn.metrics` for computing accuracy, precision, recall, F1-score, and confusion matrices
- `pandas` for organizing results into tabular format
- `openpyxl` (via pandas) for Excel output generation

**Cluster-to-Class Mapping:**
- Each cluster is assigned to a class (US or UK) using **majority vote**
- The class with the most documents in a cluster becomes that cluster's label
- All predicted cluster labels are then mapped to their corresponding class labels

**Evaluation Metrics:**

For each clustering method and each class (US and UK), the following metrics are calculated:

**1. Accuracy**
- Formula: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- Measures: Overall proportion of correctly classified documents
- Range: 0 to 1 (higher is better)

**2. Precision**
- Formula: `Precision = TP / (TP + FP)`
- Measures: Of all documents predicted as a certain class, how many actually belong to that class
- Purpose: Prevents false positives — ensures predicted labels are reliable
- Example: If precision for US = 0.85, then 85% of documents predicted as US are truly US documents

**3. Recall**
- Formula: `Recall = TP / (TP + FN)`
- Measures: Of all documents that truly belong to a class, how many were correctly identified
- Purpose: Prevents false negatives — ensures we don't miss documents from the target class
- Example: If recall for UK = 0.78, then 78% of actual UK documents were correctly identified as UK

**4. F1-Score (per class)**
- Formula: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
- Measures: Harmonic mean of precision and recall
- Purpose: Balances precision and recall into a single metric
- Range: 0 to 1 (higher is better)

**5. F1-Macro (overall)**
- Formula: `F1-Macro = (F1_US + F1_UK) / 2`
- Measures: Average F1-score across both classes
- Purpose: **Prevents class imbalance bias** — treats both classes equally regardless of size
- Example: If we have 90% US documents and 10% UK documents, and a model assigns everything to US, accuracy would be 90% but F1-Macro would be ~0.47, revealing poor performance on the minority class

**Why F1-Macro Matters:**
- Simple accuracy can be misleading with imbalanced datasets
- A naive classifier that always predicts the majority class (e.g., "US") would achieve high accuracy but fail on the minority class (UK)
- F1-Macro ensures both US and UK clusters are evaluated fairly, regardless of dataset composition

**Outputs:**
- Console output: Confusion matrices 
  (rows = true class, columns = predicted class)
, per-class metrics, and summary statistics for each method
- Excel file: [`clusters/cluster_evaluation_results.xlsx`](clusters/cluster_evaluation_results.xlsx)
  - Contains: Method name, number of clusters, accuracy, F1-Macro, and detailed per-class metrics (precision, recall, F1, support) for both US and UK

---

## Stage 5: Visualization

### UMAP Dimensionality Reduction

High-dimensional BM25 vectors (thousands of dimensions) are reduced to 2D space using **UMAP** (Uniform Manifold Approximation and Projection) for visualization. UMAP preserves both local and global structure, making it ideal for visualizing clusters.

**What is UMAP?**
- A dimensionality reduction technique similar to t-SNE but faster and better at preserving global structure
- Maps high-dimensional sparse vectors to 2D while maintaining distances between similar documents
- Uses cosine distance metric (same as clustering algorithms) for consistency

**Visualizations Created:**

1. **Ground Truth Labels** (`umap_true_labels.png`)
   - Shows the true UK vs US separation in 2D space
   - Baseline reference for evaluating clustering quality

2. **Clustering Results** (one plot per method)
   - `umap_kmeans.png` — MiniBatchKMeans cluster assignments
   - `umap_dbscan.png` — DBSCAN clusters (noise points shown as gray crosses)
   - `umap_hdbscan.png` — HDBSCAN clusters (noise points shown as gray crosses)
   - `umap_gmm.png` — GMM cluster assignments

**Interpretation:**
- **Tight, separated clusters**: Indicate clear distinction between US and UK documents
- **Overlapping regions**: Suggest documents with mixed characteristics or ambiguous language
- **Noise points** (DBSCAN/HDBSCAN): Documents that don't fit well into any cluster, shown as gray crosses

**Outputs:**
- All visualizations saved in: [`clusters/visualizations/`](clusters/visualizations/)
- High-resolution PNG files suitable for reports and presentations

---