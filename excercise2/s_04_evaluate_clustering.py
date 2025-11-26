"""
s_04_evaluate_clustering.py

Evaluate clustering results (KMeans, DBSCAN, HDBSCAN, GMM)
against the true UK/US labels using:
- Accuracy
- Precision
- Recall
- F1

Also:
- Save a summary table to Excel (per method), including per-class metrics.

Assumes:
- BM25 + labels were built by BM25 script and can be loaded via
  `load_bm25_and_metadata(VECTORS_LEMMAS_DIR)`.
- Each clustering method saved results under:
    clusters/
        kmeans/
            cluster_labels.npy
            clustering_meta.json
        dbscan/
            cluster_labels.npy
            clustering_meta.json
        hdbscan/
            cluster_labels.npy
            clustering_meta.json
        gmm/
            cluster_labels.npy
            clustering_meta.json
"""

import os
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from s_03_clustering_utils import (
    VECTORS_LEMMAS_DIR,
    CLUSTERS_ROOT_DIR,
    load_bm25_and_metadata,
)


# ============================
# Helpers for loading data
# ============================

def load_bm25_and_true_labels():
    """
    Load BM25 matrix and true labels (UK/US) via the BM25 metadata.
    Returns:
        bm25_matrix: sparse matrix (n_docs x n_terms)
        true_labels: np.ndarray of shape (n_docs,)
    """
    bm25_matrix, feature_names, filenames, true_labels = load_bm25_and_metadata(
        VECTORS_LEMMAS_DIR
    )
    true_labels = np.array(true_labels)
    return bm25_matrix, true_labels


def load_cluster_labels(method_name):
    """
    Load cluster labels and params for a given method.

    Expects:
        clusters/<method_name>/cluster_labels.npy
        clusters/<method_name>/clustering_meta.json

    Returns:
        labels: np.ndarray of shape (n_docs,)
        params: dict (may contain hyperparameters, etc.)
    """
    method_dir = os.path.join(CLUSTERS_ROOT_DIR, method_name)
    labels_path = os.path.join(method_dir, "cluster_labels.npy")
    params_path = os.path.join(method_dir, "clustering_meta.json")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    labels = np.load(labels_path)

    params = {}
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

    return labels, params


# ============================
# Majority-vote mapping
# ============================

def majority_vote_mapping(true_labels, cluster_labels):
    """
    Map each cluster_id -> {UK, US} based on majority vote.
    NO special handling for noise: all cluster IDs (including -1) are used.

    Args:
        true_labels: np.ndarray of shape (n_docs,), values like "UK" / "US"
        cluster_labels: np.ndarray of shape (n_docs,), e.g. 0,1,2,-1,...

    Returns:
        y_true_eval: np.ndarray of true labels (all docs)
        y_pred_eval: np.ndarray of predicted labels (mapped from cluster ids)
        mapping: dict {cluster_id: assigned_class_label}
    """
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    # All points are used (enable masking noise if desired later)
    mask = np.ones_like(cluster_labels, dtype=bool)

    # Classes that exist in the ground truth
    class_labels = np.unique(true_labels)

    # Build mapping cluster_id -> majority class
    mapping = {}
    cluster_ids = np.unique(cluster_labels[mask])

    for cid in cluster_ids:
        # indices of points that belong to this cluster
        idx = np.where(mask & (cluster_labels == cid))[0]
        if len(idx) == 0:
            continue

        # true labels of these points
        labels_in_cluster = true_labels[idx]

        # majority vote among class_labels
        counts = {cls: np.sum(labels_in_cluster == cls) for cls in class_labels}
        majority_class = max(counts, key=counts.get)
        mapping[cid] = majority_class

    # Now build predictions with this mapping
    y_true_eval = true_labels[mask]
    y_pred_eval = []

    # Most frequent class overall (fallback)
    overall_counts = {cls: np.sum(true_labels == cls) for cls in class_labels}
    global_majority_class = max(overall_counts, key=overall_counts.get)

    for lbl, use in zip(cluster_labels, mask):
        if not use:
            continue
        if lbl in mapping:
            y_pred_eval.append(mapping[lbl])
        else:
            # Fallback: assign overall majority class
            y_pred_eval.append(global_majority_class)

    y_pred_eval = np.array(y_pred_eval)

    return y_true_eval, y_pred_eval, mapping


# ============================
# Evaluation for one method
# ============================

def evaluate_method(method_name, true_labels):
    """
    Evaluate one clustering method (e.g. 'kmeans', 'dbscan', 'hdbscan', 'gmm').

    Prints:
        - number of clusters
        - confusion matrix
        - per-class precision, recall, F1
        - accuracy

    Returns:
        summary: dict with key metrics for this method, including per-class metrics.
    """
    print("\n" + "-" * 70)
    print(f"Evaluating method: {method_name.upper()}")
    print("-" * 70)

    labels, params = load_cluster_labels(method_name)
    n_docs = len(true_labels)

    # Map clusters to UK/US via majority voting, using ALL points
    y_true, y_pred, mapping = majority_vote_mapping(true_labels, labels)

    n_used = len(y_true)
    n_noise = 0  # no noise by definition here
    n_clusters = len(set(labels))

    print(f"Documents total       : {n_docs}")
    print(f"Documents in evaluation: {n_used}")
    print(f"Number of clusters    : {n_clusters}")
    print(f"Cluster → class mapping (majority vote): {mapping}")

    # Unique class labels (e.g. ["UK", "US"])
    class_labels = np.unique(true_labels)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("Classes:", class_labels)
    print(cm)

    # Precision, recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_labels,
        average=None,
        zero_division=0,
    )

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    print("\nPer-class metrics:")
    for cls, p, r, f, s in zip(class_labels, precision, recall, f1, support):
        print(
            f"  Class {cls}: "
            f"precision={p:.3f}, recall={r:.3f}, F1={f:.3f}, support={s}"
        )

    # Macro-averaged F1 (average over classes)
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )

    print(f"\nOverall accuracy: {acc:.3f}")
    print(f"Macro-averaged F1: {f1_macro:.3f}")

    # ---- Build per-class metrics dict ----
    per_class_metrics = {}
    for cls, p, r, f, s in zip(class_labels, precision, recall, f1, support):
        per_class_metrics[cls] = {
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f),
            "support": int(s),
        }

    # ---- Build final summary dict with more descriptive keys ----
    summary = {
        "method": method_name,
        "number_of_documents": int(n_docs),
        "number_of_used_docs": int(n_used),
        "number_of_noise_docs": int(n_noise),  # always 0 here
        "number_of_clusters": int(n_clusters),
        "overall_accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "per_class": per_class_metrics,
    }

    return summary


# ============================
# Save summaries to Excel
# ============================

def save_summaries_to_excel(summaries, output_path):
    """
    Save the list of summary dicts to an Excel file.

    The per-class metrics are flattened into columns like:
      UK_precision, UK_recall, UK_f1_score, UK_support, US_precision, ...
    """
    if not summaries:
        print("[!] No summaries to save.")
        return

    # Collect all class labels that appear in any summary
    all_classes = sorted(
        {cls for s in summaries for cls in s.get("per_class", {}).keys()}
    )

    rows = []
    for s in summaries:
        # Base fields (global metrics)
        flat = {
            "method": s["method"],
            "number_of_documents": s["number_of_documents"],
            "number_of_used_docs": s["number_of_used_docs"],
            "number_of_noise_docs": s["number_of_noise_docs"],
            "number_of_clusters": s["number_of_clusters"],
            "overall_accuracy": s["overall_accuracy"],
            "f1_macro": s["f1_macro"],
        }

        # Per-class fields
        per_class = s.get("per_class", {})
        for cls in all_classes:
            metrics = per_class.get(cls, None)
            prefix = str(cls)

            if metrics is not None:
                flat[f"{prefix}_precision"] = metrics["precision"]
                flat[f"{prefix}_recall"] = metrics["recall"]
                flat[f"{prefix}_f1_score"] = metrics["f1_score"]
                flat[f"{prefix}_support"] = metrics["support"]
            else:
                # If this method doesn't have that class (unlikely here), fill NaN
                flat[f"{prefix}_precision"] = np.nan
                flat[f"{prefix}_recall"] = np.nan
                flat[f"{prefix}_f1_score"] = np.nan
                flat[f"{prefix}_support"] = np.nan

        rows.append(flat)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"\n✅ Saved evaluation summary to Excel: {output_path}")


# ============================
# Main
# ============================

def main():
    print("=" * 70)
    print("EVALUATING CLUSTERING METHODS AGAINST UK/US LABELS (BM25 Lemmas)")
    print("=" * 70)

    # 1. Load BM25 matrix + ground-truth labels (UK/US)
    print("\n[1] Loading BM25 matrix and true labels...")
    bm25_matrix, true_labels = load_bm25_and_true_labels()
    print(f"  → Loaded {len(true_labels)} labels.")
    print(f"  → Classes: {np.unique(true_labels)}")

    # 2. Evaluate each method (if results exist)
    methods = ["kmeans", "dbscan", "hdbscan", "gmm"]
    summaries = []

    for method in methods:
        method_dir = os.path.join(CLUSTERS_ROOT_DIR, method)
        if not os.path.isdir(method_dir):
            print(f"\n[!] Skipping '{method}' (no directory {method_dir})")
            continue

        try:
            summary = evaluate_method(method, true_labels)
            summaries.append(summary)
        except FileNotFoundError as e:
            print(f"[!] Could not evaluate '{method}': {e}")

    # 3. Print compact summary table
    if summaries:
        print("\n" + "=" * 70)
        print("SUMMARY (all documents used)")
        print("=" * 70)
        print(f"{'Method':>10} | {'Clusters':>10} | {'Noise':>10} | {'Acc':>8} | {'F1-macro':>10}")
        print("-" * 70)
        for s in summaries:
            print(
                f"{s['method']:>10} | "
                f"{s['number_of_clusters']:>10} | "
                f"{s['number_of_noise_docs']:>10} | "
                f"{s['overall_accuracy']:.3f} | "
                f"{s['f1_macro']:.3f}"
            )
        print("=" * 70)

        # 4. Save to Excel
        excel_path = os.path.join(CLUSTERS_ROOT_DIR, "cluster_evaluation_results.xlsx")
        save_summaries_to_excel(summaries, excel_path)
    else:
        print("\n[!] No methods were evaluated. Nothing to save.")


if __name__ == "__main__":
    main()
