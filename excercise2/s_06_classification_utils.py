"""
classification_utils.py

Shared utilities for supervised classification scripts:
- Loading BM25 + labels
- Running cross-validation with consistent metrics and printing
- Saving CV results to Excel
- Extracting top-N features from linear models (coef_)
- Saving top features to Excel
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate

from s_03_clustering_utils import (
    VECTORS_LEMMAS_DIR,
    load_bm25_and_metadata,
)

BASE_DIR = os.path.dirname(__file__)
CLASSIFICATION_ROOT_DIR = os.path.join(BASE_DIR, "classification")

# ============================
# Load BM25 + labels
# ============================

def load_bm25_and_labels():
    """
    Wrapper around load_bm25_and_metadata(VECTORS_LEMMAS_DIR).

    Returns:
        X: sparse BM25 matrix (n_docs x n_terms)
        feature_names: list[str] of length n_terms
        filenames: list[str] of length n_docs
        y: np.ndarray of shape (n_docs,), labels like "UK"/"US"
    """
    bm25_matrix, feature_names, filenames, true_labels = load_bm25_and_metadata(
        VECTORS_LEMMAS_DIR
    )
    y = np.array(true_labels)
    return bm25_matrix, feature_names, filenames, y


# ============================
# Cross-validation helper
# ============================

def get_default_scoring():
    """
    Return the default scoring metrics dict used for all classifiers.
    """
    return {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }


def run_cross_validation(
    estimator,
    X,
    y,
    n_splits=10,
    random_state=42,
    scoring=None,
    model_name="MODEL",
):
    """
    Run Stratified K-Fold cross-validation for a given estimator
    and print a summary of the metrics.

    Args:
        estimator: any sklearn-like classifier (with fit/predict)
        X: feature matrix
        y: labels
        n_splits: number of folds
        random_state: random seed
        scoring: dict of scorers (if None, use get_default_scoring())
        model_name: name for pretty printing

    Returns:
        cv_results: dict with arrays of scores for each metric
    """
    if scoring is None:
        scoring = get_default_scoring()

    print("=" * 70)
    print(f"{model_name.upper()} - {n_splits}-FOLD CROSS VALIDATION")
    print("=" * 70)

    # Set up Stratified K-Fold CV splitter
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    cv_results = cross_validate(
        estimator,          # the model to evaluate
        X,                  # feature matrix, BM25 sparse matrix
        y,                  # labels array
        cv=cv,              # the CV splitter
        scoring=scoring,     # dict of scoring metrics
        return_train_score=False, 
        n_jobs=-1,
    )

    # Per-fold
    print("\nPer-fold scores:")
    for i in range(n_splits):
        acc = cv_results["test_accuracy"][i]
        prec = cv_results["test_precision_macro"][i]
        rec = cv_results["test_recall_macro"][i]
        f1 = cv_results["test_f1_macro"][i]
        print(
            f"  Fold {i + 1:02d}: "
            f"acc={acc:.3f}, "
            f"prec_macro={prec:.3f}, "
            f"recall_macro={rec:.3f}, "
            f"f1_macro={f1:.3f}"
        )

    # Mean ± std of all folds for each metric
    print("\nOverall summary (mean ± std):")
    for metric_key in scoring.keys():
        scores = cv_results[f"test_{metric_key}"]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"  {metric_key:15s}: {mean_score:.3f} ± {std_score:.3f}")

    return cv_results


def save_cv_results_to_excel(cv_results, output_dir, filename="cv_results.xlsx"):
    """
    Save per-fold CV metrics to Excel.

    Args:
        cv_results: dict from cross_validate
        output_dir: directory to save the file
        filename: Excel filename

    Returns:
        output_path: full path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    n_folds = len(cv_results["test_accuracy"])
    df = pd.DataFrame(
        {
            "fold": np.arange(1, n_folds + 1),
            "accuracy": cv_results["test_accuracy"],
            "precision_macro": cv_results["test_precision_macro"],
            "recall_macro": cv_results["test_recall_macro"],
            "f1_macro": cv_results["test_f1_macro"],
        }
    )

    output_path = os.path.join(output_dir, filename)
    df.to_excel(output_path, index=False)
    print(f"\n✅ Saved CV results to Excel: {output_path}")
    return output_path


def save_cv_summary_to_excel(cv_results, output_dir, filename="cv_summary.xlsx"):
    """
    Save the overall (mean ± std) CV metrics to Excel.

    Args:
        cv_results: dict returned from cross_validate
        output_dir: directory to save
        filename: output Excel file name

    Returns:
        output_path: full path to saved Excel file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute summary metrics
    summary_data = {}
    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        scores = cv_results[f"test_{metric}"]
        summary_data[f"{metric}_mean"] = np.mean(scores)
        summary_data[f"{metric}_std"] = np.std(scores)

    # Convert to DataFrame (single-row table)
    df = pd.DataFrame([summary_data])

    # Save
    output_path = os.path.join(output_dir, filename)
    df.to_excel(output_path, index=False)

    print(f"\n✅ Saved CV summary (mean ± std) to Excel: {output_path}")
    return output_path

import numpy as np
import pandas as pd


def _extract_binary_top_features(
    coef,
    classes,
    feature_names,
    top_n=20,
):
    """
    Binary LogisticRegression-style:
      coef.shape = (1, n_features), len(classes) = 2

    Args:
        coef: np.ndarray of shape (1, n_features)
        classes: np.ndarray of shape (2,), class labels
        feature_names: list of length n_features
        top_n: number of top features to extract per class
    
    Returns: dict[class_label] -> DataFrame(feature, weight, abs_weight)
    """
    weights = coef[0]
    feature_names = np.array(feature_names)
    classes = np.array(classes)

    grouped = {}

    # get top-N positive weights for positive class
    pos_class = classes[1]
    pos_indices = np.argsort(-weights)[:top_n]
    grouped[pos_class] = pd.DataFrame(
        [
            {
                "feature": feature_names[idx],
                "weight": float(weights[idx]),
                "abs_weight": abs(float(weights[idx])),
            }
            for idx in pos_indices
        ]
    )

    # get top-N negative weights for negative class
    neg_class = classes[0]
    neg_indices = np.argsort(weights)[:top_n]
    grouped[neg_class] = pd.DataFrame(
        [
            {
                "feature": feature_names[idx],
                "weight": float(weights[idx]),
                "abs_weight": abs(float(weights[idx])),
            }
            for idx in neg_indices
        ]
    )

    return grouped


def _extract_multiclass_top_features(
    coef,
    classes,
    feature_names,
    top_n=20,
):
    """
    Multiclass:
      coef.shape = (n_classes, n_features), len(classes) = n_classes

    Args:
        coef: np.ndarray of shape (n_classes, n_features)
        classes: np.ndarray of shape (n_classes,), class labels
        feature_names: list of length n_features
        top_n: number of top features to extract per class

    Returns: dict[class_label] -> DataFrame(feature, weight, abs_weight)
    """
    feature_names = np.array(feature_names)
    classes = np.array(classes)

    n_coef_rows, _ = coef.shape
    if n_coef_rows != len(classes):
        raise ValueError(
            f"Mismatch between coef_.shape[0]={n_coef_rows} and len(classes)={len(classes)}"
        )

    grouped = {}

    for class_index, class_label in enumerate(classes):
        coef_for_class = coef[class_index]
        sorted_indices = np.argsort(-np.abs(coef_for_class))
        top_indices = sorted_indices[:top_n]

        grouped[class_label] = pd.DataFrame(
            [
                {
                    "feature": feature_names[idx],
                    "weight": float(coef_for_class[idx]),
                    "abs_weight": abs(float(coef_for_class[idx])),
                }
                for idx in top_indices
            ]
        )

    return grouped


def extract_top_features(
    estimator,
    classes,
    feature_names,
    top_n=20,
):
    """
    Public wrapper: extract top-N features per class for linear models with .coef_.

    Automatically detects:
      - Binary LogisticRegression-style: coef_.shape = (1, n_features)
      - Multiclass: coef_.shape = (n_classes, n_features)

    Returns:
        dict: class_label -> DataFrame(features & weights)
    """
    coef = estimator.coef_
    classes = np.array(classes)
    n_coef_rows, _ = coef.shape

    # Binary special case
    if n_coef_rows == 1 and len(classes) == 2:
        return _extract_binary_top_features(
            coef=coef,
            classes=classes,
            feature_names=feature_names,
            top_n=top_n,
        )

    # Multiclass general case
    return _extract_multiclass_top_features(
        coef=coef,
        classes=classes,
        feature_names=feature_names,
        top_n=top_n,
    )


def save_top_features_to_excel(
    df_features,
    output_dir,
    filename="top_features.xlsx",
):
    """
    Save DataFrame of top features (for any model) to Excel.

    Args:
        df_features: DataFrame with feature importances
        output_dir: directory
        filename: Excel filename

    Returns:
        output_path: full path
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df_features.to_excel(output_path, index=False)
    print(f"\n✅ Saved top features to Excel: {output_path}")
    return output_path
