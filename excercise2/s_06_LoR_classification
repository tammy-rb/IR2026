"""
s_06_LoR_classification.py

Supervised classification of UK vs US parliamentary debates
using Logistic Regression on the BM25 (lemmas) matrix.

Steps:
- Load BM25 matrix + metadata (feature names, filenames, true_labels).
- Run 10-fold cross-validation with Logistic Regression and report:
    * Accuracy
    * Precision (macro)
    * Recall (macro)
    * F1 (macro)
- Train a final Logistic Regression model on ALL data.
- Extract the top-N most important features per class (UK / US)
  based on the absolute value of the learned coefficients.
- Save the feature importances to an Excel file for further analysis.
"""

import os

from sklearn.linear_model import LogisticRegression
import pandas as pd

from s_06_classification_utils import (
    CLASSIFICATION_ROOT_DIR,
    load_bm25_and_labels,
    run_cross_validation,
    save_cv_results_to_excel,
    extract_top_features,
    save_top_features_to_excel,
)

# Directory where we will save supervised results for Logistic Regression
SUPERVISED_ROOT_DIR = os.path.join(CLASSIFICATION_ROOT_DIR, "LoR")


# ============================
# Train final model
# ============================

def train_final_LoR_model(X, y):
    """
    Train Logistic Regression on ALL data (no split).
    This model will be used to extract feature importances.

    Args:
        X: feature matrix
        y: labels array

    Returns:
        clf: fitted LogisticRegression instance 
        classes_: np.ndarray of class labels in the same order as clf.coef_
    """
    print("\n" + "=" * 70)
    print("TRAINING FINAL LOGISTIC REGRESSION MODEL ON ALL DATA")
    print("=" * 70)

    clf = LogisticRegression(
        penalty="l2", # prevent overfitting, keep stable weights
        solver="liblinear", # binary classification
        max_iter=1000,
        class_weight=None,  # the clusters are roughly balanced
    )
    clf.fit(X, y)

    print("  → Training completed.")
    print(f"  → Classes_: {clf.classes_}")
    print(f"  → Coefficients shape: {clf.coef_.shape}")

    return clf, clf.classes_


# ============================
# Main
# ============================

def main():
    print("=" * 70)
    print("LoR SUPERVISED CLASSIFICATION (UK vs US)")
    print("=" * 70)

    # 1. Load BM25 matrix + metadata and labels
    print("\n[1] Loading BM25 matrix and labels...")
    X, feature_names, filenames, y = load_bm25_and_labels()
    print(f"  → Loaded matrix with shape: {X.shape}")
    print(f"  → Number of documents: {len(y)}")
    print(f"  → Classes: {sorted(set(y))}")

    # 2. Run 10-fold cross-validation and report performance
    print("\n[2] Running 10-fold cross-validation for LoR...")
    lor_for_cv = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        class_weight=None,  
    )

    cv_results = run_cross_validation(
        estimator=lor_for_cv,
        X=X,
        y=y,
        n_splits=10,
        random_state=42,
        scoring=None,              # use default scoring from utils
        model_name="LoR",
    )
    print("\n[2] Saving cross-validation results to Excel...")
    save_cv_results_to_excel(cv_results, SUPERVISED_ROOT_DIR, "LoR_cv_results.xlsx")

    # 3. Train final model on ALL data to extract feature importances
    print("\n[3] Training final model on ALL data...")
    clf, classes = train_final_LoR_model(X, y)

    # 4. Extract top features per class using the generic utils function
    print("\n[4] Extracting top features per class...")
    features_dict = extract_top_features(
        estimator=clf,
        classes=classes,
        feature_names=feature_names,
        top_n=20,
    )

    # combine per-class DataFrames into a single DataFrame with a `class_label` column
    df_features = pd.concat(
        [df.assign(class_label=label) for label, df in features_dict.items()],
        ignore_index=True,
    )

    print("\nTop features per class (first few rows):")
    print(df_features.head(10))

    # 5. Save feature importances to Excel
    print("\n[5] Saving top features to Excel...")
    save_top_features_to_excel(
        df_features,
        SUPERVISED_ROOT_DIR,
        filename="LoR_top_features.xlsx",
    )


if __name__ == "__main__":
    main()
