"""
s_06_SVM_classification.py

Supervised classification of UK vs US parliamentary debates
using linear SVM (LinearSVC) on the BM25 (lemmas) matrix.
"""

import os
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize   # ← NEW

from s_06_classification_utils import (
    CLASSIFICATION_ROOT_DIR,
    load_bm25_and_labels,
    run_cross_validation,
    save_cv_results_to_excel,
    extract_top_features,
    save_top_features_to_excel,
)

SUPERVISED_ROOT_DIR = os.path.join(CLASSIFICATION_ROOT_DIR, "SVM")


# ============================
# Train final model
# ============================

def getModel():
    return LinearSVC(
        penalty="l2", # prevent overfitting, punish large weights by normalizing
        loss="squared_hinge", # standard for SVM, punish misclassifications (squared for smoothness) 
        C=1.0, # how much we accept misclassifications on training data, balance margin size vs training error (overfitting)
        class_weight=None, # balanced clusters. mistake cost the same for any cluster
        max_iter=5000,
    )


def train_final_SVM_model(X, y, clf=None):
    '''
    Train final linear SVM model on all data.
    Args:
        X: Feature matrix (sparse or dense).
        y: Labels.
    Returns the trained model and the class labels.
    '''

    print("\n" + "=" * 70)
    print("TRAINING FINAL LINEAR SVM MODEL ON ALL DATA")
    print("=" * 70)

    if clf is None:
        clf = getModel()
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
    print("SVM SUPERVISED CLASSIFICATION (UK vs US)")
    print("=" * 70)

    os.makedirs(SUPERVISED_ROOT_DIR, exist_ok=True)

    # 1. Load BM25 matrix + metadata and labels
    print("\n[1] Loading BM25 matrix and labels...")
    X, feature_names, filenames, y = load_bm25_and_labels()
    print(f"  → Loaded matrix with shape: {X.shape}")

    # 1b.  Normalize vectors (L2)
    # After normalization, dot-products approximate cosine similarity
    print("\n[1b] Normalizing BM25 vectors (L2 norm)...")
    X = normalize(X, norm='l2')   
    print("  → Normalization completed.")

    print(f"  → Number of documents: {len(y)}")
    print(f"  → Classes: {sorted(set(y))}")

    # 2. 10-fold cross-validation
    print("\n[2] Running 10-fold cross-validation for SVM...")

    svm_for_cv = getModel()

    cv_results = run_cross_validation(
        estimator=svm_for_cv,
        X=X,
        y=y,
        n_splits=10,
        random_state=42,
        scoring=None,
        model_name="SVM",
    )

    print("\n[2] Saving cross-validation results to Excel...")
    save_cv_results_to_excel(
        cv_results,
        SUPERVISED_ROOT_DIR,
        "SVM_cv_results.xlsx",
    )

    # 3. Final model for feature importances
    print("\n[3] Training final model on ALL data...")
    clf, classes = train_final_SVM_model(X, y, svm_for_cv)

    # 4. Top features
    print("\n[4] Extracting top features per class...")
    features_dict = extract_top_features(
        estimator=clf,
        classes=classes,
        feature_names=feature_names,
        top_n=20,
    )

    df_features = pd.concat(
        [df.assign(class_label=label) for label, df in features_dict.items()],
        ignore_index=True,
    )

    print("\nTop features per class (first few rows):")
    print(df_features.head(10))

    # 5. Save features to Excel
    print("\n[5] Saving top features to Excel...")
    save_top_features_to_excel(
        df_features,
        SUPERVISED_ROOT_DIR,
        filename="SVM_top_features.xlsx",
    )


if __name__ == "__main__":
    main()
