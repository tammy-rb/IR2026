"""
s_06_NB_classification.py

Supervised classification of UK vs US parliamentary debates
using Naive Bayes (MultinomialNB) on the BM25 (lemmas) matrix.

Steps:
- Load BM25 matrix + metadata (feature names, filenames, true_labels).
- Run 10-fold cross-validation with Multinomial Naive Bayes and report:
    * Accuracy
    * Precision (macro)
    * Recall (macro)
    * F1 (macro)
- Train a final Naive Bayes model on ALL data.
- Extract the top-N most important features per class (UK / US)
  based on the log-probability differences between classes.
- Save the feature importances to an Excel file for further analysis.

Why MultinomialNB?
- Best suited for text classification with count-based features (like BM25)
- Works well with sparse matrices
- Assumes feature independence (naive assumption but works well in practice)
- Fast training and prediction

Note on feature importance:
Unlike linear models (LoR/SVM), Naive Bayes doesn't have coefficient weights.
Instead, we use the difference in log-probabilities: log(P(feature|UK)) - log(P(feature|US))
to identify which features are most discriminative for each class.
"""

import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from s_06_classification_utils import (
    CLASSIFICATION_ROOT_DIR,
    load_bm25_and_labels,
    run_cross_validation,
    save_cv_results_to_excel,
    save_top_features_to_excel,
)

# Directory where we will save supervised results for Naive Bayes
SUPERVISED_ROOT_DIR = os.path.join(CLASSIFICATION_ROOT_DIR, "NB")


# ============================
# Extract top features for NB
# ============================

def extract_nb_top_features(clf, classes, feature_names, top_n=20):
    """
    Extract top-N most discriminative features for Naive Bayes.
    
    For NB, we don't have coefficients like in linear models.
    Instead, we use feature_log_prob_ which gives log P(feature|class).
    
    Strategy:
    - For each class, compute the difference: log P(feature|this_class) - log P(feature|other_class)
    - Positive differences mean the feature is more indicative of this class
    - We take the top-N features with highest absolute differences
    
    Args:
        clf: fitted MultinomialNB model
        classes: array of class labels
        feature_names: list of feature names
        top_n: number of top features to extract per class
    
    Returns:
        dict: class_label -> DataFrame(feature, log_prob_diff, abs_log_prob_diff)
    """
    feature_names = np.array(feature_names)
    classes = np.array(classes)
    
    # feature_log_prob_ shape: (n_classes, n_features)
    # Each row is log P(feature|class) for that class
    log_probs = clf.feature_log_prob_
    
    grouped = {}
    
    # For binary classification (UK vs US)
    if len(classes) == 2:
        class_0, class_1 = classes[0], classes[1]
        log_prob_0 = log_probs[0]  # log P(feature|class_0)
        log_prob_1 = log_probs[1]  # log P(feature|class_1)
        
        # Difference: how much more likely is feature in class_1 vs class_0
        diff = log_prob_1 - log_prob_0
        
        # Top features for class_1 (most positive differences)
        top_indices_class_1 = np.argsort(-diff)[:top_n]
        grouped[class_1] = pd.DataFrame([
            {
                "feature": feature_names[idx],
                "log_prob_diff": float(diff[idx]),
                "abs_log_prob_diff": abs(float(diff[idx])),
            }
            for idx in top_indices_class_1
        ])
        
        # Top features for class_0 (most negative differences)
        top_indices_class_0 = np.argsort(diff)[:top_n]
        grouped[class_0] = pd.DataFrame([
            {
                "feature": feature_names[idx],
                "log_prob_diff": float(diff[idx]),
                "abs_log_prob_diff": abs(float(diff[idx])),
            }
            for idx in top_indices_class_0
        ])
    
    else:
        # Multiclass case (if needed in future)
        # For each class, compare against average of all other classes
        for class_idx, class_label in enumerate(classes):
            log_prob_this_class = log_probs[class_idx]
            
            # Average log prob of all other classes
            other_classes_mask = np.ones(len(classes), dtype=bool)
            other_classes_mask[class_idx] = False
            log_prob_others = np.mean(log_probs[other_classes_mask], axis=0)
            
            diff = log_prob_this_class - log_prob_others
            
            # Take top-N by absolute difference
            top_indices = np.argsort(-np.abs(diff))[:top_n]
            
            grouped[class_label] = pd.DataFrame([
                {
                    "feature": feature_names[idx],
                    "log_prob_diff": float(diff[idx]),
                    "abs_log_prob_diff": abs(float(diff[idx])),
                }
                for idx in top_indices
            ])
    
    return grouped


# ============================
# Train final model
# ============================

def train_final_NB_model(X, y):
    """
    Train Multinomial Naive Bayes on ALL data (no split).
    This model will be used to extract feature importances.
    
    Args:
        X: feature matrix (BM25 sparse matrix)
        y: labels array
    
    Returns:
        clf: fitted MultinomialNB instance
        classes_: np.ndarray of class labels
    """
    print("\n" + "=" * 70)
    print("TRAINING FINAL NAIVE BAYES MODEL ON ALL DATA")
    print("=" * 70)
    
    clf = MultinomialNB(
        alpha=1.0,  # Laplace smoothing to handle zero probabilities
                    # alpha=1.0 is standard, prevents overfitting to training data
        fit_prior=True,  # Learn class priors from data (recommended)
    )
    clf.fit(X, y)
    
    print("  → Training completed.")
    print(f"  → Classes_: {clf.classes_}")
    print(f"  → Feature log probabilities shape: {clf.feature_log_prob_.shape}")
    
    return clf, clf.classes_


# ============================
# Main
# ============================

def main():
    print("=" * 70)
    print("NAIVE BAYES SUPERVISED CLASSIFICATION (UK vs US)")
    print("=" * 70)
    
    os.makedirs(SUPERVISED_ROOT_DIR, exist_ok=True)
    
    # 1. Load BM25 matrix + metadata and labels
    print("\n[1] Loading BM25 matrix and labels...")
    X, feature_names, filenames, y = load_bm25_and_labels()
    print(f"  → Loaded matrix with shape: {X.shape}")
    print(f"  → Number of documents: {len(y)}")
    print(f"  → Classes: {sorted(set(y))}")
    
    # 2. Run 10-fold cross-validation and report performance
    print("\n[2] Running 10-fold cross-validation for Naive Bayes...")
    nb_for_cv = MultinomialNB(
        alpha=1.0,
        fit_prior=True,
    )
    
    cv_results = run_cross_validation(
        estimator=nb_for_cv,
        X=X,
        y=y,
        n_splits=10,
        random_state=42,
        scoring=None,  # use default scoring from utils
        model_name="Naive Bayes",
    )
    
    print("\n[2] Saving cross-validation results to Excel...")
    save_cv_results_to_excel(
        cv_results,
        SUPERVISED_ROOT_DIR,
        "NB_cv_results.xlsx"
    )
    
    # 3. Train final model on ALL data to extract feature importances
    print("\n[3] Training final model on ALL data...")
    clf, classes = train_final_NB_model(X, y)
    
    # 4. Extract top features per class using NB-specific method
    print("\n[4] Extracting top features per class...")
    features_dict = extract_nb_top_features(
        clf=clf,
        classes=classes,
        feature_names=feature_names,
        top_n=20,
    )
    
    # Combine per-class DataFrames into a single DataFrame with a `class_label` column
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
        filename="NB_top_features.xlsx",
    )
    
    print("\n" + "=" * 70)
    print("NAIVE BAYES CLASSIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()