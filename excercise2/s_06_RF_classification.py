"""
s_06_RF_classification.py

Supervised classification of UK vs US parliamentary debates
using Random Forest on the BM25 (lemmas) matrix.

Steps:
- Load BM25 matrix + metadata (feature names, filenames, true_labels).
- Run 10-fold cross-validation with Random Forest and report:
    * Accuracy
    * Precision (macro)
    * Recall (macro)
    * F1 (macro)
- Train a final Random Forest model on ALL data.
- Extract the top-N most important features per class (UK / US)
  based on the Gini importance (feature_importances_).
- Save the feature importances to an Excel file for further analysis.

Why these hyperparameters?
- n_estimators=200: Good balance between performance and speed (100-300 is standard)
- max_depth=None: Allow trees to grow until leaves are pure (RF handles overfitting well)
- min_samples_split=5: Require at least 5 samples to split (prevents too-granular splits)
- min_samples_leaf=2: Require at least 2 samples per leaf (smoothing)
- max_features='sqrt': Use sqrt(n_features) per split (standard for classification)
- bootstrap=True: Use bootstrap sampling (standard RF behavior)
- class_weight=None: Balanced classes (UK ~330, US ~360)
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from s_06_classification_utils import (
    CLASSIFICATION_ROOT_DIR,
    load_bm25_and_labels,
    run_cross_validation,
    save_cv_results_to_excel,
    save_top_features_to_excel,
)

# Directory where we will save supervised results for Random Forest
SUPERVISED_ROOT_DIR = os.path.join(CLASSIFICATION_ROOT_DIR, "RF")


# ============================
# Extract top features for RF
# ============================

def extract_rf_top_features(clf, feature_names, top_n=20):
    """
    Extract top-N most important features for Random Forest.
    
    Random Forest provides feature_importances_ (Gini importance):
    - Measures how much each feature contributes to decreasing impurity
    - Higher values = more important features
    - Sum of all importances = 1.0
    
    Note: RF gives GLOBAL importance, not per-class like linear models.
    However, we can still analyze which features are most discriminative overall.
    
    Args:
        clf: fitted RandomForestClassifier
        feature_names: list of feature names
        top_n: number of top features to extract
    
    Returns:
        DataFrame with columns: feature, importance, rank
    """
    feature_names = np.array(feature_names)
    importances = clf.feature_importances_
    
    # Sort by importance (descending)
    sorted_indices = np.argsort(-importances)
    top_indices = sorted_indices[:top_n]
    
    df_features = pd.DataFrame([
        {
            "feature": feature_names[idx],
            "importance": float(importances[idx]),
            "rank": rank + 1,
        }
        for rank, idx in enumerate(top_indices)
    ])
    
    return df_features


def extract_rf_top_features_per_class(clf, classes, feature_names, X, y, top_n=20):
    """
    Extract top-N features per class by analyzing feature importance
    in the context of each class.
    
    Strategy:
    For each class, we look at the feature importances but weight them
    by how much they appear in samples of that class vs other classes.
    
    This gives us a class-specific view of feature importance.
    
    Args:
        clf: fitted RandomForestClassifier
        classes: array of class labels
        feature_names: list of feature names
        X: feature matrix
        y: labels
        top_n: number of top features per class
    
    Returns:
        dict: class_label -> DataFrame(feature, importance, class_weighted_importance)
    """
    feature_names = np.array(feature_names)
    classes = np.array(classes)
    global_importances = clf.feature_importances_
    
    grouped = {}
    
    for class_label in classes:
        # Get indices of samples belonging to this class
        class_mask = (y == class_label)
        
        # For sparse matrices, we need to convert to array to use fancy indexing
        if hasattr(X, 'toarray'):
            X_class = X[class_mask].toarray()
        else:
            X_class = X[class_mask]
        
        # Calculate mean feature values for this class
        mean_feature_values = np.array(X_class.mean(axis=0)).flatten()
        
        # Weight global importance by class-specific feature prevalence
        # Features that are both important AND prevalent in this class get high scores
        class_weighted_importance = global_importances * mean_feature_values
        
        # Normalize to make it comparable
        if class_weighted_importance.sum() > 0:
            class_weighted_importance = class_weighted_importance / class_weighted_importance.sum()
        
        # Get top-N features for this class
        top_indices = np.argsort(-class_weighted_importance)[:top_n]
        
        grouped[class_label] = pd.DataFrame([
            {
                "feature": feature_names[idx],
                "global_importance": float(global_importances[idx]),
                "class_weighted_importance": float(class_weighted_importance[idx]),
                "rank": rank + 1,
            }
            for rank, idx in enumerate(top_indices)
        ])
    
    return grouped


# ============================
# Train final model
# ============================

def get_rf_model():
    """
    Create Random Forest model with carefully chosen hyperparameters.
    
    Returns:
        RandomForestClassifier instance
    """
    return RandomForestClassifier(
        n_estimators=200,      # Number of trees: more trees = more stable, 200 is good balance
        max_depth=None,        # Let trees grow fully (RF prevents overfitting via ensemble)
        min_samples_split=5,   # Minimum samples required to split: prevents over-granular splits
        min_samples_leaf=2,    # Minimum samples per leaf: smoothing, prevents noise
        max_features='sqrt',   # Features per split: sqrt(n_features) is standard for classification
        bootstrap=True,        # Use bootstrap sampling (standard RF)
        class_weight=None,     # Classes are balanced (UK ~330, US ~360)
        random_state=42,       # For reproducibility
        n_jobs=-1,            # Use all CPU cores for speed
        verbose=0,            # Suppress output during training
    )


def train_final_RF_model(X, y):
    """
    Train Random Forest on ALL data (no split).
    This model will be used to extract feature importances.
    
    Args:
        X: feature matrix (BM25 sparse matrix)
        y: labels array
    
    Returns:
        clf: fitted RandomForestClassifier instance
        classes_: np.ndarray of class labels
    """
    print("\n" + "=" * 70)
    print("TRAINING FINAL RANDOM FOREST MODEL ON ALL DATA")
    print("=" * 70)
    
    clf = get_rf_model()
    clf.fit(X, y)
    
    print("  → Training completed.")
    print(f"  → Classes_: {clf.classes_}")
    print(f"  → Number of trees: {clf.n_estimators}")
    print(f"  → Number of features: {clf.n_features_in_}")
    
    return clf, clf.classes_


# ============================
# Main
# ============================

def main():
    print("=" * 70)
    print("RANDOM FOREST SUPERVISED CLASSIFICATION (UK vs US)")
    print("=" * 70)
    
    os.makedirs(SUPERVISED_ROOT_DIR, exist_ok=True)
    
    # 1. Load BM25 matrix + metadata and labels
    print("\n[1] Loading BM25 matrix and labels...")
    X, feature_names, filenames, y = load_bm25_and_labels()
    print(f"  → Loaded matrix with shape: {X.shape}")
    print(f"  → Number of documents: {len(y)}")
    print(f"  → Classes: {sorted(set(y))}")
    
    # 2. Run 10-fold cross-validation and report performance
    print("\n[2] Running 10-fold cross-validation for Random Forest...")
    rf_for_cv = get_rf_model()
    
    cv_results = run_cross_validation(
        estimator=rf_for_cv,
        X=X,
        y=y,
        n_splits=10,
        random_state=42,
        scoring=None,  # use default scoring from utils
        model_name="Random Forest",
    )
    
    print("\n[2] Saving cross-validation results to Excel...")
    save_cv_results_to_excel(
        cv_results,
        SUPERVISED_ROOT_DIR,
        "RF_cv_results.xlsx"
    )
    
    # 3. Train final model on ALL data to extract feature importances
    print("\n[3] Training final model on ALL data...")
    clf, classes = train_final_RF_model(X, y)
    
    # 4. Extract top features - we'll do both global and per-class
    print("\n[4a] Extracting global top features...")
    df_global_features = extract_rf_top_features(
        clf=clf,
        feature_names=feature_names,
        top_n=20,
    )
    
    print("\nTop 20 global features:")
    print(df_global_features)
    
    print("\n[4b] Extracting top features per class (class-weighted)...")
    features_dict = extract_rf_top_features_per_class(
        clf=clf,
        classes=classes,
        feature_names=feature_names,
        X=X,
        y=y,
        top_n=20,
    )
    
    # Combine per-class DataFrames into a single DataFrame with a `class_label` column
    df_per_class_features = pd.concat(
        [df.assign(class_label=label) for label, df in features_dict.items()],
        ignore_index=True,
    )
    
    print("\nTop features per class (first few rows):")
    print(df_per_class_features.head(10))
    
    # 5. Save feature importances to Excel
    # We'll save both global and per-class features in separate sheets
    print("\n[5] Saving top features to Excel...")
    
    # Save global features
    save_top_features_to_excel(
        df_global_features,
        SUPERVISED_ROOT_DIR,
        filename="RF_top_features_global.xlsx",
    )
    
    # Save per-class features
    save_top_features_to_excel(
        df_per_class_features,
        SUPERVISED_ROOT_DIR,
        filename="RF_top_features_per_class.xlsx",
    )
    
    print("\n" + "=" * 70)
    print("RANDOM FOREST CLASSIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()