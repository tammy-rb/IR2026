"""
s_07_ann_classification.py

Supervised classification of UK vs US parliamentary debates
using feed-forward ANN (Keras) on the BM25 (lemmas) matrix.

- Load BM25 matrix + labels (UK / US)
- Train/test split: 80% train, 20% test
- From the train part: 10% validation
- Two ANN topologies:
    1) Hidden layers with ReLU
    2) Hidden layers with GELU
- Architecture (both):
    * "Embedding-like" projection layer from BM25 input
    * Hidden layer: 10 units (activation)
    * Hidden layer: 10 units (activation)
    * Hidden layer: 7 units (activation)
    * Output: softmax

Training:
    - max epochs = 15
    - batch size = 16
    - EarlyStopping on val_accuracy (patience=3, restore_best_weights=True)
    - ModelCheckpoint saving the best validation model

Evaluation:
    - Report accuracy, precision (macro), recall (macro), F1 (macro) on test set
    - Save per-model JSON + global ANN summary JSON
"""

import os
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from s_06_classification_utils import (
    CLASSIFICATION_ROOT_DIR,
    load_bm25_and_labels,
)

# ============================
# Directories
# ============================

ANN_ROOT_DIR = os.path.join(CLASSIFICATION_ROOT_DIR, "ANN")
ANN_RELU_DIR = os.path.join(ANN_ROOT_DIR, "relu")
ANN_GELU_DIR = os.path.join(ANN_ROOT_DIR, "gelu")

os.makedirs(ANN_ROOT_DIR, exist_ok=True)
os.makedirs(ANN_RELU_DIR, exist_ok=True)
os.makedirs(ANN_GELU_DIR, exist_ok=True)


# ============================
# Data loading & splitting
# ============================

def load_data_dense():
    """
    Load BM25 sparse matrix + labels, convert to dense float32,
    encode labels as integers.

    Returns:
        X_bm25: np.ndarray of shape (n_docs, n_features), dtype float32
        y_enc: np.ndarray of shape (n_docs,), int labels
        label_encoder: fitted LabelEncoder instance
    """
    X_sparse, feature_names, filenames, y_str = load_bm25_and_labels()
    print(f"  → Loaded BM25 matrix with shape: {X_sparse.shape}")
    print(f"  → Number of documents: {len(y_str)}")
    print(f"  → Classes (string): {sorted(set(y_str))}")

    # Convert sparse BM25 to dense float32 (documents x features)
    X_bm25 = X_sparse.toarray().astype("float32")

    # Encode labels ("UK"/"US") → integers 0/1
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    print(f"  → Encoded classes: {list(le.classes_)} (mapped to {sorted(set(y_enc))})")

    return X_bm25, y_enc, le


def make_splits(X, y, random_state=42):
    """
    Create train/val/test splits.

    - First: split into 80% train_full, 20% test
    - Then: split train_full into 90% train, 10% val

    Stratified splits to keep class balance.

    Returns:
        X_train, y_train
        X_val, y_val
        X_test, y_test
    """
    # 1) Train/test split: 80% / 20%
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,  # keep class balance
    )

    # 2) From train_full: train/validation split: 90% / 10%
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1,  # 10% of the 80% → 8% of total as validation
        random_state=random_state,
        stratify=y_train_full,
    )

    print("\nData splits:")
    print(f"  Train size     : {X_train.shape[0]}")
    print(f"  Validation size: {X_val.shape[0]}")
    print(f"  Test size      : {X_test.shape[0]}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================
# Model builder (generic)
# ============================

def build_ann_model(
    input_dim,
    num_classes=2,
    embedding_dim=128,
    hidden_activation="relu",
    model_name="ann_model",
):
    """
    Build ANN model with configurable hidden activation.

    Architecture:
        Input (BM25 vector) -> Dense("embedding") ->
        Dense(10, activation) -> Dense(10, activation) -> Dense(7, activation) ->
        Dense(num_classes, softmax)

    Args:
        input_dim: number of input features (BM25 dimension)
        num_classes: number of output classes
        embedding_dim: size of the first projection layer
        hidden_activation: activation for hidden layers (string or callable)
        model_name: name of the Keras model

    Returns:
        Keras Model instance
    """
    # Input layer: shape = (num_features,)
    inputs = keras.Input(shape=(input_dim,), name="bm25_input")

    # "Embedding-like" projection from high-dimensional BM25 to lower-dimensional space
    x = layers.Dense(
        embedding_dim,
        activation=None,
        name="embedding_dense",
    )(inputs)

    # Hidden layers with chosen activation
    x = layers.Dense(10, activation=hidden_activation, name="hidden_1")(x)
    x = layers.Dense(10, activation=hidden_activation, name="hidden_2")(x)
    x = layers.Dense(7, activation=hidden_activation, name="hidden_3")(x)

    # Output layer: softmax over num_classes
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="output_softmax",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


# ============================
# Training & evaluation helper
# ============================

def train_and_evaluate(
    model,
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    output_dir,
    max_epochs=15,
    batch_size=16,
):
    """
    Compile, train with EarlyStopping + ModelCheckpoint,
    and evaluate on the test set.

    Prints precision, recall, F1, accuracy (macro-averaged PRF).

    Also saves a per-model JSON metadata file in output_dir.

    Returns:
        metrics_dict: {
            "model_name": ...,
            "accuracy": ...,
            "precision_macro": ...,
            "recall_macro": ...,
            "f1_macro": ...,
            "train_size": ...,
            "val_size": ...,
            "test_size": ...,
            "model_path": ...
        }
    """
    print("\n" + "=" * 70)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 70)
    # Print model summary
    model.summary()

    # Compile model: define optimizer, loss function, and metrics to track
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # cost function
        metrics=["accuracy"],
    )

    # Callbacks
    # Path to save the best-validation model
    ckpt_path = os.path.join(output_dir, f"{model_name}.keras")

    # ModelCheckpoint: save the model that achieves the best val_accuracy
    checkpoint_cb = callbacks.ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        save_best_only=True,   # only save model better than previous best
        mode="max",            # we want to maximize val_accuracy
        verbose=1,
    )

    # EarlyStopping: stop training if no improvement on validation for 'patience' epochs
    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,            # stop after 3 epochs with no improvement
        restore_best_weights=True,
        verbose=1,
    )

    # Train
    training_history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, early_stop_cb],
        verbose=2,
    )

    # Evaluate on test
    print("\nEvaluating on TEST set...")
    y_proba = model.predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("\nTEST METRICS:")
    print(f"  accuracy       : {acc:.3f}")
    print(f"  precision_macro: {prec:.3f}")
    print(f"  recall_macro   : {rec:.3f}")
    print(f"  f1_macro       : {f1:.3f}")

    metrics_dict = {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(X_test.shape[0]),
        "model_path": ckpt_path,  # path to the saved best-validation model
    }

    # Save per-model JSON metadata
    json_path = os.path.join(output_dir, f"{model_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"\nSaved results metadata → {json_path}")

    return metrics_dict


# ============================
# Summary saver
# ============================

def save_ann_summary(relu_metrics, gelu_metrics, output_dir=ANN_ROOT_DIR):
    """
    Save a global summary JSON for the ANN models in classification/ANN.

    Args:
        relu_metrics: dict returned from train_and_evaluate for ann_relu
        gelu_metrics: dict returned from train_and_evaluate for ann_gelu
        output_dir: base ANN directory (classification/ANN)

    Returns:
        summary_path: full path to the saved JSON file
    """
    # Collect under model names
    models = {
        "ann_relu": relu_metrics,
        "ann_gelu": gelu_metrics,
    }

    # Decide best model by accuracy
    best_name = max(models.keys(), key=lambda name: models[name]["accuracy"])
    best_model_info = {
        "model_name": best_name,
        "accuracy": float(models[best_name]["accuracy"]),
        "model_path": models[best_name]["model_path"],
    }

    summary = {
        "models": models,
        "best_model_by_accuracy": best_model_info,
    }

    summary_path = os.path.join(output_dir, "ann_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSaved ANN summary JSON → {summary_path}")
    return summary_path


# ============================
# Main
# ============================

def main():
    print("=" * 70)
    print("ANN SUPERVISED CLASSIFICATION (UK vs US) - KERAS")
    print("=" * 70)

    # 1. Load data (BM25 -> dense array) and encode labels
    print("\n[1] Loading BM25 matrix and labels...")
    X_bm25, y_enc, label_encoder = load_data_dense()
    n_docs, n_features = X_bm25.shape
    num_classes = len(np.unique(y_enc))
    print(f"  → n_docs={n_docs}, n_features={n_features}, num_classes={num_classes}")

    # 2. Create train/val/test splits
    print("\n[2] Creating train/val/test splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = make_splits(X_bm25, y_enc)

    # 3. Build models (ReLU & GELU) using the generic builder
    print("\n[3] Building ANN models...")
    model_relu = build_ann_model(
        input_dim=n_features,
        num_classes=num_classes,
        embedding_dim=128,
        hidden_activation="relu",
        model_name="ann_relu",
    )

    model_gelu = build_ann_model(
        input_dim=n_features,
        num_classes=num_classes,
        embedding_dim=128,
        hidden_activation=tf.keras.activations.gelu,
        model_name="ann_gelu",
    )

    # 4. Train & evaluate ReLU model
    print("\n[4] Training ReLU-based ANN...")
    relu_metrics = train_and_evaluate(
        model=model_relu,
        model_name="ann_relu",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        output_dir=ANN_RELU_DIR,
        max_epochs=15,
        batch_size=16,
    )

    # 5. Train & evaluate GELU model
    print("\n[5] Training GELU-based ANN...")
    gelu_metrics = train_and_evaluate(
        model=model_gelu,
        model_name="ann_gelu",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        output_dir=ANN_GELU_DIR,
        max_epochs=15,
        batch_size=16,
    )

    # 6. Summary comparison to console
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON (TEST SET)")
    print("=" * 70)
    print("ReLU-based ANN:")
    for k, v in relu_metrics.items():
        if k in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            print(f"  {k:15s}: {v:.3f}")

    print("\nGELU-based ANN:")
    for k, v in gelu_metrics.items():
        if k in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            print(f"  {k:15s}: {v:.3f}")

    # 7. Save global ANN summary JSON
    save_ann_summary(relu_metrics, gelu_metrics, output_dir=ANN_ROOT_DIR)


if __name__ == "__main__":
    main()
