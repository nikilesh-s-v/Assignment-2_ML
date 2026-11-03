"""
Dry Bean Classification - model training script.

Trains five classifiers on the UCI Dry Bean dataset, evaluates them with six
metrics, prints a comparison table, and saves the trained models plus the
preprocessing artefacts (scaler + label encoder) for the Streamlit app.

Author: Nikilesh S V (BITS ID: 2026AJ05026)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef,
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_XLSX = os.path.join(HERE, "_drybean", "DryBeanDataset", "Dry_Bean_Dataset.xlsx")
MODEL_DIR = os.path.join(HERE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "Class"


def load_data():
    """Load the Dry Bean dataset from the UCI Excel file."""
    df = pd.read_excel(DATA_XLSX)
    return df


def build_models():
    """Return the five classifiers. probability=True lets SVM output scores for AUC."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
    }


def evaluate(model, X_test, y_test, n_classes):
    """Compute the six required metrics for a fitted model."""
    y_pred = model.predict(X_test)

    # AUC needs probability / score estimates; use one-vs-rest macro for multi-class.
    try:
        y_proba = model.predict_proba(X_test)
        if n_classes == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan

    avg = "binary" if n_classes == 2 else "macro"
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_test, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    df = load_data()
    print(f"Dataset shape: {df.shape}  ({df.shape[1] - 1} features, {df.shape[0]} instances)")

    X = df.drop(columns=[TARGET])
    y_raw = df[TARGET]
    feature_names = list(X.columns)

    # Encode the string class labels to integers.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)

    # Stratified split keeps the class balance in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features - essential for KNN, SVM and Logistic Regression.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        results[name] = evaluate(model, X_test_scaled, y_test, n_classes)
        # Persist each trained model.
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(MODEL_DIR, fname))
        print(f"  trained + saved: {name}")

    # Save shared preprocessing artefacts and metadata.
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump({
            "feature_names": feature_names,
            "target": TARGET,
            "classes": list(label_encoder.classes_),
            "n_classes": n_classes,
            "model_files": {n: n.lower().replace(" ", "_") + ".joblib" for n in models},
        }, f, indent=2)

    # Comparison table.
    table = pd.DataFrame(results).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    table = table.round(4)
    print("\n=== Model Comparison (on held-out test set) ===")
    print(table.to_string())
    table.to_csv(os.path.join(MODEL_DIR, "metrics_comparison.csv"))

    # Build test_data.csv (original, unscaled test rows + target) for the Streamlit app.
    test_df = X_test.copy()
    test_df[TARGET] = label_encoder.inverse_transform(y_test)
    test_df.to_csv(os.path.join(HERE, "test_data.csv"), index=False)
    print(f"\nSaved test_data.csv with {len(test_df)} rows.")

    winner = table["F1"].idxmax()
    print(f"\nBest model by macro F1: {winner} (F1 = {table.loc[winner, 'F1']:.4f})")


if __name__ == "__main__":
    main()
