"""
Dry Bean Classification - Streamlit demo app.

Loads the five pre-trained classifiers and evaluates them on a user-uploaded
test CSV. Models are NEVER retrained here - the app only loads saved artefacts,
scales the uploaded features with the saved scaler, predicts, and reports the
six evaluation metrics plus a confusion matrix / classification report.

Author: Nikilesh S V (BITS ID: 2026AJ05026)
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
TEST_CSV = os.path.join(HERE, "test_data.csv")

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")


@st.cache_data
def load_sample_bytes():
    """Read the bundled test_data.csv as raw bytes (for the download button)."""
    if os.path.exists(TEST_CSV):
        with open(TEST_CSV, "rb") as fh:
            return fh.read()
    return None


# --------------------------------------------------------------------------- #
# Loading of saved artefacts (cached so it runs once)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in meta["model_files"].items()
    }
    return meta, scaler, label_encoder, models


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Six required metrics, computed the same way as during training."""
    try:
        if y_proba is None:
            auc = np.nan
        elif n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan

    avg = "binary" if n_classes == 2 else "macro"
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Sidebar / header
# --------------------------------------------------------------------------- #
st.title("🫘 Dry Bean Multi-Class Classifier")
st.markdown(
    "Evaluate five pre-trained classifiers on your **test data**. "
    "The app loads saved models and **does not retrain** them."
)

try:
    meta, scaler, label_encoder, models = load_artifacts()
except FileNotFoundError:
    st.error("Model artefacts not found. Run `python train_models.py` first to "
             "create the `model/` folder.")
    st.stop()

feature_names = meta["feature_names"]
target_col = meta["target"]
class_names = meta["classes"]
n_classes = meta["n_classes"]

with st.sidebar:
    st.header("About")
    st.write("**Dataset:** UCI Dry Bean (7 classes, 16 features)")
    st.write(f"**Classes:** {', '.join(class_names)}")
    st.write("**Author:** Nikilesh S V")
    st.write("**BITS ID:** 2026AJ05026")
    st.divider()
    st.caption("Upload a CSV containing the 16 feature columns and the `Class` "
               "target column, or use the bundled `test_data.csv` below.")
    _sample_bytes = load_sample_bytes()
    if _sample_bytes is not None:
        st.download_button(
            "⬇️ Download test_data.csv",
            data=_sample_bytes,
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

# --------------------------------------------------------------------------- #
# Feature (a): CSV upload (or use the bundled test_data.csv)
# --------------------------------------------------------------------------- #
st.subheader("1. Provide test data (CSV)")
sample_available = os.path.exists(TEST_CSV)

source = st.radio(
    "Select the test data to evaluate:",
    options=["Bundled test_data.csv", "Upload my own CSV"],
    index=0 if sample_available else 1,
    horizontal=True,
)

if source == "Bundled test_data.csv":
    if not sample_available:
        st.warning("The bundled test_data.csv is not available; please upload a CSV instead.")
        st.stop()
    data = pd.read_csv(TEST_CSV)
    st.caption("Using the bundled **test_data.csv** (repository hold-out split).")
else:
    uploaded = st.file_uploader(
        "Upload a CSV with the feature columns + target column", type=["csv"]
    )
    if uploaded is None:
        st.info("Awaiting a CSV upload. The CSV needs the 16 feature columns and the "
                "`Class` target column.")
        st.stop()
    data = pd.read_csv(uploaded)

st.write(f"Test data: **{data.shape[0]} rows x {data.shape[1]} columns**")
st.dataframe(data.head(), use_container_width=True)

# Validate columns
missing = [c for c in feature_names if c not in data.columns]
if missing:
    st.error(f"Uploaded CSV is missing required feature columns: {missing}")
    st.stop()
if target_col not in data.columns:
    st.error(f"Uploaded CSV must contain the target column '{target_col}'.")
    st.stop()

X = data[feature_names]
y_true = label_encoder.transform(data[target_col].astype(str))
X_scaled = scaler.transform(X)

# --------------------------------------------------------------------------- #
# Feature (b): model selection dropdown
# --------------------------------------------------------------------------- #
st.subheader("2. Choose a model")
choice = st.selectbox("Model", list(models.keys()))
model = models[choice]

# Predictions (no retraining)
y_pred = model.predict(X_scaled)
try:
    y_proba = model.predict_proba(X_scaled)
except Exception:
    y_proba = None

# --------------------------------------------------------------------------- #
# Feature (c): evaluation metrics display
# --------------------------------------------------------------------------- #
st.subheader("3. Evaluation metrics")
metrics = compute_metrics(y_true, y_pred, y_proba, n_classes)
st.caption(f"Metrics for **{choice}** computed on {len(data)} test rows.")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
c2.metric("AUC", "n/a" if np.isnan(metrics["AUC"]) else f"{metrics['AUC']:.4f}")
c3.metric("Precision", f"{metrics['Precision']:.4f}")
c4.metric("Recall", f"{metrics['Recall']:.4f}")
c5.metric("F1", f"{metrics['F1']:.4f}")
c6.metric("MCC", f"{metrics['MCC']:.4f}")

# Metrics across ALL models on the uploaded test data
with st.expander("Show all five models on this test data"):
    rows = {}
    for name, mdl in models.items():
        yp = mdl.predict(X_scaled)
        try:
            pr = mdl.predict_proba(X_scaled)
        except Exception:
            pr = None
        rows[name] = compute_metrics(y_true, yp, pr, n_classes)
    st.dataframe(pd.DataFrame(rows).T.round(4), use_container_width=True)

# --------------------------------------------------------------------------- #
# Feature (d): confusion matrix + classification report
# --------------------------------------------------------------------------- #
st.subheader("4. Confusion matrix & classification report")
col_a, col_b = st.columns([1, 1])

with col_a:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {choice}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig)

with col_b:
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

st.success(f"Evaluation complete for **{choice}** on {len(data)} test rows.")

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.divider()
st.caption("Dry Bean Classifier - ML Assignment 2 | models are loaded from disk and "
           "never retrained on uploaded data.")
