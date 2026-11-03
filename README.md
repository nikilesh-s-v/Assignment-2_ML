# Dry Bean Multi-Class Classification — ML Assignment 2

**Name:** Nikilesh S V
**BITS ID:** 2026AJ05026
**Course:** M.Tech (AIML) — Machine Learning — Assignment 2

---

## a. Problem Statement

Classify dry bean grains into one of **seven cultivated varieties** using 16 geometric
and shape-based features extracted from grain images. This is a **multi-class
classification** problem. Five classifiers are trained on the same dataset, evaluated
with six metrics, and served through an interactive Streamlit web application that
loads the saved models and scores a user-uploaded test CSV **without retraining**.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository — *Dry Bean Dataset*
  (https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Instances:** 13,611  (≥ 500 required ✔)
- **Features:** 16 numeric features  (≥ 12 required ✔)
- **Target:** `Class` — 7 bean varieties: *BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ,
  SEKER, SIRA*
- **Missing values:** none
- **Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation,
  Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness,
  ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4

**Preprocessing:** string labels encoded with `LabelEncoder`; a stratified 80/20
train-test split; features standardised with `StandardScaler` (fit on the training set
only). The scaler and label encoder are saved and reused by the app so that uploaded
test data is transformed exactly as during training.

## c. GitHub Repository Link

`https://github.com/<your-username>/dry-bean-classifier`   *(replace with your repo URL after pushing)*

## d. Models Used

Five classifiers were trained on the same standardised dataset. Metrics below are on the
held-out 20% test set (2,723 rows). AUC uses macro one-vs-rest; Precision/Recall/F1 use
macro averaging (multi-class).

### Comparison Table

| ML Model Name           | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression     | 0.9214   | 0.9948 | 0.9354    | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree           | 0.8920   | 0.9450 | 0.9075    | 0.9090 | 0.9081 | 0.8696 |
| KNN                     | 0.9137   | 0.9858 | 0.9290    | 0.9245 | 0.9265 | 0.8956 |
| Naive Bayes             | 0.8979   | 0.9916 | 0.9112    | 0.9092 | 0.9091 | 0.8773 |
| Support Vector Machine  | 0.9221   | 0.9949 | 0.9359    | 0.9330 | 0.9343 | 0.9059 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Very strong linear baseline with the joint-highest AUC (~0.995); after scaling the seven bean classes are largely linearly separable, so a linear decision boundary already performs excellently. |
| **Decision Tree** | Lowest scores of the five. A single unpruned tree produces axis-aligned, high-variance boundaries and overfits the training set, generalising less smoothly than the margin/linear models. |
| **KNN** | Strong performance (k = 7). Because it is distance-based, standardisation is essential; on the scaled features it captures the local structure of each class well. |
| **Naive Bayes** | Solid despite its feature-independence assumption, which is only partially valid here (several shape factors are correlated). Fast to train and a reasonable probabilistic baseline. |
| **Support Vector Machine** | **Best overall** — highest Accuracy, F1, AUC and MCC. The RBF-kernel maximum-margin boundary handles the 7-class geometry best, edging out Logistic Regression. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** (macro F1 = 0.9343, MCC = 0.9059), with **Logistic Regression** a very close second (F1 = 0.9335). SVM is the recommended model for this dataset. |

## Repository Structure

```
project/
├── app.py                       # Streamlit application (loads models, no retraining)
├── train_models.py              # Trains all 5 models and saves artefacts
├── requirements.txt             # Pinned dependencies
├── README.md                    # This file
├── test_data.csv                # Held-out test set for upload into the app
├── notebook/
│   └── dry_bean_classification.ipynb   # EDA + training + evaluation (with outputs)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── scaler.joblib            # StandardScaler (preprocessing artefact)
    ├── label_encoder.joblib     # LabelEncoder (preprocessing artefact)
    ├── metadata.json            # feature names, classes, model file map
    └── metrics_comparison.csv   # saved comparison table
```

## Live Streamlit App

`https://<your-app-name>.streamlit.app`   *(replace with your deployed URL)*

The app provides:
- **CSV upload** of test data
- **Model-selection dropdown** across the five classifiers
- **Evaluation-metrics display** (Accuracy, AUC, Precision, Recall, F1, MCC)
- **Confusion matrix + classification report**

The app loads the saved models and the saved scaler, transforms the uploaded test data,
predicts, and reports metrics — it does **not** retrain on the uploaded data.

## How to Run Locally

```bash
pip install -r requirements.txt
python train_models.py        # (optional) regenerate the model/ artefacts
streamlit run app.py          # launch the app, then upload test_data.csv
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. **New app** → select the repo → branch `main` → main file `app.py` → **Deploy**.
4. Once live, open the app and upload `test_data.csv` to see the results.
