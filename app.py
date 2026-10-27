"""
app.py — Dry Bean Classifier (BITS ML Assignment 2)
===================================================
An interactive Streamlit front-end that loads FIVE pre-trained classifiers and
evaluates them on a *test* CSV the user uploads. Nothing is re-trained here:
the app only loads the saved ``Pipeline`` objects, runs predictions, and reports
the six required metrics plus a confusion matrix / classification report.

Required features implemented (Assignment Step 6):
  (a) CSV upload for test data
  (b) model-selection dropdown
  (c) display of evaluation metrics
  (d) confusion matrix + classification report
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------------------------------- #
# Static configuration
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
TARGET_COLUMN = "Class"

# display name -> saved file stem (order defines the dropdown order)
MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}

BEAN_VARIETIES = [
    "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA",
]

st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Cached loaders — run once per session, not on every interaction
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the five pipelines + the label encoder from disk."""
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    models = {
        name: joblib.load(MODEL_DIR / f"{stem}.joblib")
        for name, stem in MODEL_FILES.items()
    }
    return models, label_encoder


def compute_metrics(pipeline, X, y_true_enc) -> dict:
    """Six evaluation metrics (macro-averaged for the 7-class problem)."""
    y_pred = pipeline.predict(X)
    metrics = {
        "Accuracy": accuracy_score(y_true_enc, y_pred),
        "Precision": precision_score(
            y_true_enc, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_true_enc, y_pred, average="macro", zero_division=0
        ),
        "F1": f1_score(y_true_enc, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true_enc, y_pred),
    }
    # AUC needs class probabilities and every class to appear in y_true.
    try:
        y_proba = pipeline.predict_proba(X)
        metrics["AUC"] = roc_auc_score(
            y_true_enc, y_proba, multi_class="ovr", average="macro"
        )
    except Exception:
        metrics["AUC"] = np.nan
    # Present in the assignment's column order.
    return {k: metrics[k] for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}


def prepare_inputs(df, pipeline, label_encoder):
    """Validate the uploaded frame and return (X, y_encoded) or raise ValueError."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"The uploaded CSV must contain the target column '{TARGET_COLUMN}'."
        )

    required = list(pipeline.feature_names_in_)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature column(s): {', '.join(missing)}")

    # Reorder to the exact training feature order (sklearn is strict about this).
    X = df[required]

    unknown = set(df[TARGET_COLUMN].unique()) - set(label_encoder.classes_)
    if unknown:
        raise ValueError(
            f"Unknown class label(s) in '{TARGET_COLUMN}': {', '.join(map(str, unknown))}"
        )
    y_enc = label_encoder.transform(df[TARGET_COLUMN])
    return X, y_enc


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("🫘 About this app")
    st.markdown(
        """
        Classifies **dry bean** images into **7 varieties** using 16 geometric
        shape features (area, perimeter, axis lengths, roundness, shape factors …).

        **Five models** are pre-trained and compared:
        Logistic Regression · Decision Tree · KNN · Gaussian Naive Bayes · SVM.

        Upload a **test CSV** (16 features + `Class`). The app loads the saved
        models and scores them — **no retraining** happens here.
        """
    )
    st.caption("BITS WILP · M.Tech (AIML) · Machine Learning · Assignment 2")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("🫘 Dry Bean Variety Classifier")
st.markdown(
    "Compare five machine-learning classifiers on the "
    "[UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) "
    "(7 classes, 16 features)."
)

try:
    models, label_encoder = load_artifacts()
except FileNotFoundError:
    st.error(
        "Saved models not found. Run `python train_models.py` first to create "
        "the files in `model/`."
    )
    st.stop()

st.subheader("1 · Upload your test data")
st.markdown(
    "Upload a CSV containing the **16 feature columns and the `Class` column** "
    "(e.g. the provided `test_data.csv`)."
)
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is None:
    st.info("⬆️ Upload `test_data.csv` to see the model comparison and metrics.")
    st.stop()

# ---- Read + validate ------------------------------------------------------- #
try:
    data = pd.read_csv(uploaded)
except Exception as exc:  # pragma: no cover - defensive
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

# Any one pipeline tells us the required feature set.
any_pipeline = next(iter(models.values()))
try:
    X, y_enc = prepare_inputs(data, any_pipeline, label_encoder)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.success(f"Loaded **{len(data):,} rows** with all required columns.")
with st.expander("Preview uploaded data"):
    st.dataframe(data.head(10), use_container_width=True)

# ---- 2 · Compare ALL models ----------------------------------------------- #
st.subheader("2 · All models compared on your test data")

comparison = {name: compute_metrics(pipe, X, y_enc) for name, pipe in models.items()}
comparison_df = pd.DataFrame(comparison).T
comparison_df.index.name = "Model"

st.dataframe(
    comparison_df.style
    .format("{:.4f}")
    .highlight_max(axis=0, color="#b6e3b6")   # best value per metric
    .highlight_min(axis=0, color="#f6c9c9"),  # worst value per metric
    use_container_width=True,
)
winner = comparison_df["F1"].idxmax()
st.markdown(f"🏆 **Best model on this data (macro-F1): `{winner}`**")

# ---- 3 · Inspect one model ------------------------------------------------- #
st.subheader("3 · Inspect a single model")
choice = st.selectbox("Select a model", list(MODEL_FILES.keys()))
pipeline = models[choice]

row = comparison_df.loc[choice]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{row['Accuracy']:.4f}")
c2.metric("AUC", "—" if pd.isna(row["AUC"]) else f"{row['AUC']:.4f}")
c3.metric("Precision", f"{row['Precision']:.4f}")
c4.metric("Recall", f"{row['Recall']:.4f}")
c5.metric("F1", f"{row['F1']:.4f}")
c6.metric("MCC", f"{row['MCC']:.4f}")

y_pred = pipeline.predict(X)
class_names = list(label_encoder.classes_)

left, right = st.columns([1, 1])

with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_enc, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGnBu",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{choice}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig, use_container_width=True)

with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y_enc, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

st.caption(
    "Metrics are computed live from the uploaded CSV using the saved model — "
    "no model is retrained in this app."
)
