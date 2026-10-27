"""
train_models.py
================
Dry Bean multi-class classification — trains five classifiers, evaluates them
with six metrics each, and saves everything the Streamlit app needs.

Pipeline design (why it looks the way it does)
-----------------------------------------------
Every model is wrapped in a scikit-learn ``Pipeline`` of
``StandardScaler -> estimator``. The scaler is *fit on the training split only*
and travels *inside* the saved model file. That means the Streamlit app can feed
RAW feature values straight into ``pipeline.predict(...)`` and the scaling happens
automatically and identically to training — no chance of a "forgot to scale"
inference bug.

Run:
    python3 train_models.py --data /path/to/Dry_Bean_Dataset.xlsx

If --data is omitted it looks for a local ./Dry_Bean_Dataset.xlsx .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
TARGET_COLUMN = "Class"
RANDOM_STATE = 2026          # fixed seed -> reproducible split & models
TEST_FRACTION = 0.20         # 80 / 20 stratified split

# Human-friendly name  ->  (filename stem, estimator factory)
# Each estimator is paired with a StandardScaler inside a Pipeline below.
MODEL_SPECS = {
    "Logistic Regression": (
        "logistic_regression",
        lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    ),
    "Decision Tree": (
        "decision_tree",
        lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
    ),
    "KNN": (
        "knn",
        lambda: KNeighborsClassifier(n_neighbors=7),
    ),
    "Naive Bayes": (
        "naive_bayes",
        lambda: GaussianNB(),          # Gaussian -> features are continuous
    ),
    "Support Vector Machine": (
        "svm",
        lambda: SVC(probability=True, random_state=RANDOM_STATE),  # proba -> AUC
    ),
}


def build_pipeline(estimator_factory):
    """Scaler + model as one saveable object that accepts RAW features."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator_factory()),
        ]
    )


def evaluate(pipeline, X_test, y_test_enc) -> dict:
    """Compute the six required metrics for a fitted pipeline.

    For a 7-class problem AUC / Precision / Recall / F1 are *macro-averaged*
    (every bean variety counts equally, regardless of how many samples it has).
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)   # shape (n_samples, 7)

    return {
        "Accuracy": accuracy_score(y_test_enc, y_pred),
        "AUC": roc_auc_score(
            y_test_enc, y_proba, multi_class="ovr", average="macro"
        ),
        "Precision": precision_score(
            y_test_enc, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_test_enc, y_pred, average="macro", zero_division=0
        ),
        "F1": f1_score(y_test_enc, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test_enc, y_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dry Bean classifiers.")
    parser.add_argument(
        "--data",
        default=str(HERE / "Dry_Bean_Dataset.xlsx"),
        help="Path to the Dry Bean .xlsx (or .csv) file.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    print(f"Reading dataset: {data_path}")
    if data_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    print(f"  shape = {df.shape}  (rows x columns)")
    feature_columns = [c for c in df.columns if c != TARGET_COLUMN]
    print(f"  {len(feature_columns)} feature columns, target = '{TARGET_COLUMN}'")
    assert len(feature_columns) >= 12, "Assignment needs >= 12 features!"
    assert len(df) >= 500, "Assignment needs >= 500 instances!"

    X = df[feature_columns]
    y_text = df[TARGET_COLUMN]

    # ----- Stratified 80/20 split (keeps class proportions in both halves) ---
    X_train, X_test, y_train_text, y_test_text = train_test_split(
        X,
        y_text,
        test_size=TEST_FRACTION,
        stratify=y_text,
        random_state=RANDOM_STATE,
    )
    print(f"  train = {len(X_train)} rows, test = {len(X_test)} rows")

    # ----- Encode the 7 class names to integers 0..6 -------------------------
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train_text)
    y_test_enc = label_encoder.transform(y_test_text)
    print(f"  classes: {list(label_encoder.classes_)}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Train, evaluate, and save every model -----------------------------
    results = {}
    for display_name, (stem, factory) in MODEL_SPECS.items():
        print(f"\nTraining: {display_name} ...")
        pipeline = build_pipeline(factory)
        pipeline.fit(X_train, y_train_enc)

        metrics = evaluate(pipeline, X_test, y_test_enc)
        results[display_name] = metrics
        print(
            "  "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

        out_file = MODEL_DIR / f"{stem}.joblib"
        joblib.dump(pipeline, out_file)
        print(f"  saved -> {out_file.relative_to(HERE)}")

    # ----- Persist the label encoder (app needs it to decode predictions) ----
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")
    print(f"\nsaved -> model/label_encoder.joblib")

    # ----- Export the held-out TEST split as the app's upload file -----------
    # Original (unscaled) features + the human-readable Class column.
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test_text.values
    test_csv = HERE / "test_data.csv"
    test_df.to_csv(test_csv, index=False)
    print(f"saved -> test_data.csv  ({len(test_df)} rows, for Streamlit upload)")

    # ----- Save the comparison table (README + reference) --------------------
    metrics_df = pd.DataFrame(results).T.round(4)
    metrics_df.index.name = "Model"
    metrics_df.to_csv(HERE / "metrics.csv")
    (HERE / "metrics.json").write_text(json.dumps(results, indent=2))
    print("saved -> metrics.csv, metrics.json")

    print("\n================ COMPARISON TABLE ================")
    print(metrics_df.to_string())
    best = metrics_df["F1"].idxmax()
    print(f"\nBest model by macro-F1: {best}")


if __name__ == "__main__":
    main()
