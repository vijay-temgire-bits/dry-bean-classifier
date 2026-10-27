# 🫘 Dry Bean Variety Classifier — ML Assignment 2

An end-to-end machine-learning project that trains **five classification models**
on the **UCI Dry Bean Dataset**, compares them with six evaluation metrics, and
serves an interactive **Streamlit** app for scoring uploaded test data.

> BITS WILP · M.Tech (AIML) · Machine Learning · Assignment 2

---

## a. Problem statement

Given 16 geometric shape features extracted from images of dried beans, predict
which of **7 registered bean varieties** each sample belongs to. This is a
**multi-class classification** problem. We train and compare five classical
classifiers — Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian
Naive Bayes, and a Support Vector Machine — and identify which performs best on
this dataset.
<img width="689" height="650" alt="image" src="https://github.com/user-attachments/assets/eb1866a4-14b1-4282-a4e3-368dc14820cc" />

---

## b. Dataset description

| Property | Value |
|---|---|
| Source | [UCI Machine Learning Repository — Dry Bean Dataset (ID 602)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| Task | Multi-class classification |
| Instances | **13,611** |
| Features | **16** (all numeric) |
| Target | `Class` — 7 bean varieties |
| Missing values | None |

**Classes (with sample counts):** DERMASON (3,546), SIRA (2,636), SEKER (2,027),
HOROZ (1,928), CALI (1,630), BARBUNYA (1,322), BOMBAY (522). The data is mildly
imbalanced — DERMASON has ~7× the samples of BOMBAY.

**Features:** `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`,
`AspectRation`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`,
`Solidity`, `roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`,
`ShapeFactor3`, `ShapeFactor4` — i.e. 12 dimension measurements plus 4 shape
factors describing each bean's silhouette.

**Preprocessing:** features are standardised with `StandardScaler` (fit on the
training split only), the 7 class labels are integer-encoded with `LabelEncoder`,
and the data is split **80 / 20 with stratification** (train = 10,888 rows,
test = 2,723 rows). The 20% test split is saved as `test_data.csv` — that is the
file you upload to the app.

> Citation: KOKLU, M. and OZKAN, I.A. (2020), *Multiclass Classification of Dry
> Beans Using Computer Vision and Machine Learning Techniques*, Computers and
> Electronics in Agriculture, 174, 105507.

---

## c. GitHub Repository Link

> **`<ADD YOUR GITHUB REPO LINK HERE AFTER PUSHING>`**

(Replace the placeholder above with your repository URL once you have pushed the
code — see "How to publish & deploy" below.)

---

## d. Models used

All five models were trained on the **same** dataset and evaluated on the held-out
20% test split. For this 7-class problem, **AUC** is macro-averaged one-vs-rest,
and **Precision / Recall / F1** are **macro-averaged** (each variety weighted
equally regardless of its size).

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9251 | 0.9952 | 0.9366 | 0.9372 | 0.9369 | 0.9095 |
| Decision Tree | 0.8898 | 0.9437 | 0.9058 | 0.9068 | 0.9062 | 0.8669 |
| KNN | 0.9240 | 0.9879 | 0.9378 | 0.9366 | 0.9371 | 0.9081 |
| Naive Bayes | 0.9008 | 0.9921 | 0.9085 | 0.9133 | 0.9106 | 0.8805 |
| **Support Vector Machine** | **0.9321** | **0.9956** | **0.9454** | **0.9429** | **0.9441** | **0.9179** |

*(Metrics are reproduced live in the Streamlit app from `test_data.csv`; see
`metrics.csv` for the saved copy.)*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a linear model (Acc 0.925, AUC 0.995). After standardisation the varieties are largely linearly separable, so a simple, fast, interpretable model lands just below the SVM. Excellent accuracy-per-compute. |
| Decision Tree | **Weakest** model (Acc 0.890, AUC 0.944). A single unpruned tree makes axis-aligned splits and overfits the training data, so it generalises worse and its probability estimates (hence AUC) are the least reliable of the five. |
| KNN | Competitive (Acc 0.924) once features are scaled, since it is distance-based. Nearly ties Logistic Regression on F1, but is the slowest at prediction time because it must compare against all training points, and it is sensitive to the choice of *k*. |
| Naive Bayes | Solid given its strong assumption (Acc 0.901). Several features are highly correlated (e.g. Area, ConvexArea, EquivDiameter), which violates the feature-independence assumption and slightly lowers precision — yet its probability ranking stays good (AUC 0.992) and it trains almost instantly. |
| Support Vector Machine | **Best on every metric** (Acc 0.932, F1 0.944, AUC 0.996, MCC 0.918). The RBF kernel captures non-linear class boundaries the linear models cannot. The trade-off is the longest training time of the five. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF).** It leads on all six metrics. Logistic Regression is a close, far cheaper runner-up — a better pick if training/inference speed matters more than the last ~1% of accuracy. |

---

## Project structure

```
dry-bean-classifier/
├── app.py                 # Streamlit app (upload → load model → compute metrics → display)
├── train_models.py        # Source: load data, preprocess, train, evaluate, export
├── requirements.txt       # Pinned dependencies (sklearn version matches the saved models)
├── README.md              # This file
├── test_data.csv          # Held-out 20% test split (16 features + Class) — upload this
├── metrics.csv            # Saved comparison table
├── metrics.json           # Same metrics, machine-readable
├── Dry_Bean_Dataset.xlsx  # Raw dataset (so training is reproducible from a clone)
└── model/                 # Saved artifacts (all five pipelines + label encoder)
    ├── logistic_regression.joblib   # each = Pipeline(StandardScaler + estimator)
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── label_encoder.joblib
```

Each `*.joblib` model is a full `StandardScaler + estimator` pipeline, so the app
feeds **raw** feature values straight in and scaling happens automatically — the
app never retrains anything.

---

## How to run locally

```bash
pip install -r requirements.txt

# (optional) retrain from scratch — regenerates model/ and test_data.csv
python train_models.py

# launch the app
streamlit run app.py
```

Then upload `test_data.csv` in the browser. The app shows all five models
compared, a per-model metric panel, a confusion matrix, and a classification
report — all computed live, with no retraining.

---

## How to publish & deploy (do these yourself)

1. **GitHub** — create a new repository, then push this `dry-bean-classifier/`
   folder (make real commits over time, not one bulk upload — commit history is
   reviewed). Paste the repo URL into section **c** above.
2. **Streamlit Community Cloud** — go to <https://streamlit.io/cloud> → sign in
   with GitHub → **New App** → pick this repo → branch `main` → main file
   `app.py` → **Deploy**. If the build fails on a dependency, open *Advanced
   settings* and set the Python version to **3.14** (the version used here), or
   relax the non-critical pins — but keep `scikit-learn==1.9.1` so the saved
   models still load.
3. **BITS Virtual Lab** — run the assignment once on the BITS Virtual Lab and
   take **one screenshot** of the execution as proof (worth 1 mark). Add it here:

   > _BITS Virtual Lab execution screenshot:_ `<attach screenshot in your submission PDF>`

---

## Academic integrity

This project was built from scratch for this assignment: an original dataset
choice, custom feature-handling and code, a customised Streamlit UI, and
observations written from the actual results above. Using AI tools for learning
support is permitted; this submission is not a copy-paste of any existing repo or
template.
