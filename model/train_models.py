"""
Trains all 5 required classification models on the UNSW-NB15 network
intrusion detection dataset (10-class multiclass: attack_cat), evaluates
them on a held-out test split, and persists:

  - model/saved_models/*.joblib   -> fitted (preprocessing + classifier) pipelines
  - test_data.csv                 -> held-out test split (raw schema), used by
                                      the Streamlit app for demos
  - results/metrics.csv           -> comparison table (Accuracy/AUC/Precision/
                                      Recall/F1/MCC) for all 5 models
  - results/metrics_table.md      -> the same table, pre-formatted as a
                                      Markdown table for pasting into README.md

The full UNSW-NB15 training partition has 175,341 rows. Training all 5
models (especially kNN, whose prediction cost scales with training-set
size) on the full set is unnecessarily slow for a demo app, so we take a
class-stratified subsample first: every class is capped at
MAX_PER_CLASS rows, and classes smaller than the cap (e.g. Worms, only
130 rows total) are kept in full. This preserves the dataset's real
class imbalance -- which is the whole point of this problem -- while
keeping training/inference fast.

Run from the project root:
    python model/train_models.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.preprocessing import (  # noqa: E402
    build_preprocessor,
    clean_data,
    load_raw_data,
    split_features_target,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(ROOT, "data", "UNSW_NB15_full.csv")
SAVED_MODELS_DIR = os.path.join(ROOT, "model", "saved_models")
RESULTS_DIR = os.path.join(ROOT, "results")
TEST_DATA_PATH = os.path.join(ROOT, "test_data.csv")

RANDOM_STATE = 42
MAX_PER_CLASS = 6000       # cap for large classes when subsampling
TEST_SIZE = 0.15

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000, random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def stratified_subsample(df: pd.DataFrame, target_col: str, cap: int) -> pd.DataFrame:
    parts = []
    for cls, group in df.groupby(target_col):
        if len(group) > cap:
            group = group.sample(n=cap, random_state=RANDOM_STATE)
        parts.append(group)
    out = pd.concat(parts, axis=0)
    return out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def main():
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading raw data from {RAW_DATA_PATH} ...")
    raw_df = load_raw_data(RAW_DATA_PATH)
    df = clean_data(raw_df)
    print(f"Full cleaned dataset: {len(df)} rows")
    print("Full class distribution:")
    print(df["attack_cat"].value_counts())

    df = stratified_subsample(df, "attack_cat", MAX_PER_CLASS)
    print(f"\nSubsampled dataset for training/eval: {len(df)} rows (cap={MAX_PER_CLASS}/class)")
    print(df["attack_cat"].value_counts())

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["attack_cat"],
    )

    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"\nSaved held-out test data ({len(test_df)} rows) -> {TEST_DATA_PATH}")

    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    rows = []
    for name, clf in MODELS.items():
        print(f"\nTraining {name} ...")
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        class_order = pipeline.named_steps["clf"].classes_

        # Multiclass metrics: macro-averaged so every attack category
        # (including rare ones like Worms) counts equally, rather than
        # being swamped by the large Normal/Generic/Exploits classes.
        metrics = {
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="macro", labels=class_order
            ),
            "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }
        rows.append(metrics)
        print(
            f"  Accuracy={metrics['Accuracy']:.4f}  AUC={metrics['AUC']:.4f}  "
            f"Precision={metrics['Precision']:.4f}  Recall={metrics['Recall']:.4f}  "
            f"F1={metrics['F1']:.4f}  MCC={metrics['MCC']:.4f}"
        )

        model_path = os.path.join(SAVED_MODELS_DIR, f"{slugify(name)}.joblib")
        joblib.dump(pipeline, model_path, compress=3)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Saved pipeline -> {model_path} ({size_mb:.1f} MB)")

    metrics_df = pd.DataFrame(rows)
    metrics_csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved metrics table -> {metrics_csv_path}")

    md_lines = ["| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |",
                "|---|---|---|---|---|---|---|"]
    for r in rows:
        md_lines.append(
            f"| {r['ML Model Name']} | {r['Accuracy']:.4f} | {r['AUC']:.4f} | "
            f"{r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['MCC']:.4f} |"
        )
    md_path = os.path.join(RESULTS_DIR, "metrics_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved Markdown comparison table -> {md_path}")

    print("\nDone. Comparison table:\n")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
