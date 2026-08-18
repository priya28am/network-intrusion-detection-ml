"""
Streamlit app: Network Intrusion Detection (UNSW-NB15)

Lets a user upload a test CSV of network connection records, pick one of
5 pretrained classifiers (or compare all of them), and see how well the
model identifies each traffic type (Normal + 9 attack categories) --
metrics, confusion matrix, and classification report.

Models are trained OFFLINE by model/train_models.py and loaded here as
saved pipelines, so the app itself never retrains anything -- this keeps
it fast and light enough for Streamlit Community Cloud's free tier.
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model.preprocessing import ATTACK_CATEGORIES, clean_data, split_features_target

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(ROOT, "model", "saved_models")
SAMPLE_TEST_DATA_PATH = os.path.join(ROOT, "test_data.csv")

# Display order for the model dropdown / comparison table.
MODEL_ORDER = ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"]

# Fixed categorical colors, one per model, so the same model is always the
# same color across every chart in the app (identity, not rank).
MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",  # blue
    "Decision Tree": "#eb6834",        # orange
    "kNN": "#1baf7a",                  # aqua
    "Naive Bayes": "#eda100",          # yellow
    "Random Forest": "#e87ba4",        # magenta
}


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


@st.cache_resource(show_spinner="Loading model...")
def load_pipeline(model_name: str):
    path = os.path.join(SAVED_MODELS_DIR, f"{slugify(model_name)}.joblib")
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_sample_test_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_TEST_DATA_PATH)


def available_models():
    return [
        name for name in MODEL_ORDER
        if os.path.exists(os.path.join(SAVED_MODELS_DIR, f"{slugify(name)}.joblib"))
    ]


def compute_metrics(y_true, y_pred, y_proba, class_order) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=class_order
        ),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    # Explicit light styling regardless of the viewer's Streamlit theme, so
    # the chart is never rendered as a bright white box on a dark page (or
    # vice versa) with mismatched text color.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=True, annot_kws={"size": 9}, linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Predicted attack_cat", fontsize=10)
    ax.set_ylabel("Actual attack_cat", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=12, pad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def plot_metric_bar(comparison_df: pd.DataFrame, metric: str):
    """Single-metric bar chart across all models, with value labels on each bar."""
    models = comparison_df.index.tolist()
    values = comparison_df[metric].values
    colors = [MODEL_COLORS.get(m, "#2a78d6") for m in models]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    bars = ax.bar(models, values, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=11, color="#0b0b0b")

    ax.set_ylim(0, 1.08)
    ax.set_ylabel(metric, fontsize=11, color="#0b0b0b")
    ax.set_title(f"{metric} by model", fontsize=15, color="#0b0b0b", pad=14)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="x", colors="#0b0b0b", labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.tick_params(axis="y", colors="#898781", labelsize=9)
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def class_distribution_chart(y: pd.Series):
    counts = y.value_counts().reindex(
        [c for c in ATTACK_CATEGORIES if c in y.unique()]
    )
    return counts


# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Network Intrusion Detection - UNSW-NB15",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Network Intrusion Detection — UNSW-NB15")
st.markdown(
    """
This app demonstrates 5 classification models trained to identify network
traffic as **Normal** or one of **9 attack categories**
(Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor,
Shellcode, Worms) from the [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
network flow dataset.

Upload a test CSV (same schema as `test_data.csv` in this repo), choose a
model, and see its predictions, evaluation metrics, and confusion matrix.
"""
)
st.divider()

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

st.sidebar.subheader("1. Data source")
uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"],
    help=(
        "CSV must contain the UNSW-NB15 feature columns. "
        "Include an `attack_cat` column to see evaluation metrics; "
        "omit it to get predictions only."
    ),
)
use_sample = uploaded_file is None
if use_sample:
    st.sidebar.caption("No file uploaded — using bundled sample `test_data.csv`.")

st.sidebar.subheader("2. Model")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    options=["Compare all 5 models"] + available_models(),
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Models are pretrained offline (`model/train_models.py`) and loaded "
    "here as saved pipelines — this app only performs inference."
)

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
if use_sample:
    raw_df = load_sample_test_data()
    st.success(f"Using bundled sample test data — {len(raw_df):,} rows.")
else:
    raw_df = pd.read_csv(uploaded_file)
    st.success(f"Loaded uploaded file '{uploaded_file.name}' — {len(raw_df):,} rows.")

with st.expander("🔍 Preview data", expanded=False):
    st.dataframe(raw_df.head(20), width="stretch")

try:
    df = clean_data(raw_df)
    X, y_true = split_features_target(df)
except ValueError as e:
    st.error(f"Uploaded CSV is not compatible with the model: {e}")
    st.stop()

has_labels = y_true is not None
if not has_labels:
    st.warning(
        "No `attack_cat` column found — showing predictions only "
        "(evaluation metrics need true labels to compute)."
    )

present_labels = sorted(set(y_true.unique()) if has_labels else []) or ATTACK_CATEGORIES
label_order = [c for c in ATTACK_CATEGORIES if c in present_labels] or ATTACK_CATEGORIES

# ----------------------------------------------------------------------
# Dataset overview — quick context before diving into a model
# ----------------------------------------------------------------------
st.subheader("📁 Dataset overview")
overview_cols = st.columns(3)
overview_cols[0].metric("Rows loaded", f"{len(X):,}")
overview_cols[1].metric("Features", f"{X.shape[1]}")
overview_cols[2].metric("Classes present", f"{len(present_labels)}" if has_labels else "n/a")

if has_labels:
    with st.expander("Class distribution in this data", expanded=False):
        st.bar_chart(class_distribution_chart(y_true))

st.divider()

# ----------------------------------------------------------------------
# Single-model view
# ----------------------------------------------------------------------
if model_choice != "Compare all 5 models":
    st.header(f"📊 Results — {model_choice}")

    pipeline = load_pipeline(model_choice)
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    class_order = pipeline.named_steps["clf"].classes_

    if has_labels:
        metrics = compute_metrics(y_true, y_pred, y_proba, class_order)
        metric_items = list(metrics.items())
        row1, row2 = st.columns(3), st.columns(3)
        for col, (label, value) in zip(row1 + row2, metric_items):
            col.metric(label, f"{value:.4f}")

        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(y_true, y_pred, label_order)
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(
            y_true, y_pred, labels=label_order, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(report_df, width="stretch")
    else:
        st.subheader("Predicted class distribution")
        st.bar_chart(pd.Series(y_pred).value_counts())

    st.subheader("Sample predictions")
    preview = X.copy()
    preview["predicted_attack_cat"] = y_pred
    preview["confidence"] = y_proba.max(axis=1).round(4)
    if has_labels:
        preview.insert(0, "actual_attack_cat", y_true.values)
    st.dataframe(preview.head(50), width="stretch")
    st.download_button(
        "⬇️ Download all predictions (CSV)",
        data=preview.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{slugify(model_choice)}.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------
# Compare-all-models view
# ----------------------------------------------------------------------
else:
    st.header("📊 Comparing All 5 Models")

    if not has_labels:
        st.error("Model comparison requires true labels — upload a CSV with an `attack_cat` column.")
        st.stop()

    rows = []
    predictions = {}
    for name in available_models():
        pipeline = load_pipeline(name)
        y_pred = pipeline.predict(X)
        y_proba = pipeline.predict_proba(X)
        class_order = pipeline.named_steps["clf"].classes_
        metrics = compute_metrics(y_true, y_pred, y_proba, class_order)
        metrics["ML Model Name"] = name
        rows.append(metrics)
        predictions[name] = y_pred

    comparison_df = pd.DataFrame(rows).set_index("ML Model Name")
    comparison_df = comparison_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]].round(4)

    st.subheader("Comparison table")
    st.caption("🟩 Highlighted = best model for that metric (higher is better).")
    styled = comparison_df.style.highlight_max(axis=0, color="#8fd19e")
    st.dataframe(styled, width="stretch")
    st.download_button(
        "⬇️ Download comparison table (CSV)",
        data=comparison_df.to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    # MCC is the single most reliable metric here since it accounts for all
    # confusion-matrix quadrants across every class and isn't inflated by
    # the large Normal/Generic classes the way raw Accuracy can be.
    best_model = comparison_df["MCC"].idxmax()
    st.success(f"🏆 **Best overall model (by MCC): {best_model}** — MCC = {comparison_df.loc[best_model, 'MCC']:.4f}")

    st.subheader("Metric comparison chart")
    metric_to_plot = st.selectbox("Metric to visualize", comparison_df.columns.tolist())
    st.pyplot(plot_metric_bar(comparison_df, metric_to_plot))

    st.subheader("Confusion matrix per model")
    tabs = st.tabs(available_models())
    for tab, name in zip(tabs, available_models()):
        with tab:
            fig = plot_confusion_matrix(y_true, predictions[name], label_order)
            st.pyplot(fig)

st.markdown("---")
st.caption(
    "M.Tech (AIML/DSE) — Machine Learning Assignment 2 · UNSW-NB15 Network Intrusion Detection"
)
