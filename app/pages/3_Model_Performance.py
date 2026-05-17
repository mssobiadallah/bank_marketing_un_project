"""3_Model_Performance.py — Model comparison, curves, threshold tuning."""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.config import REPORTS_DIR, MODELS_DIR, PREPROCESSING_PIPELINE_B_FILE, REALISTIC_MODEL_FILE, RAW_DIR, MAIN_DATASET, CSV_SEP, TARGET_COL, TEST_SIZE, RANDOM_SEED, OPTIMAL_THRESHOLD
from src.features import encode_target, add_features
from src.preprocessing import split_data

st.set_page_config(page_title="Model Performance", page_icon="🏆", layout="wide")
st.title("🏆 Model Performance")

# ------------------------------------------------------------------
# Load metrics CSV
# ------------------------------------------------------------------
metrics_path = REPORTS_DIR / "model_metrics.csv"

@st.cache_data(show_spinner="Loading metrics…")
def _load_metrics():
    return pd.read_csv(metrics_path)

try:
    metrics_df = _load_metrics()
except FileNotFoundError:
    st.error("model_metrics.csv not found. Run `make train` first.")
    st.stop()

# ------------------------------------------------------------------
# Model comparison table
# ------------------------------------------------------------------
st.subheader("📊 Model Comparison — All Baselines")
feature_sets = metrics_df["feature_set"].unique().tolist() if "feature_set" in metrics_df.columns else ["set_b"]
selected_fs = st.radio("Feature Set", feature_sets, horizontal=True)
subset = metrics_df[metrics_df["feature_set"] == selected_fs].copy() if "feature_set" in metrics_df.columns else metrics_df.copy()

# Highlight best PR-AUC
pr_col = "pr_auc" if "pr_auc" in subset.columns else "average_precision"
if pr_col in subset.columns:
    subset_sorted = subset.sort_values(pr_col, ascending=False)
    st.dataframe(
        subset_sorted.style.highlight_max(subset=pr_col, color="#d4edda"),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.dataframe(subset, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# ROC & PR curves for best model
# ------------------------------------------------------------------
st.subheader("📉 ROC & PR Curves (Realistic Business Model)")

@st.cache_resource(show_spinner="Loading model for curves…")
def _load_model_and_data():
    from src.inference import load_model_and_pipeline
    from src.config import FEATURE_SET_B_COLS
    model, pipeline = load_model_and_pipeline(
        MODELS_DIR / REALISTIC_MODEL_FILE,
        MODELS_DIR / PREPROCESSING_PIPELINE_B_FILE,
    )
    df_raw = pd.read_csv(RAW_DIR / MAIN_DATASET, sep=CSV_SEP)
    df_raw = encode_target(df_raw, TARGET_COL)
    df_raw = add_features(df_raw)
    _, X_test, _, y_test = split_data(df_raw, TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    # Select only Feature Set B columns (excludes duration)
    X_test = X_test[FEATURE_SET_B_COLS]
    return model, pipeline, X_test, y_test

try:
    model, pipeline, X_test, y_test = _load_model_and_data()
    X_test_t = pipeline.transform(X_test)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec, prec)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rec, prec, color="#ff7f0e", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
        baseline = y_test.mean()
        ax.axhline(baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # Threshold tuning slider
    st.subheader("🎚️ Threshold Tuning")
    st.markdown(
        f"**Optimal threshold = `{OPTIMAL_THRESHOLD}`** (tuned in Notebook 09 for best Subscribe F1). "
        "Move the slider to see the precision / recall / F1 tradeoff interactively."
    )
    threshold = st.slider("Classification threshold", 0.05, 0.90, OPTIMAL_THRESHOLD, 0.01)
    y_pred = (y_proba >= threshold).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Threshold", f"{threshold:.2f}")
    c2.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    c3.metric("Precision (Subscribe)", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
    c4.metric("Recall (Subscribe)", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
    c5.metric("F1 (Subscribe)", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")

except FileNotFoundError as e:
    st.warning(f"Could not load model for curves: {e}")

# ------------------------------------------------------------------
# Advanced Pipeline Results (Notebook 09)
# ------------------------------------------------------------------
st.divider()
st.subheader("🚀 Advanced Pipeline — Notebook 09 Results")
st.markdown(
    """
    After the baseline comparison, an advanced pipeline was developed using:
    **FLAML AutoML → Class Imbalance Handling → Feature Selection → GridSearch → Threshold Tuning**
    """
)

# Key metrics comparison
col1, col2, col3, col4 = st.columns(4)
col1.metric("Champion Model", "FLAML-lgbm")
col2.metric("Accuracy (thr=0.27)", "88.7%", delta="vs 90.2% at thr=0.5")
col3.metric("Subscribe Recall", "56%", delta="+32pp vs default", delta_color="normal")
col4.metric("Subscribe F1", "0.53", delta="+0.17 vs default", delta_color="normal")

# Show saved charts from the pipeline
col_a, col_b = st.columns(2)
with col_a:
    adv_chart = REPORTS_DIR / "advanced_pipeline_comparison.png"
    if adv_chart.exists():
        from PIL import Image
        st.image(str(adv_chart), caption="PR & ROC Curves — All Advanced Pipeline Models", use_container_width=True)
    else:
        st.info("Run Notebook 09 to generate the advanced pipeline comparison chart.")

with col_b:
    impr_chart = REPORTS_DIR / "improvement_comparison.png"
    if impr_chart.exists():
        from PIL import Image
        st.image(str(impr_chart), caption="Improvement Analysis — Recall & Precision across Strategies", use_container_width=True)
    else:
        st.info("Run Notebook 09 to generate the improvement comparison chart.")

# Show diagnosis and threshold tuning charts
col_c, col_d = st.columns(2)
with col_c:
    diag_chart = REPORTS_DIR / "diagnosis_imbalance.png"
    if diag_chart.exists():
        st.image(str(diag_chart), caption="Root Cause: Class Imbalance + Default Threshold = Low Recall", use_container_width=True)
with col_d:
    thr_chart = REPORTS_DIR / "threshold_tuning.png"
    if thr_chart.exists():
        st.image(str(thr_chart), caption="Threshold Tuning — Before vs After", use_container_width=True)

# Confusion matrix comparison
cm_chart = REPORTS_DIR / "confusion_matrix_comparison.png"
if cm_chart.exists():
    st.image(str(cm_chart), caption="Confusion Matrix Comparison — Threshold-Tuned Models", use_container_width=True)

# Strategy summary table
st.markdown("#### Strategy Summary")
strategy_data = {
    "Strategy": [
        "FLAML-lgbm (original, thr=0.50)",
        f"FLAML-lgbm (threshold={OPTIMAL_THRESHOLD})",
        "FLAML-F1 metric (threshold=0.22)",
        "Balanced LGBM (threshold=0.63)",
    ],
    "Accuracy": ["90.2%", "88.7%", "87.3%", "88.0%"],
    "Subscribe Precision": ["0.69", "0.50", "0.45", "0.48"],
    "Subscribe Recall": ["0.24", "0.56", "0.53", "0.60"],
    "Subscribe F1": ["0.36", "0.53", "0.49", "0.53"],
    "Macro F1": ["0.65", "0.71", "0.71", "0.73"],
}
import pandas as pd
strat_df = pd.DataFrame(strategy_data)
st.dataframe(
    strat_df.style.apply(
        lambda row: ["background-color: #c8e6c9; font-weight: bold"] * len(row)
        if row.name == 1 else [""] * len(row),
        axis=1,
    ),
    use_container_width=True,
    hide_index=True,
)
