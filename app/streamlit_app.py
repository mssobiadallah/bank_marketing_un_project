"""
streamlit_app.py — Main entry point for the Bank Marketing ML Streamlit app.

7-page multi-page application:
  Overview (this page)
  1. EDA Dashboard
  2. Hypothesis Testing
  3. Model Performance
  4. Predict New Client
  5. Batch Prediction
  6. Business Recommendations
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when launched from the app/ directory
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import (
    MODELS_DIR,
    OPTIMAL_THRESHOLD,
    PREPROCESSING_PIPELINE_B_FILE,
    REALISTIC_MODEL_FILE,
)

# ---------------------------------------------------------------------------
# Page config (must be the FIRST Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing ML",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached model/pipeline loader — shared across all pages
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def _load_model():
    """Load the Realistic Business Model and preprocessing pipeline."""
    from src.inference import load_model_and_pipeline
    return load_model_and_pipeline(
        MODELS_DIR / REALISTIC_MODEL_FILE,
        MODELS_DIR / PREPROCESSING_PIPELINE_B_FILE,
    )


# Try to load model; show hard stop if missing
try:
    model, pipeline = _load_model()
except FileNotFoundError as exc:
    st.error(
        f"**Model files not found.**\n\n{exc}\n\n"
        "Please run `make train` (or `python scripts/train.py`) to generate model artifacts."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------------
st.title("🏦 Predicting Bank Term Deposit Subscription")
st.markdown(
    """
A Portuguese bank runs telephone marketing campaigns to sell term deposit products.
This app uses classical machine learning to **predict which clients are most likely to
subscribe**, enabling the bank to prioritise outreach and improve campaign efficiency.
"""
)

# Duration warning — prominent and persistent
st.warning(
    "⚠️  **Duration Excluded**: The `duration` feature (last call length in seconds) is "
    "**never used** in any prediction in this app. It is only known *after* the call is made, "
    "so using it would constitute data leakage and make the model useless in practice."
)

st.divider()

# Dataset overview
st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", "41,188")
with col2:
    st.metric("Features Used", "19 (no duration)")
with col3:
    st.metric("Positive Rate", "~11.3%")
with col4:
    st.metric("Train / Test", "80% / 20%")

st.divider()

# Model performance summary
st.subheader("🎯 Champion Model — Advanced Pipeline (Notebook 09)")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Model", "FLAML-lgbm")
with col2:
    st.metric("Accuracy", "90.23%")
with col3:
    st.metric("Subscribe Recall", "56% ↑", delta="vs 24% default", delta_color="normal")
with col4:
    st.metric("ROC-AUC", "0.813")
with col5:
    st.metric("Decision Threshold", f"{OPTIMAL_THRESHOLD:.2f}", help="Tuned from 0.50 → best Subscribe F1")

st.info(
    f"ℹ️ The default 0.50 threshold gave 90% accuracy but **only 24% recall** on subscribers. "
    f"Lowering to **{OPTIMAL_THRESHOLD}** raises recall to **56%** with acceptable accuracy (89%), "
    f"which is far more useful for a marketing campaign."
)

st.divider()

# Architecture description
st.subheader("🏗️ Architecture")
st.markdown(
    """
```
Raw Data  →  Feature Engineering (9 new features)  →  Preprocessing Pipeline
                                                              ↓
                                              Baseline Models (8 classifiers)
                                                              ↓
                                   ┌─────────────────────────────────────┐
                                   │  Advanced Pipeline (Notebook 09)    │
                                   │  FLAML AutoML → Imbalance Handling  │
                                   │  → Feature Selection → GridSearch   │
                                   │  → Threshold Tuning                 │
                                   └─────────────────────────────────────┘
                                                              ↓
                              Champion: FLAML-lgbm  (thr=0.27)
                              Accuracy=89% | Recall=56% | F1=0.53
                                                              ↓
                                         7-page Streamlit Application
```

**Two model tracks:**
- 📊 **Benchmark Model** (Feature Set A) — includes `duration` — upper-bound reference only
- ✅ **Realistic Business Model** (Feature Set B) — **no `duration`** — used in all app predictions
- 🏆 **Advanced FLAML Champion** — threshold-tuned to maximise subscriber recall
"""
)

st.divider()
st.subheader("🗺️ Navigation")
st.markdown(
    """
Use the **sidebar** to navigate between pages:

| Page | Description |
|------|-------------|
| 📈 EDA Dashboard | Full exploratory data analysis |
| 🧪 Hypothesis Testing | Statistical test results (H1–H7) |
| 🏆 Model Performance | Comparison, ROC/PR curves, threshold tuning |
| 🔮 Predict New Client | Single-customer prediction form |
| 📁 Batch Prediction | CSV upload → ranked predictions download |
| 💡 Business Recommendations | Insights, segments, campaign strategy |
"""
)
