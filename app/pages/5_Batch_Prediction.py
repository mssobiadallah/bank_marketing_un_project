"""5_Batch_Prediction.py — CSV upload → batch predictions download."""
from __future__ import annotations
import sys
from pathlib import Path
import io

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.config import MODELS_DIR, PREPROCESSING_PIPELINE_B_FILE, REALISTIC_MODEL_FILE

st.set_page_config(page_title="Batch Prediction", page_icon="📁", layout="wide")
st.title("📁 Batch Prediction")

st.markdown(
    """
Upload a CSV file with client records to get subscription probability predictions for all clients.
The results are sorted by probability (highest first) and available for download.
"""
)

st.warning(
    "⚠️  **Duration Not Used**: If your CSV contains a `duration` column, "
    "it will be automatically removed before prediction. "
    "Duration is excluded to prevent data leakage."
)

# Load model
@st.cache_resource(show_spinner="Loading model…")
def _load():
    from src.inference import load_model_and_pipeline
    return load_model_and_pipeline(
        MODELS_DIR / REALISTIC_MODEL_FILE,
        MODELS_DIR / PREPROCESSING_PIPELINE_B_FILE,
    )

try:
    model, pipeline = _load()
except FileNotFoundError as e:
    st.error(f"Model not found: {e}. Run `make train` first.")
    st.stop()

# Required columns info
from src.config import FEATURE_SET_B_COLS
st.subheader("Required Input Columns (19 features)")
st.code(", ".join(FEATURE_SET_B_COLS))

# File uploader
uploaded = st.file_uploader(
    "Upload CSV file (semicolon `;` or comma `,` separator)",
    type=["csv"],
)

if uploaded is not None:
    # Auto-detect separator
    raw_bytes = uploaded.read()
    sample = raw_bytes[:2000].decode("utf-8", errors="ignore")
    sep = ";" if sample.count(";") > sample.count(",") else ","
    uploaded.seek(0)

    try:
        df_input = pd.read_csv(uploaded, sep=sep)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.success(f"Loaded **{len(df_input):,} rows** × {df_input.shape[1]} columns (sep=`{sep}`)")

    # Duration warning
    if "duration" in df_input.columns:
        st.warning("⚠️ `duration` column detected and will be excluded from predictions.")

    st.subheader("Data Preview (first 5 rows)")
    st.dataframe(df_input.head(), use_container_width=True)

    if st.button("🚀 Run Batch Predictions", use_container_width=True):
        try:
            from src.inference import predict_batch, validate_input_schema

            errors = validate_input_schema(df_input, feature_set="set_b")
            if errors:
                st.error("Input validation failed:\n\n" + "\n".join(f"- {e}" for e in errors))
                st.stop()

            results_df = predict_batch(model, pipeline, df_input)

            st.success(f"✅ Predicted {len(results_df):,} clients")

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                high = (results_df["risk_level"] == "High").sum()
                st.metric("High Risk Clients", f"{high:,}")
            with col2:
                med = (results_df["risk_level"] == "Medium").sum()
                st.metric("Medium Risk Clients", f"{med:,}")
            with col3:
                low = (results_df["risk_level"] == "Low").sum()
                st.metric("Low Risk Clients", f"{low:,}")

            st.subheader("Top 20 Predicted Subscribers (by probability)")
            st.dataframe(results_df.head(20), use_container_width=True)

            # Download button
            csv_out = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Full Results CSV",
                data=csv_out,
                file_name="batch_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except ValueError as e:
            st.error(f"Validation error: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)
