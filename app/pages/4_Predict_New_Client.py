"""4_Predict_New_Client.py — Single prediction form."""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import MODELS_DIR, PREPROCESSING_PIPELINE_B_FILE, REALISTIC_MODEL_FILE

st.set_page_config(page_title="Predict New Client", page_icon="🔮", layout="wide")
st.title("🔮 Predict New Client")

st.warning(
    "⚠️  **Duration Not Used**: This form deliberately excludes `duration` "
    "(call length in seconds). Using call duration would be data leakage — "
    "you only know it *after* the call has already happened."
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

st.subheader("Client Information")

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographic**")
        age = st.number_input("Age", min_value=17, max_value=100, value=35)
        job = st.selectbox("Job", [
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"
        ])
        marital = st.selectbox("Marital Status", ["divorced", "married", "single", "unknown"])
        education = st.selectbox("Education", [
            "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
            "professional.course", "university.degree", "unknown"
        ])

    with col2:
        st.markdown("**Financial**")
        default = st.selectbox("Credit Default", ["no", "yes", "unknown"])
        housing = st.selectbox("Housing Loan", ["no", "yes", "unknown"])
        loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"])
        balance_proxy = st.number_input("emp.var.rate (Employment Variation Rate)", value=-1.8, step=0.1, format="%.1f")

    with col3:
        st.markdown("**Campaign**")
        contact = st.selectbox("Contact Type", ["cellular", "telephone"])
        month = st.selectbox("Last Contact Month", [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ])
        day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"])
        campaign = st.number_input("Number of Contacts This Campaign", min_value=1, max_value=50, value=2)
        pdays = st.number_input("Days Since Last Contact (999 = never)", min_value=0, max_value=999, value=999)
        previous = st.number_input("Previous Campaign Contacts", min_value=0, max_value=20, value=0)
        poutcome = st.selectbox("Previous Outcome", ["failure", "nonexistent", "success"])

    st.markdown("**Economic Indicators**")
    ecol1, ecol2, ecol3 = st.columns(3)
    with ecol1:
        cons_price_idx = st.number_input("Consumer Price Index", value=93.2, format="%.3f")
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=-42.0, format="%.1f")
    with ecol2:
        euribor3m = st.number_input("Euribor 3-Month Rate", value=1.0, format="%.3f")
        nr_employed = st.number_input("Number Employed (thousands)", value=5099.1, format="%.1f")
    with ecol3:
        emp_var_rate = balance_proxy  # already captured

    submitted = st.form_submit_button("🔮 Predict Subscription Probability", use_container_width=True)

if submitted:
    input_dict = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
    }

    try:
        from src.inference import predict_single
        result = predict_single(model, pipeline, input_dict)

        st.divider()
        st.subheader("Prediction Result")
        prob = result["subscription_probability"]
        risk = result["risk_level"]

        col1, col2, col3 = st.columns(3)
        with col1:
            if risk == "High":
                st.error(f"🔴 Risk Level: **{risk}**")
            elif risk == "Medium":
                st.warning(f"🟡 Risk Level: **{risk}**")
            else:
                st.success(f"🟢 Risk Level: **{risk}**")
        with col2:
            st.metric("Subscription Probability", f"{prob:.1%}")
        with col3:
            verdict = "✅ Subscribe" if result["prediction"] == 1 else "❌ No Subscription"
            st.metric("Prediction (threshold=0.5)", verdict)

        st.info(f"**Duration excluded**: {result.get('duration_excluded', True)}")

        # Top SHAP features (if available)
        shap_features = result.get("top_shap_features")
        if shap_features:
            st.subheader("Top 5 Influential Features (SHAP)")
            import pandas as pd, matplotlib.pyplot as plt
            shap_df = pd.DataFrame(shap_features)
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ["#1f77b4" if v >= 0 else "#d62728" for v in shap_df["shap_value"]]
            ax.barh(shap_df["feature"], shap_df["shap_value"], color=colors)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("SHAP Value (impact on prediction)")
            ax.set_title("Feature Contributions")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
