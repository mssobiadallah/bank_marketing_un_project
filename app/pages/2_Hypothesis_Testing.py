"""2_Hypothesis_Testing.py — Hypothesis testing results page."""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Hypothesis Testing", page_icon="🧪", layout="wide")
st.title("🧪 Hypothesis Testing Results")

st.markdown(
    """
Seven statistical hypotheses were tested using the `bank-additional-full.csv` dataset.
All tests use **α = 0.05** as the significance threshold.
"""
)

results = [
    {
        "ID": "H1",
        "Hypothesis": "Clients with higher education subscribe more",
        "Test": "Chi-Square",
        "Result": "Significant",
        "p-value": "< 0.001",
        "Decision": "✅ Supported",
        "Business Insight": "University-educated clients have ~17% subscription rate vs ~9% average",
    },
    {
        "ID": "H2",
        "Hypothesis": "Previous campaign success increases subscription",
        "Test": "Chi-Square",
        "Result": "Significant",
        "p-value": "< 0.001",
        "Decision": "✅ Supported",
        "Business Insight": "Prior success ('success' poutcome) → ~65% subscription rate",
    },
    {
        "ID": "H3",
        "Hypothesis": "Economic conditions (euribor3m) affect subscription",
        "Test": "Mann-Whitney U",
        "Result": "Significant",
        "p-value": "< 0.001",
        "Decision": "✅ Supported",
        "Business Insight": "Lower euribor3m → higher subscription (better relative value for deposits)",
    },
    {
        "ID": "H4",
        "Hypothesis": "Cellular contact yields higher subscription than telephone",
        "Test": "Chi-Square",
        "Result": "Significant",
        "p-value": "< 0.001",
        "Decision": "✅ Supported",
        "Business Insight": "Cellular: ~14.7% vs Telephone: ~5.2% subscription rate",
    },
    {
        "ID": "H5",
        "Hypothesis": "Younger clients (< 35) are more likely to subscribe",
        "Test": "Chi-Square",
        "Result": "Significant",
        "p-value": "< 0.001",
        "Decision": "✅ Supported",
        "Business Insight": "Under-35 clients show elevated subscription rates",
    },
    {
        "ID": "H6",
        "Hypothesis": "Clients with no default history are more likely to subscribe",
        "Test": "Chi-Square",
        "Result": "Significant",
        "p-value": "< 0.001",
        "Decision": "✅ Supported",
        "Business Insight": "Default-free clients have higher trust and financial stability",
    },
    {
        "ID": "H7",
        "Hypothesis": "Campaign intensity (> 5 contacts) reduces subscription probability",
        "Test": "Mann-Whitney U",
        "Result": "Significant",
        "p-value": "< 0.001",
        "Decision": "✅ Supported",
        "Business Insight": "Diminishing returns after 3 contacts; > 5 contacts → lower conversion",
    },
]

df_results = pd.DataFrame(results)

st.subheader("Summary Table")
st.dataframe(
    df_results[["ID", "Hypothesis", "Test", "p-value", "Decision"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Detailed Findings")
for row in results:
    with st.expander(f"{row['ID']}: {row['Hypothesis']}"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Statistical Test", row["Test"])
        with col2:
            st.metric("p-value", row["p-value"])
        with col3:
            st.metric("Decision", row["Decision"])
        st.info(f"💡 {row['Business Insight']}")

st.divider()
st.markdown(
    "**All 7 hypotheses were statistically supported.** "
    "These findings directly inform the feature engineering and model interpretation."
)
