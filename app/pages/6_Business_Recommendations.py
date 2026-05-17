"""6_Business_Recommendations.py — Insights, segments, strategy."""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="Business Recommendations", page_icon="💡", layout="wide")
st.title("💡 Business Recommendations")

# ------------------------------------------------------------------
st.subheader("🎯 High-Value Customer Segments")

segments = [
    {
        "Segment": "Previous Success",
        "Description": "Clients with `poutcome = success` in prior campaigns",
        "Subscription Rate": "~65%",
        "Strategy": "Prioritise immediately — highest conversion",
        "Priority": "🔴 Critical",
    },
    {
        "Segment": "University Educated, Cellular",
        "Description": "University degree holders contacted by mobile",
        "Subscription Rate": "~17–22%",
        "Strategy": "Lead generation campaigns via SMS/app",
        "Priority": "🟠 High",
    },
    {
        "Segment": "Students & Retired",
        "Description": "Age extremes — students < 25, retired > 60",
        "Subscription Rate": "~20–25%",
        "Strategy": "Personalised offers: students (future planning), retired (security)",
        "Priority": "🟠 High",
    },
    {
        "Segment": "Low Euribor3m Environment",
        "Description": "Periods when euribor3m < 1.5%",
        "Subscription Rate": "Higher than average",
        "Strategy": "Intensify campaigns when interest rates are low",
        "Priority": "🟡 Medium",
    },
    {
        "Segment": "No Default, No Loans",
        "Description": "Clients with no credit default and no personal loan",
        "Subscription Rate": "~12–15%",
        "Strategy": "Cross-sell during routine outreach",
        "Priority": "🟡 Medium",
    },
]

import pandas as pd
st.dataframe(pd.DataFrame(segments), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
st.divider()
st.subheader("📞 Campaign Efficiency Guidelines")

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
**Optimal Contact Strategy**
- ✅ **1–3 contacts** per client per campaign (diminishing returns after 3)
- ✅ **Cellular** contact is ~3× more effective than telephone
- ✅ **March, September, October, December** show highest subscription rates
- ✅ **Tuesday–Thursday** contacts slightly outperform Monday/Friday
- ❌ Avoid contacting clients > 5 times in a single campaign
"""
    )
with col2:
    st.markdown(
        """
**Prioritisation Workflow**
1. Score all clients using this app's batch prediction
2. Rank by subscription probability (highest first)
3. Focus call centre resources on **High** and **Medium** risk tiers
4. Re-engage "previous success" clients first
5. Track outcomes → retrain model quarterly
"""
    )

# ------------------------------------------------------------------
st.divider()
st.subheader("⚠️ Model Limitations")

st.error(
    "**Duration Exclusion**: The realistic model deliberately excludes `duration`. "
    "The benchmark model (with duration) achieves PR-AUC ~0.70, but this is inflated by leakage. "
    "Always use the **Realistic Business Model** (PR-AUC ~0.50) for real decisions."
)

st.markdown(
    """
**Known Limitations:**
- Data sourced from a single Portuguese bank (2008–2013) — may not generalise globally
- Class imbalance (~11.3% positive) means even the best model has limited absolute recall
- Economic features (euribor3m, nr.employed) lag real-world changes — retrain regularly
- SHAP explanations are approximations, not causal explanations
- Model assumes current marketing channel mix; changes to channel strategy may affect performance
"""
)

# ------------------------------------------------------------------
st.divider()
st.subheader("⚖️ Ethical Considerations")

st.info(
    """
**Fair Lending & Non-Discrimination**
- The model uses `age`, `job`, `marital`, `education` — all protected-class adjacent attributes
- Predictions must not be used to *deny* services; only to *prioritise* outreach
- Regular bias audits recommended across demographic segments
- Clients should be informed if AI-based targeting is used (GDPR compliance)

**Data Privacy**
- All training data should be handled per applicable data protection regulations
- Client PII must not be included in model features beyond what is strictly necessary
- Model artifacts should be versioned and access-controlled
"""
)

# ------------------------------------------------------------------
st.divider()
st.subheader("📈 Expected Business Impact")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Expected Call Reduction", "30–40%", help="By targeting top deciles only")
with col2:
    st.metric("Conversion Rate Improvement", "2–3×", help="vs. random dialing")
with col3:
    st.metric("ROI on Campaign", "Significant", help="Fewer wasted calls = lower cost per acquisition")
