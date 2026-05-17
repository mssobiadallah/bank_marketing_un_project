"""1_EDA_Dashboard.py — Exploratory Data Analysis page."""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import RAW_DIR, MAIN_DATASET, CSV_SEP, TARGET_COL
from src.features import encode_target, add_features

st.set_page_config(page_title="EDA Dashboard", page_icon="📈", layout="wide")
st.title("📈 Exploratory Data Analysis")

@st.cache_data(show_spinner="Loading dataset…")
def _load():
    df = pd.read_csv(RAW_DIR / MAIN_DATASET, sep=CSV_SEP)
    df = encode_target(df, TARGET_COL)
    df = add_features(df)
    return df

try:
    df = _load()
except FileNotFoundError:
    st.error("Raw dataset not found. Ensure `bank-additional/bank-additional-full.csv` exists.")
    st.stop()

tabs = st.tabs(["Target Distribution", "Numeric Features", "Categorical Features",
                "Bivariate Analysis", "Engineered Features"])

# ------------------------------------------------------------------
with tabs[0]:
    st.subheader("Target Variable Distribution")
    st.markdown(
        "The dataset is **highly imbalanced**: roughly 88.7% of clients did *not* subscribe."
    )
    counts = df[TARGET_COL].value_counts().rename({0: "No (0)", 1: "Yes (1)"})
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(
            counts.rename("count").reset_index().rename(columns={"index": "Class"}),
            use_container_width=True,
        )
        rate = df[TARGET_COL].mean() * 100
        st.metric("Positive Rate", f"{rate:.1f}%")
    with col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        counts.plot(kind="bar", ax=ax, color=["#d62728", "#1f77b4"], edgecolor="white")
        ax.set_xticklabels(["No", "Yes"], rotation=0)
        ax.set_ylabel("Count")
        ax.set_title("Subscription Counts")
        st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------------
with tabs[1]:
    st.subheader("Numeric Feature Distributions")
    numeric_cols = df.select_dtypes("number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]
    selected = st.multiselect("Select features", numeric_cols, default=numeric_cols[:4])
    if selected:
        fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 3))
        if len(selected) == 1:
            axes = [axes]
        for ax, col in zip(axes, selected):
            sns.histplot(df[col], kde=True, ax=ax, color="#1f77b4")
            ax.set_title(col, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(df[selected].describe().round(2), use_container_width=True)

# ------------------------------------------------------------------
with tabs[2]:
    st.subheader("Categorical Feature Value Counts")
    cat_cols = df.select_dtypes("object").columns.tolist()
    selected_cat = st.selectbox("Select feature", cat_cols)
    if selected_cat:
        vc = df[selected_cat].value_counts()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(vc.rename("count").reset_index(), use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 3))
            vc.plot(kind="bar", ax=ax, color="#1f77b4", edgecolor="white")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ------------------------------------------------------------------
with tabs[3]:
    st.subheader("Bivariate Analysis — Feature vs. Subscription")
    numeric_cols2 = [c for c in df.select_dtypes("number").columns if c != TARGET_COL]
    feature = st.selectbox("Numeric feature", numeric_cols2, key="biv_num")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    # Create a copy with string target for plotting
    df_plot = df.copy()
    df_plot[TARGET_COL] = df_plot[TARGET_COL].map({0: "No", 1: "Yes"})
    sns.boxplot(data=df_plot, x=TARGET_COL, y=feature, ax=axes[0], palette={"No": "#d62728", "Yes": "#1f77b4"})
    axes[0].set_title(f"{feature} by Subscription")
    sns.violinplot(data=df_plot, x=TARGET_COL, y=feature, ax=axes[1], palette={"No": "#d62728", "Yes": "#1f77b4"})
    axes[1].set_title("Violin Plot")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------------
with tabs[4]:
    st.subheader("Engineered Features")
    from src.config import ENGINEERED_FEATURE_NAMES
    eng_present = [c for c in ENGINEERED_FEATURE_NAMES if c in df.columns]
    st.markdown(f"**{len(eng_present)} engineered features** added to the raw dataset.")
    st.dataframe(df[eng_present].describe().round(2), use_container_width=True)
    st.markdown("**Subscription rate by `age_group`**")
    if "age_group" in df.columns:
        grp = df.groupby("age_group", observed=True)[TARGET_COL].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3))
        grp.plot(kind="bar", ax=ax, color="#1f77b4", edgecolor="white")
        ax.set_ylabel("Subscription Rate")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
