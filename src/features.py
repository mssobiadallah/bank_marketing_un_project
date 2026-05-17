"""
features.py — Feature engineering and target encoding.

All 9 engineered features are added here. The target is encoded yes→1 / no→0.
Call add_features() AFTER loading and BEFORE preprocessing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLS,
    ENGINEERED_CATEGORICAL,
    ENGINEERED_FEATURE_NAMES,
    ENGINEERED_NUMERIC,
    NUMERIC_COLS,
    NUMERIC_COLS_SET_B,
    TARGET_COL,
)
from src.utils import get_logger

logger = get_logger(__name__)

# Month order lookup (jan=1 … dec=12)
_MONTH_ORDER: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------

def encode_target(df: pd.DataFrame, target: str = TARGET_COL) -> pd.DataFrame:
    """Map target column values from 'yes'/'no' strings to 1/0 integers.

    Operates **in-place** on a copy of *df* to avoid mutating the caller's
    DataFrame.

    Parameters
    ----------
    df:
        DataFrame containing the raw target column.
    target:
        Name of the target column.  Default ``"y"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with target column replaced by integer 0/1.

    Raises
    ------
    KeyError
        If *target* is not a column in *df*.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    df = df.copy()
    df[target] = df[target].map({"yes": 1, "no": 0})

    n_null = df[target].isnull().sum()
    if n_null > 0:
        warnings.warn(
            f"encode_target: {n_null} NaN values after mapping (unexpected values in '{target}')."
        )

    logger.info("Encoded target '%s' → 0/1 (%d rows).", target, len(df))
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame, dataset_type: str = "additional") -> pd.DataFrame:
    """Add all 9 engineered features to *df*.

    Engineered features
    -------------------
    1. ``was_previously_contacted`` — 1 if pdays != 999 else 0
    2. ``campaign_intensity_group`` — low (1) / medium (2–5) / high (>5)
    3. ``age_group`` — young (<30) / middle (30–60) / senior (>60)
    4. ``economic_stress_index`` — euribor3m + emp.var.rate composite
    5. ``has_any_loan`` — 1 if housing==yes OR loan==yes
    6. ``month_order`` — integer 1–12
    7. ``previous_contact_success_flag`` — 1 if poutcome==success
    8. ``contact_is_cellular`` — 1 if contact==cellular
    9. ``client_financial_pressure_flag`` — 1 if default==yes OR has_any_loan==1

    Parameters
    ----------
    df:
        DataFrame after loading (raw column names expected).
    dataset_type:
        ``"additional"`` (21-column schema) or ``"original"`` (17-column
        schema without economic features).  Default ``"additional"``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with 9 new columns appended.
    """
    df = df.copy()

    # 1. Was previously contacted?
    df["was_previously_contacted"] = (df["pdays"] != 999).astype(int)

    # 2. Campaign intensity group
    df["campaign_intensity_group"] = pd.cut(
        df["campaign"],
        bins=[0, 1, 5, 9999],
        labels=["low", "medium", "high"],
        right=True,
    ).astype(str)

    # 3. Age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 29, 60, 200],
        labels=["young", "middle", "senior"],
        right=True,
    ).astype(str)

    # 4. Economic stress index (only for bank-additional schema)
    if dataset_type == "additional" and "euribor3m" in df.columns and "emp.var.rate" in df.columns:
        df["economic_stress_index"] = df["euribor3m"] + df["emp.var.rate"]
    else:
        df["economic_stress_index"] = 0.0

    # 5. Has any loan
    df["has_any_loan"] = (
        (df["housing"].str.lower() == "yes") | (df["loan"].str.lower() == "yes")
    ).astype(int)

    # 6. Month order
    df["month_order"] = df["month"].str.lower().map(_MONTH_ORDER).fillna(0).astype(int)

    # 7. Previous contact success flag
    df["previous_contact_success_flag"] = (
        df["poutcome"].str.lower() == "success"
    ).astype(int)

    # 8. Contact is cellular
    df["contact_is_cellular"] = (
        df["contact"].str.lower() == "cellular"
    ).astype(int)

    # 9. Client financial pressure flag
    df["client_financial_pressure_flag"] = (
        (df["default"].str.lower() == "yes") | (df["has_any_loan"] == 1)
    ).astype(int)

    logger.info("add_features: added %d engineered features.", len(ENGINEERED_FEATURE_NAMES))
    return df


# ---------------------------------------------------------------------------
# Feature list builder
# ---------------------------------------------------------------------------

def get_feature_lists(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    exclude_duration: bool = False,
) -> dict:
    """Return separate lists of numeric and categorical feature columns.

    Parameters
    ----------
    df:
        DataFrame after target encoding and feature engineering.
    target:
        Target column name to exclude from feature lists.
    exclude_duration:
        If ``True`` (Feature Set B — Realistic Business Model), ``duration``
        is excluded from numeric columns.

    Returns
    -------
    dict
        Keys: ``"numeric"`` (list), ``"categorical"`` (list), ``"target"`` (str).
    """
    numeric = [c for c in NUMERIC_COLS if c in df.columns]
    if exclude_duration:
        numeric = [c for c in numeric if c != "duration"]

    # Add engineered numeric features
    for col in ENGINEERED_NUMERIC:
        if col in df.columns and col not in numeric:
            numeric.append(col)

    categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
    # Add engineered categorical features
    for col in ENGINEERED_CATEGORICAL:
        if col in df.columns and col not in categorical:
            categorical.append(col)

    # Sanity check — no target or duplicate columns
    numeric = [c for c in numeric if c != target]
    categorical = [c for c in categorical if c != target]

    # Remove duplicates while preserving order
    seen: set[str] = set()
    numeric_deduped = []
    for c in numeric:
        if c not in seen:
            numeric_deduped.append(c)
            seen.add(c)

    seen_cat: set[str] = set()
    categorical_deduped = []
    for c in categorical:
        if c not in seen_cat and c not in seen:
            categorical_deduped.append(c)
            seen_cat.add(c)

    return {"numeric": numeric_deduped, "categorical": categorical_deduped, "target": target}
