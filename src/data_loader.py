"""
data_loader.py — Functions for loading, validating, and comparing datasets.

All functions return plain pandas objects and raise clear exceptions on bad input.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

from src.config import (
    ALL_DATASETS,
    CATEGORICAL_COLS,
    CSV_SEP,
    FEATURE_SET_A_COLS,
    FEATURE_SET_B_COLS,
    TARGET_COL,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path, sep: str = ";") -> pd.DataFrame:
    """Load a CSV dataset from *path*.

    Parameters
    ----------
    path:
        Path to the CSV file (absolute or relative to the working directory).
    sep:
        Column separator character.  The bank-additional datasets use ``";"``
        (default).

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the resulting DataFrame is empty, or if the wrong separator is
        detected (i.e. the file loaded as a single column, suggesting the
        separator did not split any fields).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p.resolve()}")

    df = pd.read_csv(p, sep=sep)

    if df.empty:
        raise ValueError(f"Dataset loaded as empty DataFrame: {p}")

    if len(df.columns) == 1:
        raise ValueError(
            f"Dataset loaded with only 1 column — wrong separator? "
            f"File: {p}, sep={sep!r}.  Try sep=';' for bank datasets."
        )

    logger.info("Loaded %s — shape %s", p.name, df.shape)
    return df


# ---------------------------------------------------------------------------
# Summary / profiling
# ---------------------------------------------------------------------------

def summarize_dataset(df: pd.DataFrame) -> dict:
    """Compute a summary dictionary for *df*.

    Parameters
    ----------
    df:
        Any pandas DataFrame (typically one of the four bank datasets).

    Returns
    -------
    dict
        Keys:

        - ``shape`` — (n_rows, n_cols)
        - ``columns`` — list of column names
        - ``dtypes`` — dict of column → dtype string
        - ``missing_values`` — dict of column → missing count (where > 0)
        - ``duplicate_rows`` — number of fully duplicate rows
        - ``target_distribution`` — dict ``{value: count}`` if ``y`` is present,
          else ``None``
        - ``unknown_counts`` — dict of column → count of ``"unknown"`` values
          for string/object columns
    """
    summary: dict = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {
            col: int(count)
            for col, count in df.isnull().sum().items()
            if count > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": None,
        "unknown_counts": {},
    }

    if TARGET_COL in df.columns:
        summary["target_distribution"] = df[TARGET_COL].value_counts().to_dict()

    for col in df.select_dtypes(include=["object", "str"]).columns:
        n_unknown = int((df[col] == "unknown").sum())
        if n_unknown > 0:
            summary["unknown_counts"][col] = n_unknown

    return summary


# ---------------------------------------------------------------------------
# Multi-dataset comparison
# ---------------------------------------------------------------------------

def compare_datasets(paths: Mapping[str, str | Path]) -> pd.DataFrame:
    """Load and compare multiple datasets.

    Parameters
    ----------
    paths:
        Mapping of dataset label → file path.  Example::

            {
                "bank-full": "data/raw/bank-full.csv",
                "bank-additional-full": "data/raw/bank-additional-full.csv",
            }

    Returns
    -------
    pd.DataFrame
        One row per dataset with columns:

        ``name``, ``rows``, ``columns``, ``yes_pct``, ``no_pct``,
        ``unknown_total``, ``duplicate_rows``
    """
    records = []
    for name, path in paths.items():
        try:
            df = load_dataset(path)
            summary = summarize_dataset(df)
            target_dist = summary.get("target_distribution") or {}
            total = sum(target_dist.values()) if target_dist else 0
            yes_count = target_dist.get("yes", 0)
            no_count = target_dist.get("no", 0)
            records.append(
                {
                    "name": name,
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "yes_pct": round(yes_count / total * 100, 2) if total else None,
                    "no_pct": round(no_count / total * 100, 2) if total else None,
                    "unknown_total": sum(summary["unknown_counts"].values()),
                    "duplicate_rows": summary["duplicate_rows"],
                }
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping %s — %s", name, exc)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_required_columns(
    df: pd.DataFrame,
    feature_set: str = "set_b",
) -> list[str]:
    """Return a list of missing required columns for the given *feature_set*.

    Parameters
    ----------
    df:
        DataFrame to validate.
    feature_set:
        ``"set_a"`` (includes ``duration``) or ``"set_b"`` (excludes
        ``duration``).  Default ``"set_b"`` — the Realistic Business Model.

    Returns
    -------
    list[str]
        Empty list if all required columns are present; otherwise the names
        of missing columns.

    Raises
    ------
    ValueError
        If *feature_set* is not ``"set_a"`` or ``"set_b"``.
    """
    if feature_set == "set_a":
        required = FEATURE_SET_A_COLS
    elif feature_set == "set_b":
        required = FEATURE_SET_B_COLS
    else:
        raise ValueError(f"feature_set must be 'set_a' or 'set_b', got {feature_set!r}")

    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.warning(
            "validate_required_columns (%s): missing columns: %s", feature_set, missing
        )
    return missing
