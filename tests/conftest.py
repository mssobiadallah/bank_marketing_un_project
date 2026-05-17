"""
conftest.py — Shared pytest fixtures for the Bank Marketing ML test suite.

Fixtures are available to all test files automatically via pytest's conftest
discovery mechanism.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

# Ensure the project root is on sys.path so `src.*` imports work
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MAIN_DATASET_PATH,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Raw data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """A minimal 10-row DataFrame with all 21 bank-additional columns.

    Uses real values representative of the dataset schema.
    This fixture is fast (no file I/O) and suitable for unit tests.
    """
    data = {
        "age":           [44, 53, 28, 39, 55, 31, 67, 47, 36, 24],
        "job":           ["management", "blue-collar", "student", "technician",
                          "retired", "admin.", "retired", "services", "entrepreneur", "student"],
        "marital":       ["married", "married", "single", "married", "married",
                          "single", "married", "divorced", "married", "single"],
        "education":     ["university.degree", "high.school", "university.degree",
                          "professional.course", "basic.4y", "high.school",
                          "basic.6y", "high.school", "university.degree", "university.degree"],
        "default":       ["no", "unknown", "no", "no", "no", "no", "no", "no", "no", "no"],
        "housing":       ["yes", "yes", "no", "yes", "no", "yes", "no", "yes", "yes", "no"],
        "loan":          ["no", "no", "no", "no", "no", "yes", "no", "no", "no", "no"],
        "contact":       ["cellular"] * 8 + ["telephone"] * 2,
        "month":         ["may", "may", "jun", "aug", "nov", "jun", "dec", "may", "jul", "aug"],
        "day_of_week":   ["mon", "tue", "wed", "thu", "fri", "mon", "tue", "wed", "thu", "fri"],
        "duration":      [261, 149, 226, 151, 307, 198, 83, 92, 341, 75],
        "campaign":      [1, 1, 2, 1, 3, 1, 1, 2, 1, 4],
        "pdays":         [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        "previous":      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "poutcome":      ["nonexistent"] * 10,
        "emp.var.rate":  [1.1, 1.1, 1.4, 1.4, -1.8, 1.4, -3.4, 1.1, 1.4, 1.4],
        "cons.price.idx":[93.994, 93.994, 94.465, 93.444, 94.767, 94.465,
                          92.893, 93.994, 93.918, 93.444],
        "cons.conf.idx": [-36.4, -36.4, -41.8, -36.1, -46.2, -41.8,
                          -31.4, -36.4, -42.7, -36.1],
        "euribor3m":     [4.857, 4.857, 4.961, 4.963, 1.334, 4.961,
                          0.787, 4.857, 4.959, 4.963],
        "nr.employed":   [5191.0, 5191.0, 5228.1, 5228.1, 5099.1, 5228.1,
                          5008.7, 5191.0, 5228.1, 5228.1],
        TARGET_COL:      ["no", "no", "yes", "no", "yes", "no", "yes", "no", "no", "no"],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def full_df() -> pd.DataFrame:
    """Load the full bank-additional-full.csv dataset (41188 rows × 21 columns).

    Session-scoped so the file is read only once per test session.
    Skips tests automatically if the file is missing.
    """
    if not MAIN_DATASET_PATH.exists():
        pytest.skip(f"Main dataset not found: {MAIN_DATASET_PATH}")
    return pd.read_csv(MAIN_DATASET_PATH, sep=";")


@pytest.fixture(scope="session")
def processed_df(full_df: pd.DataFrame) -> pd.DataFrame:
    """Return the main dataset with target encoded and all 9 engineered features added.

    Depends on ``full_df``.  Session-scoped.
    """
    from src.features import encode_target, add_features

    df = full_df.copy()
    df = encode_target(df, target=TARGET_COL)
    df = add_features(df, dataset_type="additional")
    return df


@pytest.fixture(scope="session")
def fitted_pipeline(processed_df: pd.DataFrame):
    """Return a fitted sklearn Pipeline (preprocessor only) on the processed data.

    Uses Feature Set B (no duration). Session-scoped.
    """
    from src.features import get_feature_lists
    from src.preprocessing import build_preprocessing_pipeline, split_data

    feature_info = get_feature_lists(processed_df, target=TARGET_COL, exclude_duration=True)
    X = processed_df[feature_info["numeric"] + feature_info["categorical"]]
    y = processed_df[TARGET_COL]

    X_train, _, y_train, _ = split_data(
        processed_df, target=TARGET_COL, test_size=0.2, random_state=RANDOM_SEED
    )
    preprocessor = build_preprocessing_pipeline(
        numeric_cols=feature_info["numeric"],
        categorical_cols=feature_info["categorical"],
    )
    preprocessor.fit(X_train[feature_info["numeric"] + feature_info["categorical"]])
    return preprocessor
