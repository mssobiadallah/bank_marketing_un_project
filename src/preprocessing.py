"""
preprocessing.py — sklearn preprocessing pipeline factory and train/test split.

All preprocessing logic lives here. The pipeline is saved/loaded as a .joblib
artifact and reused consistently across training and inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_SEED, TEST_SIZE
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build a :class:`~sklearn.compose.ColumnTransformer` preprocessing pipeline.

    Categorical columns are one-hot encoded with ``handle_unknown='ignore'``
    to handle unseen categories at inference time (per research.md Decision 6).

    Numeric columns are passed through by default (``scale_numeric=False``).
    Passing ``scale_numeric=True`` wraps numerics in a
    :class:`~sklearn.preprocessing.StandardScaler` — useful for
    :class:`~sklearn.linear_model.LogisticRegression` and KNN.

    Parameters
    ----------
    numeric_cols:
        List of numeric column names.
    categorical_cols:
        List of categorical column names.
    scale_numeric:
        Whether to apply ``StandardScaler`` to numeric columns.

    Returns
    -------
    ColumnTransformer
        Unfitted transformer.
    """
    if scale_numeric:
        num_transformer: Any = Pipeline(steps=[("scaler", StandardScaler())])
    else:
        num_transformer = "passthrough"

    cat_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target: str = "y",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Parameters
    ----------
    df:
        DataFrame with features and target column (target must already be 0/1).
    target:
        Name of the target column.
    test_size:
        Fraction of data for the test set.  Default ``0.20``.
    random_state:
        Random seed for reproducibility.  Default ``42``.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``
    """
    from sklearn.model_selection import train_test_split  # local import avoids
    # circular dependency at module load time when running tests

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "split_data: train=%d, test=%d (stratified, test_size=%.0f%%)",
        len(X_train), len(X_test), test_size * 100,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_pipeline(pipeline: object, path: str | Path) -> Path:
    """Serialise *pipeline* to *path* using joblib.

    Creates the parent directory automatically.

    Parameters
    ----------
    pipeline:
        Any fitted sklearn transformer or pipeline.
    path:
        Destination ``.joblib`` path.

    Returns
    -------
    Path
        Resolved destination path.
    """
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(pipeline, p)
    logger.info("Saved pipeline → %s", p)
    return p


def load_pipeline(path: str | Path) -> object:
    """Load a joblib-serialised pipeline from *path*.

    Parameters
    ----------
    path:
        Path to the ``.joblib`` file.

    Returns
    -------
    object
        The deserialised pipeline.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pipeline file not found: {p.resolve()}")
    obj = joblib.load(p)
    logger.info("Loaded pipeline ← %s", p)
    return obj
