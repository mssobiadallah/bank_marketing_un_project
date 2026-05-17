"""
modeling.py — Baseline model definitions, training, and persistence.

All classifiers use class_weight='balanced' where supported to handle the
~11.3% positive class imbalance (research.md Decision 3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import CV_FOLDS, RANDOM_SEED
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

def get_baseline_models(random_state: int = RANDOM_SEED) -> dict[str, Any]:
    """Return a dict of {model_name: unfitted estimator} for all baseline models.

    All tree-based / ensemble models use ``class_weight='balanced'`` where
    supported.  Optional boosting libraries (XGBoost, LightGBM, CatBoost)
    are added if available.

    Parameters
    ----------
    random_state:
        Integer seed for reproducibility.

    Returns
    -------
    dict[str, estimator]
        Mapping of short model name to unfitted sklearn-compatible estimator.
    """
    models: dict[str, Any] = {
        "DummyClassifier": DummyClassifier(
            strategy="stratified", random_state=random_state
        ),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            random_state=random_state,
            class_weight="balanced",
        ),
        "KNeighborsClassifier": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=1,
        ),
    }

    # Optional: XGBoost
    try:
        from xgboost import XGBClassifier  # type: ignore
        models["XGBClassifier"] = XGBClassifier(
            scale_pos_weight=8,  # approx. (no/yes ratio)
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=1,
            verbosity=0,
        )
        logger.info("XGBClassifier added.")
    except ImportError:
        logger.info("XGBoost not installed — skipping XGBClassifier.")

    # Optional: LightGBM
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        models["LGBMClassifier"] = LGBMClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        )
        logger.info("LGBMClassifier added.")
    except ImportError:
        logger.info("LightGBM not installed — skipping LGBMClassifier.")

    # Optional: CatBoost
    try:
        from catboost import CatBoostClassifier  # type: ignore
        models["CatBoostClassifier"] = CatBoostClassifier(
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=0,
        )
        logger.info("CatBoostClassifier added.")
    except ImportError:
        logger.info("CatBoost not installed — skipping CatBoostClassifier.")

    return models


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    cv: int = CV_FOLDS,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit a full sklearn Pipeline (preprocessor → model) and compute CV scores.

    Parameters
    ----------
    model:
        Unfitted sklearn-compatible estimator.
    X_train:
        Training features (raw, before preprocessing).
    y_train:
        Training labels (0/1 integers).
    preprocessor:
        Unfitted :class:`~sklearn.compose.ColumnTransformer`.
    cv:
        Number of cross-validation folds.

    Returns
    -------
    tuple
        ``(fitted_pipeline, cv_scores_dict)``

        ``cv_scores_dict`` keys: ``"cv_roc_auc_mean"``, ``"cv_roc_auc_std"``,
        ``"cv_average_precision_mean"``, ``"cv_average_precision_std"``.
    """
    from sklearn.model_selection import cross_validate
    from sklearn.base import clone

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    cv_pipeline = Pipeline(
        steps=[("preprocessor", clone(preprocessor)), ("model", clone(model))]
    )
    cv_results = cross_validate(
        cv_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=["roc_auc", "average_precision"],
        n_jobs=1,
    )

    cv_scores = {
        "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        "cv_average_precision_mean": float(np.mean(cv_results["test_average_precision"])),
        "cv_average_precision_std": float(np.std(cv_results["test_average_precision"])),
    }
    logger.info(
        "Trained %s — CV PR-AUC=%.4f ± %.4f",
        type(model).__name__,
        cv_scores["cv_average_precision_mean"],
        cv_scores["cv_average_precision_std"],
    )
    return pipeline, cv_scores


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(pipeline: Pipeline, path: str | Path) -> Path:
    """Serialise *pipeline* to *path* using joblib.

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline.
    path:
        Destination ``.joblib`` file path.

    Returns
    -------
    Path
        Resolved destination path.
    """
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(pipeline, p)
    logger.info("Saved model → %s", p)
    return p


def load_model(path: str | Path) -> Pipeline:
    """Load a joblib-serialised model pipeline.

    Parameters
    ----------
    path:
        Path to the ``.joblib`` file.

    Returns
    -------
    Pipeline
        Deserialised sklearn Pipeline.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model file not found: {p.resolve()}\n"
            "Run `make train` (or `python scripts/train.py`) to generate model artifacts."
        )
    pipeline = joblib.load(p)
    logger.info("Loaded model ← %s", p)
    return pipeline
