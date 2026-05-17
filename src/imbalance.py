"""
imbalance.py — Strategies for handling class imbalance (~11.3% positive rate).

Provides resampling pipelines using imbalanced-learn:
- SMOTE (Synthetic Minority Over-sampling Technique)
- SMOTENC (SMOTE for mixed numeric + categorical features)
- RandomOverSampler
- RandomUnderSampler
- SMOTETomek (combined over + under-sampling)
- ADASYN (Adaptive Synthetic Sampling)

Each function returns an imblearn Pipeline that preprocesses then resamples,
ready to be wrapped in an outer sklearn Pipeline with a classifier.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Strategy catalogue
# ---------------------------------------------------------------------------

STRATEGY_DESCRIPTIONS = {
    "none":              "No resampling — rely on class_weight='balanced'",
    "smote":             "SMOTE — synthetic over-sampling of minority class",
    "smotenc":           "SMOTENC — SMOTE for mixed numeric + categorical features",
    "random_over":       "RandomOverSampler — duplicate minority samples",
    "random_under":      "RandomUnderSampler — drop majority samples",
    "smote_tomek":       "SMOTETomek — SMOTE + Tomek link removal",
    "adasyn":            "ADASYN — adaptive synthetic over-sampling",
}


def get_resampler(
    strategy: str = "smote",
    random_state: int = 42,
    categorical_features: list[int] | None = None,
    **kwargs: Any,
) -> Any:
    """Return an unfitted imbalanced-learn resampler for the given *strategy*.

    Parameters
    ----------
    strategy:
        One of: ``'none'``, ``'smote'``, ``'smotenc'``, ``'random_over'``,
        ``'random_under'``, ``'smote_tomek'``, ``'adasyn'``.
    random_state:
        Random seed.
    categorical_features:
        List of column *indices* of categorical features (required for ``'smotenc'``).
    **kwargs:
        Additional keyword arguments forwarded to the resampler constructor.

    Returns
    -------
    imbalanced-learn resampler, or ``None`` if strategy is ``'none'``.
    """
    try:
        from imblearn.over_sampling import (
            SMOTE, SMOTENC, RandomOverSampler, ADASYN,
        )
        from imblearn.combine import SMOTETomek
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError as e:
        raise ImportError(
            "imbalanced-learn is required. Install with: pip install imbalanced-learn"
        ) from e

    strategy = strategy.lower()

    if strategy == "none":
        return None
    elif strategy == "smote":
        return SMOTE(random_state=random_state, **kwargs)
    elif strategy == "smotenc":
        if categorical_features is None:
            raise ValueError("'smotenc' requires categorical_features (list of int indices).")
        return SMOTENC(
            categorical_features=categorical_features,
            random_state=random_state, **kwargs,
        )
    elif strategy == "random_over":
        return RandomOverSampler(random_state=random_state, **kwargs)
    elif strategy == "random_under":
        return RandomUnderSampler(random_state=random_state, **kwargs)
    elif strategy == "smote_tomek":
        return SMOTETomek(
            smote=SMOTE(random_state=random_state),
            random_state=random_state, **kwargs,
        )
    elif strategy == "adasyn":
        return ADASYN(random_state=random_state, **kwargs)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGY_DESCRIPTIONS)}"
        )


def build_imbalanced_pipeline(
    preprocessor: ColumnTransformer,
    classifier: Any,
    strategy: str = "smote",
    random_state: int = 42,
    categorical_feature_indices: list[int] | None = None,
) -> Any:
    """Build an imbalanced-learn Pipeline: preprocessor → resampler → classifier.

    Parameters
    ----------
    preprocessor:
        Unfitted :class:`~sklearn.compose.ColumnTransformer`.
    classifier:
        Unfitted sklearn-compatible classifier.
    strategy:
        Resampling strategy name (see :func:`get_resampler`).
    random_state:
        Random seed.
    categorical_feature_indices:
        Column indices for SMOTENC (only needed if strategy='smotenc').

    Returns
    -------
    imblearn.pipeline.Pipeline
        Unfitted pipeline: preprocessor → resampler → classifier.
        Falls back to a plain sklearn Pipeline if strategy is ``'none'``.
    """
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError as e:
        raise ImportError(
            "imbalanced-learn is required. Install with: pip install imbalanced-learn"
        ) from e

    from sklearn.pipeline import Pipeline as SkPipeline

    resampler = get_resampler(
        strategy=strategy,
        random_state=random_state,
        categorical_features=categorical_feature_indices,
    )

    if resampler is None:
        logger.info("No resampling — building plain sklearn Pipeline.")
        return SkPipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

    logger.info("Building imbalanced pipeline with strategy='%s'.", strategy)
    return ImbPipeline([
        ("preprocessor", preprocessor),
        ("resampler", resampler),
        ("classifier", classifier),
    ])


def compare_resampling_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    classifier_factory,
    strategies: list[str] | None = None,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compare multiple resampling strategies using cross-validation.

    Parameters
    ----------
    X_train, y_train:
        Training set.
    X_test, y_test:
        Test set.
    preprocessor:
        Unfitted ColumnTransformer.
    classifier_factory:
        Callable returning a fresh unfitted classifier, e.g.
        ``lambda: LGBMClassifier(random_state=42)``.
    strategies:
        List of strategy names to compare (default: all strategies).
    cv:
        Number of CV folds.
    random_state:
        Random seed.
    n_jobs:
        Parallelism for cross_validate.

    Returns
    -------
    pd.DataFrame
        One row per strategy with CV and test PR-AUC / ROC-AUC.
    """
    import time
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold, cross_validate as sk_cv
    from sklearn.metrics import average_precision_score, roc_auc_score

    if strategies is None:
        strategies = list(STRATEGY_DESCRIPTIONS.keys())

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []

    for strat in strategies:
        logger.info("Evaluating resampling strategy: %s", strat)
        t0 = time.time()
        try:
            pipe = build_imbalanced_pipeline(
                clone(preprocessor),
                classifier_factory(),
                strategy=strat,
                random_state=random_state,
            )
            cv_res = sk_cv(
                pipe, X_train, y_train,
                cv=skf,
                scoring=["average_precision", "roc_auc"],
                n_jobs=n_jobs,
                error_score="raise",
            )
            pipe.fit(X_train, y_train)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            test_pr  = average_precision_score(y_test, y_proba)
            test_roc = roc_auc_score(y_test, y_proba)

            rows.append({
                "strategy":           strat,
                "description":        STRATEGY_DESCRIPTIONS[strat],
                "cv_pr_auc_mean":     round(cv_res["test_average_precision"].mean(), 4),
                "cv_pr_auc_std":      round(cv_res["test_average_precision"].std(), 4),
                "cv_roc_auc_mean":    round(cv_res["test_roc_auc"].mean(), 4),
                "test_pr_auc":        round(test_pr, 4),
                "test_roc_auc":       round(test_roc, 4),
                "elapsed_s":          round(time.time() - t0, 1),
                "error":              "",
            })
            logger.info(
                "  %s → CV PR-AUC=%.4f  Test PR-AUC=%.4f",
                strat, rows[-1]["cv_pr_auc_mean"], test_pr,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("  %s failed: %s", strat, exc)
            rows.append({
                "strategy": strat, "description": STRATEGY_DESCRIPTIONS[strat],
                "cv_pr_auc_mean": float("nan"), "cv_pr_auc_std": float("nan"),
                "cv_roc_auc_mean": float("nan"),
                "test_pr_auc": float("nan"), "test_roc_auc": float("nan"),
                "elapsed_s": round(time.time() - t0, 1), "error": str(exc),
            })

    return pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False).reset_index(drop=True)
