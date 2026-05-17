"""
inference.py — Single and batch prediction entry points.

All prediction calls go through this module. `duration` is never accepted
in Feature Set B predictions — it is stripped with a warning if accidentally
provided.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_SET_B_COLS,
    MODELS_DIR,
    OPTIMAL_THRESHOLD,
    PREPROCESSING_PIPELINE_B_FILE,
    RANDOM_SEED,
    REALISTIC_MODEL_FILE,
    TARGET_COL,
)
from src.features import add_features, encode_target
from src.utils import get_logger

logger = get_logger(__name__)

# Required raw input columns for a valid single/batch prediction
_REQUIRED_INPUT_COLS: list[str] = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "campaign", "pdays", "previous",
    "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_input_schema(
    input_df: pd.DataFrame,
    feature_set: str = "set_b",
) -> list[str]:
    """Validate the input DataFrame schema.

    If ``duration`` is present in a ``set_b`` request, it is stripped with
    a logged warning (not an error).

    Parameters
    ----------
    input_df:
        DataFrame to validate.
    feature_set:
        Only ``"set_b"`` is currently supported for app predictions.

    Returns
    -------
    list[str]
        Empty list if valid; otherwise list of error message strings.
    """
    errors: list[str] = []

    if input_df.empty:
        errors.append("Input DataFrame is empty.")
        return errors

    # Strip duration silently if present (set_b only)
    if feature_set == "set_b" and "duration" in input_df.columns:
        warnings.warn(
            "'duration' column was present in the input but has been removed. "
            "Duration is not used in the Realistic Business Model (Feature Set B) "
            "as it is only known after the call is completed.",
            UserWarning,
            stacklevel=2,
        )

    # Check required columns
    for col in _REQUIRED_INPUT_COLS:
        if col not in input_df.columns:
            errors.append(f"Missing required column: '{col}'")

    return errors


# ---------------------------------------------------------------------------
# Load model + pipeline
# ---------------------------------------------------------------------------

def load_model_and_pipeline(
    model_path: str | Path | None = None,
    pipeline_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Load the fitted model and preprocessing pipeline from disk.

    Uses default paths if not specified.

    Parameters
    ----------
    model_path:
        Path to the ``*.joblib`` model file.  Defaults to the realistic
        business model in ``models/``.
    pipeline_path:
        Path to the preprocessing pipeline ``.joblib`` file.

    Returns
    -------
    tuple
        ``(model, pipeline)`` — both fitted sklearn objects.

    Raises
    ------
    FileNotFoundError
        If either file is missing, with a clear message directing the user
        to run ``make train``.
    """
    if model_path is None:
        model_path = MODELS_DIR / REALISTIC_MODEL_FILE
    if pipeline_path is None:
        pipeline_path = MODELS_DIR / PREPROCESSING_PIPELINE_B_FILE

    mp = Path(model_path)
    pp = Path(pipeline_path)

    if not mp.exists():
        raise FileNotFoundError(
            f"Model file not found: {mp.resolve()}\n"
            "Run `make train` (or `python scripts/train.py`) to train the model first."
        )
    if not pp.exists():
        raise FileNotFoundError(
            f"Pipeline file not found: {pp.resolve()}\n"
            "Run `make train` (or `python scripts/train.py`) to generate pipeline artifacts."
        )

    model = joblib.load(mp)
    pipeline = joblib.load(pp)
    logger.info("Model loaded ← %s", mp.name)
    logger.info("Pipeline loaded ← %s", pp.name)
    return model, pipeline


# ---------------------------------------------------------------------------
# Risk level helper
# ---------------------------------------------------------------------------

def risk_level(probability: float) -> str:
    """Convert a probability score to a risk/opportunity tier.

    Parameters
    ----------
    probability:
        Predicted subscription probability (0–1).

    Returns
    -------
    str
        ``"High"`` (≥ 0.6), ``"Medium"`` (≥ 0.3), or ``"Low"`` (< 0.3).
    """
    if probability >= 0.6:
        return "High"
    elif probability >= 0.3:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

def predict_single(
    model: Any,
    pipeline: Any,
    input_dict: dict,
) -> dict:
    """Predict subscription probability for a single customer.

    Parameters
    ----------
    model:
        Fitted sklearn Pipeline (includes the preprocessing step).
    pipeline:
        Unused — kept for API compatibility.  The ``model`` argument already
        contains the full pipeline.
    input_dict:
        Customer data as a plain dict.  Must contain all 19 required columns.
        ``duration`` is silently dropped if present.

    Returns
    -------
    dict
        Keys: ``predicted_class`` (int), ``probability`` (float),
        ``risk_level`` (str), ``top_features`` (list[dict]),
        ``duration_excluded`` (bool = True).

    Raises
    ------
    ValueError
        If schema validation fails.
    """
    # Build DataFrame
    df = pd.DataFrame([input_dict])

    # Strip duration silently
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
        logger.warning("'duration' stripped from single prediction input.")

    # Validate schema
    errors = validate_input_schema(df, feature_set="set_b")
    if errors:
        raise ValueError("Input schema validation failed:\n" + "\n".join(errors))

    # Feature engineering
    df = add_features(df, dataset_type="additional")

    # Predict — model is the full Pipeline
    y_proba = model.predict_proba(df)[:, 1][0]
    y_class = int(y_proba >= OPTIMAL_THRESHOLD)  # tuned threshold: best Subscribe F1
    tier = risk_level(y_proba)

    # SHAP top features (graceful fallback)
    top_features: list[dict] = []
    try:
        from src.explainability import shap_single_prediction
        final_estimator = model.named_steps.get("model", model)
        X_transformed = model.named_steps["preprocessor"].transform(df)
        try:
            feature_names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
        except Exception:
            feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]
        X_df = pd.DataFrame(X_transformed, columns=feature_names)
        top_features = shap_single_prediction(final_estimator, X_df)
    except Exception as exc:
        logger.debug("SHAP failed (non-critical): %s", exc)

    return {
        "prediction": y_class,
        "subscription_probability": float(y_proba),
        "risk_level": tier,
        "top_shap_features": top_features,
        "duration_excluded": True,
    }


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_batch(
    model: Any,
    pipeline: Any,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict subscription probabilities for a batch of customers.

    Parameters
    ----------
    model:
        Fitted sklearn Pipeline.
    pipeline:
        Unused — kept for API compatibility.
    df:
        DataFrame with all 19 required input columns.  ``duration`` is
        stripped silently if present.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with columns:
        ``predicted_class``, ``subscription_probability``, ``rank``.
        Sorted by ``subscription_probability`` descending.

    Raises
    ------
    ValueError
        If input is empty or schema validation fails.
    """
    if df.empty:
        raise ValueError("Input DataFrame for batch prediction is empty.")

    df = df.copy()

    # Strip duration
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
        logger.warning("'duration' column stripped from batch prediction input.")

    # Validate schema
    errors = validate_input_schema(df, feature_set="set_b")
    if errors:
        raise ValueError("Batch input schema validation failed:\n" + "\n".join(errors))

    # Feature engineering
    df_features = add_features(df, dataset_type="additional")

    # Predict — use tuned threshold for better Subscribe recall
    y_proba = model.predict_proba(df_features)[:, 1]
    y_class = (y_proba >= OPTIMAL_THRESHOLD).astype(int)

    result = df.copy()
    result["predicted_class"] = y_class
    result["subscription_probability"] = y_proba
    result["risk_level"] = [risk_level(p) for p in y_proba]
    result = result.sort_values("subscription_probability", ascending=False)
    result["rank"] = range(1, len(result) + 1)

    logger.info(
        "Batch prediction complete: %d rows, %d predicted positive",
        len(result), int(y_class.sum()),
    )
    return result
