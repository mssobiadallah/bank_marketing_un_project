"""
explainability.py — SHAP-based and permutation-based feature importance helpers.

SHAP explainer is auto-selected per model type (research.md Decision 4):
- TreeExplainer  → tree-based models (RF, GB, XGB, LGB, CatBoost, etc.)
- LinearExplainer → linear models (LogisticRegression, etc.)
- KernelExplainer → all other models (slow — sampled to ≤200 rows)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure as MplFigure

from src.utils import ensure_dir, get_logger, save_figure

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def permutation_importance_table(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importance.

    Parameters
    ----------
    model:
        Fitted sklearn Pipeline or estimator.
    X_test:
        Test features (raw, before pipeline preprocessing if *model* is a
        Pipeline).
    y_test:
        True binary labels.
    n_repeats:
        Number of permutation repeats per feature.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``importance_mean``, ``importance_std``,
        sorted descending by ``importance_mean``.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="average_precision",
        n_jobs=1,
    )
    feature_names = X_test.columns.tolist()
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Tree / linear feature importance plot
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
    output_path: Optional[str | Path] = None,
) -> MplFigure:
    """Horizontal bar chart of model-internal feature importances.

    Handles tree-based ``.feature_importances_`` and linear ``.coef_``.

    Parameters
    ----------
    model:
        Fitted estimator (not a Pipeline — extract the final step first).
    feature_names:
        Ordered list of feature names matching the model's training columns.
    top_n:
        Number of top features to display.
    output_path:
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
    else:
        raise AttributeError(
            f"Model {type(model).__name__} has neither .feature_importances_ nor .coef_"
        )

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — Top {top_n}")
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

def get_shap_explainer(model: Any, X_sample: pd.DataFrame) -> Any:
    """Auto-select and return a fitted SHAP explainer for *model*.

    Selection logic (research.md Decision 4):
    - ``shap.TreeExplainer``   for tree-based models
    - ``shap.LinearExplainer`` for linear models
    - ``shap.KernelExplainer`` for all others (sampled to ≤200 rows)

    Parameters
    ----------
    model:
        Fitted estimator (extract from Pipeline if needed).
    X_sample:
        Background / reference data for the explainer.

    Returns
    -------
    SHAP explainer instance.

    Raises
    ------
    ImportError
        If ``shap`` is not installed.
    """
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "shap is required for explainability. Install with: pip install shap"
        ) from exc

    tree_types = (
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "HistGradientBoostingClassifier",
        "DecisionTreeClassifier",
        "XGBClassifier",
        "LGBMClassifier",
        "CatBoostClassifier",
    )
    linear_types = ("LogisticRegression", "LinearSVC", "SGDClassifier")

    model_class = type(model).__name__

    if model_class in tree_types:
        logger.info("Using TreeExplainer for %s", model_class)
        explainer = shap.TreeExplainer(model)
    elif model_class in linear_types:
        logger.info("Using LinearExplainer for %s", model_class)
        explainer = shap.LinearExplainer(model, X_sample)
    else:
        logger.info(
            "Using KernelExplainer for %s (slow — capped at 200 rows)", model_class
        )
        background = X_sample.iloc[:200] if len(X_sample) > 200 else X_sample
        explainer = shap.KernelExplainer(model.predict_proba, background)

    return explainer


def shap_summary(
    model: Any,
    X_sample: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> None:
    """Generate and optionally save a SHAP beeswarm summary plot.

    Parameters
    ----------
    model:
        Fitted estimator.
    X_sample:
        Sample of features to compute SHAP values for.
    output_path:
        If given, save the figure to this path.
    """
    import shap  # type: ignore

    explainer = get_shap_explainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # For multi-class output, take the positive-class values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, show=False)
    if output_path:
        ensure_dir(Path(output_path).parent)
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info("SHAP summary plot saved → %s", output_path)
    plt.close()


def shap_single_prediction(
    model: Any,
    X_row: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> list[dict]:
    """Compute SHAP values for a single prediction and optionally save a waterfall plot.

    Parameters
    ----------
    model:
        Fitted estimator.
    X_row:
        Single-row DataFrame (the customer to explain).
    output_path:
        If given, save the waterfall figure to this path.

    Returns
    -------
    list[dict]
        Top 5 features by |SHAP value|.
        Each dict has keys ``"feature"`` and ``"shap_value"``.
    """
    import shap  # type: ignore

    explainer = get_shap_explainer(model, X_row)
    shap_values = explainer.shap_values(X_row)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    feature_names = X_row.columns.tolist()
    values = shap_values[0] if shap_values.ndim > 1 else shap_values

    importance = sorted(
        zip(feature_names, values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top5 = [{"feature": f, "shap_value": float(v)} for f, v in importance[:5]]

    if output_path:
        fig, ax = plt.subplots(figsize=(8, 4))
        names = [d["feature"] for d in top5]
        vals = [d["shap_value"] for d in top5]
        colors = ["steelblue" if v >= 0 else "tomato" for v in vals]
        ax.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on prediction)")
        ax.set_title("Top 5 Feature Contributions (SHAP)")
        plt.tight_layout()
        save_figure(fig, output_path)
        plt.close()

    return top5


# ---------------------------------------------------------------------------
# Business insights
# ---------------------------------------------------------------------------

def generate_business_insights(
    feature_importance_df: pd.DataFrame,
    eda_summary: Optional[dict] = None,
) -> list[str]:
    """Generate plain-English business insight strings from feature importance.

    Parameters
    ----------
    feature_importance_df:
        Output of :func:`permutation_importance_table`.
    eda_summary:
        Optional dict from ``summarize_dataset`` for additional context.

    Returns
    -------
    list[str]
        List of human-readable insight strings.
    """
    insights: list[str] = []

    if feature_importance_df.empty:
        return ["No feature importance data available."]

    top_feature = feature_importance_df.iloc[0]["feature"]
    top_importance = feature_importance_df.iloc[0]["importance_mean"]

    insights.append(
        f"The most influential predictor is '{top_feature}' "
        f"(importance score: {top_importance:.4f})."
    )

    # Check for economic indicators
    economic = ["euribor3m", "emp.var.rate", "cons.price.idx",
                "cons.conf.idx", "nr.employed", "economic_stress_index"]
    econ_present = [
        r["feature"] for _, r in feature_importance_df.iterrows()
        if r["feature"] in economic
    ]
    if econ_present:
        insights.append(
            f"Economic indicators ({', '.join(econ_present[:3])}) appear in the top "
            "features — campaign timing relative to the economic cycle matters."
        )

    # Check for contact-related features
    contact = ["contact_is_cellular", "contact", "month_order", "month", "day_of_week"]
    contact_present = [
        r["feature"] for _, r in feature_importance_df.iterrows()
        if r["feature"] in contact
    ]
    if contact_present:
        insights.append(
            "Contact channel and timing features are important — "
            "cellular contacts and specific months show higher conversion rates."
        )

    # Check for previous campaign features
    prev = ["was_previously_contacted", "previous", "poutcome",
            "previous_contact_success_flag"]
    prev_present = [
        r["feature"] for _, r in feature_importance_df.iterrows()
        if r["feature"] in prev
    ]
    if prev_present:
        insights.append(
            "Previous campaign outcome is a strong signal — customers "
            "who subscribed in a prior campaign are far more likely to subscribe again."
        )

    insights.append(
        "⚠️  'duration' (last call length) is excluded from this model. "
        "It is only known after the call and would create data leakage in a "
        "live campaign setting."
    )

    return insights
