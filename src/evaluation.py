"""
evaluation.py — Model evaluation, metrics, and visualisation helpers.

All functions accept fitted sklearn estimators and return plain Python dicts,
DataFrames, or matplotlib figures. The primary metric is ``average_precision``
(PR-AUC) per research.md Decision 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure as MplFigure
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.utils import ensure_dir, get_logger, save_figure

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

def evaluate_binary_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Compute all evaluation metrics for a fitted binary classifier.

    Parameters
    ----------
    model:
        Fitted sklearn-compatible estimator (or Pipeline).
    X_test:
        Test features.
    y_test:
        True binary labels (0/1).
    threshold:
        Decision threshold for converting probabilities to class labels.
        Default ``0.5``.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``balanced_accuracy``, ``precision``, ``recall``,
        ``f1``, ``roc_auc``, ``average_precision``, ``log_loss``,
        ``confusion_matrix`` (2×2 list).
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "log_loss": float(log_loss(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Optional[str | Path] = None,
) -> MplFigure:
    """Plot a confusion matrix.

    Parameters
    ----------
    y_true:
        True binary labels.
    y_pred:
        Predicted binary labels.
    output_path:
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_roc_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    model_name: str = "",
    output_path: Optional[str | Path] = None,
) -> MplFigure:
    """Plot a ROC curve.

    Parameters
    ----------
    y_true:
        True binary labels.
    y_proba:
        Predicted probabilities for the positive class.
    model_name:
        Label shown in the plot title / legend.
    output_path:
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_true, y_proba, name=model_name or "Model", ax=ax
    )
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_pr_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    model_name: str = "",
    output_path: Optional[str | Path] = None,
) -> MplFigure:
    """Plot a Precision-Recall curve.

    Parameters
    ----------
    y_true:
        True binary labels.
    y_proba:
        Predicted probabilities for the positive class.
    model_name:
        Label shown in the plot title / legend.
    output_path:
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_true, y_proba, name=model_name or "Model", ax=ax
    )
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def save_metrics_csv(results: list[dict], path: str | Path) -> Path:
    """Append metric rows to a CSV file.

    If the file already exists, new rows are appended (no header re-written).
    If it does not exist, it is created with a header.

    Parameters
    ----------
    results:
        List of metric dicts (e.g. from :func:`evaluate_binary_classifier`
        augmented with model_name, feature_set, etc.).
    path:
        CSV file path.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    p = Path(path)
    ensure_dir(p.parent)
    new_df = pd.DataFrame(results)
    if p.exists():
        existing = pd.read_csv(p)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(p, index=False)
    logger.info("Saved metrics → %s (%d rows)", p, len(combined))
    return p


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "average_precision",
    output_path: Optional[str | Path] = None,
) -> MplFigure:
    """Horizontal bar chart comparing models by a chosen metric.

    Parameters
    ----------
    results_df:
        DataFrame with at least ``model_name`` and *metric* columns.
    metric:
        Column name of the metric to plot.  Default ``"average_precision"``.
    output_path:
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = results_df.sort_values(metric, ascending=True).copy()
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.5)))
    ax.barh(df["model_name"], df[metric], color="steelblue")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}")
    ax.axvline(x=df[metric].max(), color="red", linestyle="--", alpha=0.5, label="Best")
    ax.legend()
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig
