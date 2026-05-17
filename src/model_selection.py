"""
model_selection.py — Model comparison, best-model selection, threshold tuning,
lift analysis, and related reporting helpers.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure as MplFigure
from sklearn.metrics import f1_score, precision_score, recall_score

from src.utils import get_logger, save_figure

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Comparison & selection
# ---------------------------------------------------------------------------

def compare_model_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Sort models by ``average_precision`` descending and add a rank column.

    Parameters
    ----------
    results_df:
        DataFrame loaded from ``reports/model_metrics.csv``.

    Returns
    -------
    pd.DataFrame
        Sorted copy with a ``rank`` column (1 = best).
    """
    df = results_df.copy().sort_values("average_precision", ascending=False)
    df = df.reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def select_best_model(
    results_df: pd.DataFrame,
    primary_metric: str = "average_precision",
    feature_set: str = "set_b",
) -> str:
    """Return the name of the best model for a given feature set.

    Parameters
    ----------
    results_df:
        DataFrame loaded from ``reports/model_metrics.csv``.
    primary_metric:
        Metric column to maximise.  Default ``"average_precision"``.
    feature_set:
        Filter to ``"set_a"`` or ``"set_b"`` rows.  Default ``"set_b"``
        (Realistic Business Model).

    Returns
    -------
    str
        ``model_name`` of the row with the highest *primary_metric*.
    """
    subset = results_df[results_df["feature_set"] == feature_set]
    if subset.empty:
        raise ValueError(
            f"No rows found for feature_set='{feature_set}' in results_df."
        )
    best_idx = subset[primary_metric].idxmax()
    best_name = subset.loc[best_idx, "model_name"]
    best_score = subset.loc[best_idx, primary_metric]
    logger.info(
        "Best model [%s] by %s: %s (%.4f)",
        feature_set, primary_metric, best_name, best_score,
    )
    return str(best_name)


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_classification_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    strategy: str = "max_f1",
    target_value: Optional[float] = None,
) -> tuple[float, dict]:
    """Find an optimal probability threshold using the given strategy.

    Strategies
    ----------
    - ``"max_f1"``         — maximise F1 score
    - ``"target_recall"``  — find the lowest threshold that achieves
      ``recall >= target_value``
    - ``"target_precision"`` — find the lowest threshold that achieves
      ``precision >= target_value``

    Parameters
    ----------
    y_true:
        True binary labels.
    y_proba:
        Predicted probabilities for the positive class.
    strategy:
        One of ``"max_f1"``, ``"target_recall"``, ``"target_precision"``.
    target_value:
        Required for ``"target_recall"`` and ``"target_precision"``
        strategies.  Ignored for ``"max_f1"``.

    Returns
    -------
    tuple[float, dict]
        ``(optimal_threshold, metrics_at_threshold)``

        Metrics dict keys: ``threshold``, ``precision``, ``recall``, ``f1``.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_threshold = 0.5
    best_score = -1.0

    y_true_arr = np.array(y_true)

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        if strategy == "max_f1":
            score = f1_score(y_true_arr, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = t

        elif strategy == "target_recall":
            if target_value is None:
                raise ValueError("target_value required for strategy='target_recall'")
            rec = recall_score(y_true_arr, y_pred, zero_division=0)
            if rec >= target_value:
                prec = precision_score(y_true_arr, y_pred, zero_division=0)
                f1 = f1_score(y_true_arr, y_pred, zero_division=0)
                if f1 > best_score:
                    best_score = f1
                    best_threshold = t

        elif strategy == "target_precision":
            if target_value is None:
                raise ValueError("target_value required for strategy='target_precision'")
            prec = precision_score(y_true_arr, y_pred, zero_division=0)
            if prec >= target_value:
                f1 = f1_score(y_true_arr, y_pred, zero_division=0)
                if f1 > best_score:
                    best_score = f1
                    best_threshold = t

        else:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                "Use 'max_f1', 'target_recall', or 'target_precision'."
            )

    metrics = evaluate_at_threshold(y_true, y_proba, best_threshold)
    logger.info(
        "Optimal threshold [%s]: %.4f → F1=%.4f, P=%.4f, R=%.4f",
        strategy, best_threshold, metrics["f1"], metrics["precision"], metrics["recall"],
    )
    return best_threshold, metrics


def evaluate_at_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """Compute precision, recall, and F1 at a given *threshold*.

    Parameters
    ----------
    y_true:
        True binary labels.
    y_proba:
        Predicted probabilities.
    threshold:
        Decision boundary.

    Returns
    -------
    dict
        Keys: ``threshold``, ``precision``, ``recall``, ``f1``.
    """
    y_pred = (np.array(y_proba) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Lift analysis
# ---------------------------------------------------------------------------

def create_lift_table(
    y_true: pd.Series,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a gains / lift table by decile.

    Parameters
    ----------
    y_true:
        True binary labels (0/1).
    y_proba:
        Predicted probabilities for the positive class.
    n_bins:
        Number of equal-sized bins (default 10 → deciles).

    Returns
    -------
    pd.DataFrame
        Columns: ``decile``, ``n_customers``, ``n_subscribers``,
        ``conversion_rate``, ``lift``.
    """
    df = pd.DataFrame({"y_true": np.array(y_true), "y_proba": np.array(y_proba)})
    df = df.sort_values("y_proba", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, q=n_bins, labels=range(1, n_bins + 1))

    baseline_rate = df["y_true"].mean()

    records = []
    for decile in range(1, n_bins + 1):
        mask = df["decile"] == decile
        n = int(mask.sum())
        pos = int(df.loc[mask, "y_true"].sum())
        rate = pos / n if n > 0 else 0.0
        lift = rate / baseline_rate if baseline_rate > 0 else 0.0
        records.append({
            "decile": decile,
            "n_customers": n,
            "n_subscribers": pos,
            "conversion_rate": round(rate, 4),
            "lift": round(lift, 4),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_lift_chart(
    lift_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> MplFigure:
    """Bar chart of lift per decile.

    Parameters
    ----------
    lift_df:
        Output of :func:`create_lift_table`.
    output_path:
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(lift_df["decile"].astype(str), lift_df["lift"], color="steelblue")
    ax.axhline(y=1.0, color="red", linestyle="--", label="Baseline lift = 1")
    ax.set_xlabel("Decile (1 = highest predicted probability)")
    ax.set_ylabel("Lift")
    ax.set_title("Lift Chart by Decile")
    ax.legend()
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_threshold_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    output_path: Optional[str] = None,
) -> MplFigure:
    """Plot F1, precision, and recall vs decision threshold.

    Parameters
    ----------
    y_true:
        True binary labels.
    y_proba:
        Predicted probabilities for the positive class.
    output_path:
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    f1s, precs, recs = [], [], []
    y_true_arr = np.array(y_true)

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_true_arr, y_pred, zero_division=0))
        precs.append(precision_score(y_true_arr, y_pred, zero_division=0))
        recs.append(recall_score(y_true_arr, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s, label="F1", color="blue")
    ax.plot(thresholds, precs, label="Precision", color="green")
    ax.plot(thresholds, recs, label="Recall", color="orange")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.legend()
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig
