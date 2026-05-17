"""
utils.py — Shared utility helpers used across all src/ modules.

Includes: path management, figure/dataframe saving, reproducibility,
timing decorator, and centralised logging.
"""

from __future__ import annotations

import logging
import os
import random
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and all parents) if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    Path
        The resolved ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_figure(fig: Any, path: str | Path, dpi: int = 150) -> Path:
    """Save a matplotlib or plotly figure to *path*.

    Creates the parent directory automatically.

    Parameters
    ----------
    fig:
        A ``matplotlib.figure.Figure`` or ``plotly.graph_objects.Figure``.
    path:
        Destination file path (e.g. ``reports/figures/target_dist.png``).
    dpi:
        Resolution for raster formats (ignored for plotly).

    Returns
    -------
    Path
        Resolved destination path.
    """
    p = Path(path)
    ensure_dir(p.parent)

    # Detect figure type without a hard import of matplotlib at module level
    fig_type = type(fig).__module__
    if fig_type.startswith("matplotlib"):
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    elif fig_type.startswith("plotly"):
        fig.write_image(str(p))
    else:
        raise TypeError(f"Unsupported figure type: {type(fig)}")

    return p


def save_dataframe(df: Any, path: str | Path, index: bool = False) -> Path:
    """Save a pandas DataFrame to *path* (CSV).

    Creates the parent directory automatically.

    Parameters
    ----------
    df:
        pandas DataFrame to save.
    path:
        Destination CSV path.
    index:
        Whether to write the row index (default ``False``).

    Returns
    -------
    Path
        Resolved destination path.
    """
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=index)
    return p


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and (if available) TensorFlow/Torch.

    Parameters
    ----------
    seed:
        Integer seed value (default ``42``).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional: tensorflow
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
    except ImportError:
        pass

    # Optional: torch
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timer(func: Callable) -> Callable:
    """Decorator that logs the wall-clock execution time of *func*.

    Usage::

        @timer
        def my_function():
            ...

    Parameters
    ----------
    func:
        The callable to wrap.

    Returns
    -------
    Callable
        Wrapped function.
    """
    logger = get_logger(func.__module__ or __name__)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("⏱  %s completed in %.2f s", func.__qualname__, elapsed)
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with a consistent format.

    If the logger already has handlers (e.g. called multiple times),
    new handlers are not added to avoid duplicate log lines.

    Parameters
    ----------
    name:
        Logger name — typically ``__name__`` of the calling module.
    level:
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger
