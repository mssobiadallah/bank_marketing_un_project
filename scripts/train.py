"""
train.py — Full training pipeline for the Bank Marketing ML project.

Trains all baseline models on both Feature Set A (with duration) and
Feature Set B (without duration), evaluates them, saves model artifacts,
and writes reports/model_metrics.csv.

Usage
-----
    python scripts/train.py
    make train
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    BENCHMARK_MODEL_FILE,
    MAIN_DATASET_PATH,
    MODELS_DIR,
    MODEL_METRICS_CSV,
    PREPROCESSING_PIPELINE_B_FILE,
    PREPROCESSING_PIPELINE_FILE,
    RANDOM_SEED,
    REALISTIC_MODEL_FILE,
    TARGET_COL,
)
from src.data_loader import load_dataset
from src.evaluation import evaluate_binary_classifier, save_metrics_csv
from src.features import add_features, encode_target, get_feature_lists
from src.modeling import get_baseline_models, save_model, train_model
from src.preprocessing import build_preprocessing_pipeline, save_pipeline, split_data
from src.utils import ensure_dir, get_logger, set_random_seed, timer

logger = get_logger(__name__)


@timer
def run_training() -> None:
    """Execute the full training pipeline."""
    set_random_seed(RANDOM_SEED)
    ensure_dir(MODELS_DIR)

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    logger.info("Loading main dataset: %s", MAIN_DATASET_PATH)
    df = load_dataset(MAIN_DATASET_PATH)
    df = encode_target(df, target=TARGET_COL)
    df = add_features(df, dataset_type="additional")
    logger.info("Dataset shape after feature engineering: %s", df.shape)

    # ------------------------------------------------------------------
    # 2. Feature Set A (with duration) — Benchmark
    # ------------------------------------------------------------------
    logger.info("=== Feature Set A — Benchmark Model (with duration) ===")
    feat_a = get_feature_lists(df, target=TARGET_COL, exclude_duration=False)
    X_train_a, X_test_a, y_train_a, y_test_a = split_data(
        df, target=TARGET_COL, test_size=0.2, random_state=RANDOM_SEED
    )
    X_train_a = X_train_a[feat_a["numeric"] + feat_a["categorical"]]
    X_test_a = X_test_a[feat_a["numeric"] + feat_a["categorical"]]

    preprocessor_a = build_preprocessing_pipeline(
        numeric_cols=feat_a["numeric"],
        categorical_cols=feat_a["categorical"],
    )

    models = get_baseline_models(random_state=RANDOM_SEED)
    all_results: list[dict] = []

    best_ap_a = 0.0
    best_pipeline_a = None

    for name, model in models.items():
        logger.info("Training [Set A] %s …", name)
        try:
            pipeline, cv_scores = train_model(
                model, X_train_a, y_train_a, preprocessor_a
            )
            metrics = evaluate_binary_classifier(pipeline, X_test_a, y_test_a)
            row = {
                "model_name": name,
                "feature_set": "set_a",
                **cv_scores,
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            }
            all_results.append(row)

            if metrics["average_precision"] > best_ap_a:
                best_ap_a = metrics["average_precision"]
                best_pipeline_a = pipeline

            logger.info(
                "  [Set A] %s — PR-AUC=%.4f, ROC-AUC=%.4f",
                name, metrics["average_precision"], metrics["roc_auc"],
            )
        except Exception as exc:
            logger.warning("Failed to train %s on Set A: %s", name, exc)

    if best_pipeline_a is not None:
        save_model(best_pipeline_a, MODELS_DIR / BENCHMARK_MODEL_FILE)
        save_pipeline(preprocessor_a, MODELS_DIR / PREPROCESSING_PIPELINE_FILE)
        logger.info("Benchmark model saved.")

    # ------------------------------------------------------------------
    # 3. Feature Set B (without duration) — Realistic Business Model
    # ------------------------------------------------------------------
    logger.info("=== Feature Set B — Realistic Business Model (no duration) ===")
    feat_b = get_feature_lists(df, target=TARGET_COL, exclude_duration=True)
    X_train_b, X_test_b, y_train_b, y_test_b = split_data(
        df, target=TARGET_COL, test_size=0.2, random_state=RANDOM_SEED
    )
    X_train_b = X_train_b[feat_b["numeric"] + feat_b["categorical"]]
    X_test_b = X_test_b[feat_b["numeric"] + feat_b["categorical"]]

    preprocessor_b = build_preprocessing_pipeline(
        numeric_cols=feat_b["numeric"],
        categorical_cols=feat_b["categorical"],
    )

    best_ap_b = 0.0
    best_pipeline_b = None
    best_model_name_b = ""

    models_b = get_baseline_models(random_state=RANDOM_SEED)

    for name, model in models_b.items():
        logger.info("Training [Set B] %s …", name)
        try:
            pipeline, cv_scores = train_model(
                model, X_train_b, y_train_b, preprocessor_b
            )
            metrics = evaluate_binary_classifier(pipeline, X_test_b, y_test_b)
            row = {
                "model_name": name,
                "feature_set": "set_b",
                **cv_scores,
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            }
            all_results.append(row)

            if metrics["average_precision"] > best_ap_b:
                best_ap_b = metrics["average_precision"]
                best_pipeline_b = pipeline
                best_model_name_b = name

            logger.info(
                "  [Set B] %s — PR-AUC=%.4f, ROC-AUC=%.4f",
                name, metrics["average_precision"], metrics["roc_auc"],
            )
        except Exception as exc:
            logger.warning("Failed to train %s on Set B: %s", name, exc)

    if best_pipeline_b is not None:
        save_model(best_pipeline_b, MODELS_DIR / REALISTIC_MODEL_FILE)
        save_pipeline(preprocessor_b, MODELS_DIR / PREPROCESSING_PIPELINE_B_FILE)
        logger.info(
            "Realistic Business Model saved: %s (PR-AUC=%.4f)",
            best_model_name_b, best_ap_b,
        )

    # ------------------------------------------------------------------
    # 4. Save metrics CSV
    # ------------------------------------------------------------------
    if all_results:
        save_metrics_csv(all_results, MODEL_METRICS_CSV)
        logger.info("Metrics saved → %s (%d rows)", MODEL_METRICS_CSV, len(all_results))

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    logger.info("=== Training Complete ===")
    logger.info("Best Benchmark Model PR-AUC (Set A):  %.4f", best_ap_a)
    logger.info("Best Realistic Model PR-AUC (Set B):  %.4f", best_ap_b)
    logger.info("Model artifacts in:  %s", MODELS_DIR)
    logger.info("Metrics CSV:         %s", MODEL_METRICS_CSV)


if __name__ == "__main__":
    run_training()
