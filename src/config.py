"""
config.py — Project-wide constants, paths, and feature column definitions.

All modules import from here. Never hard-code paths or column names elsewhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Dataset filenames
# ---------------------------------------------------------------------------
MAIN_DATASET = "bank-additional-full.csv"
MAIN_DATASET_PATH = RAW_DIR / MAIN_DATASET

ALL_DATASETS: dict[str, Path] = {
    "bank-additional-full": RAW_DIR / "bank-additional-full.csv",
    "bank-additional": RAW_DIR / "bank-additional.csv",
    "bank-full": RAW_DIR / "bank-full.csv",
    "bank": RAW_DIR / "bank.csv",
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
TARGET_COL: str = "y"

# ---------------------------------------------------------------------------
# Raw column groups (bank-additional* schema — 21 columns)
# ---------------------------------------------------------------------------
# Client attributes
CLIENT_COLS: list[str] = [
    "age", "job", "marital", "education", "default", "housing", "loan",
]

# Last contact of current campaign
CONTACT_COLS: list[str] = [
    "contact", "month", "day_of_week", "duration",
]

# Other campaign attributes
CAMPAIGN_COLS: list[str] = [
    "campaign", "pdays", "previous", "poutcome",
]

# Social and economic context attributes
ECONOMIC_COLS: list[str] = [
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]

# All numeric columns in the raw dataset
NUMERIC_COLS: list[str] = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]

# All categorical columns in the raw dataset
CATEGORICAL_COLS: list[str] = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]

# ---------------------------------------------------------------------------
# Engineered feature names (added by src/features.py)
# ---------------------------------------------------------------------------
ENGINEERED_FEATURE_NAMES: list[str] = [
    "was_previously_contacted",
    "campaign_intensity_group",
    "age_group",
    "economic_stress_index",
    "has_any_loan",
    "month_order",
    "previous_contact_success_flag",
    "contact_is_cellular",
    "client_financial_pressure_flag",
]

# Numeric engineered features (for ColumnTransformer passthrough)
ENGINEERED_NUMERIC: list[str] = [
    "was_previously_contacted",
    "economic_stress_index",
    "month_order",
    "previous_contact_success_flag",
    "contact_is_cellular",
    "client_financial_pressure_flag",
    "has_any_loan",
]

# Categorical engineered features
ENGINEERED_CATEGORICAL: list[str] = [
    "campaign_intensity_group",
    "age_group",
]

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
# Feature Set A: includes duration (benchmark / upper-bound model only)
FEATURE_SET_A_COLS: list[str] = (
    CLIENT_COLS
    + CONTACT_COLS          # includes duration
    + CAMPAIGN_COLS
    + ECONOMIC_COLS
    + ENGINEERED_FEATURE_NAMES
)

# Feature Set B: excludes duration (Realistic Business Model)
FEATURE_SET_B_COLS: list[str] = [
    col for col in FEATURE_SET_A_COLS if col != "duration"
]

# Numeric columns for Feature Set B (no duration)
NUMERIC_COLS_SET_B: list[str] = [col for col in NUMERIC_COLS if col != "duration"]

# ---------------------------------------------------------------------------
# Model artifact filenames
# ---------------------------------------------------------------------------
BENCHMARK_MODEL_FILE = "benchmark_model_with_duration.joblib"
REALISTIC_MODEL_FILE = "realistic_model_without_duration.joblib"
PREPROCESSING_PIPELINE_FILE = "preprocessing_pipeline.joblib"
PREPROCESSING_PIPELINE_B_FILE = "preprocessing_pipeline_set_b.joblib"

# Advanced pipeline champion (FLAML-lgbm, notebook 09)
ADVANCED_CHAMPION_MODEL_FILE = "champion_model_advanced_lgbm.joblib"

# ---------------------------------------------------------------------------
# Optimal decision threshold (tuned in notebook 09 — best Subscribe F1)
# At 0.27: Accuracy=88.7%, Subscribe Recall=56%, Subscribe F1=0.53
# Default 0.50 gave: Accuracy=90.2%, Subscribe Recall=24%, Subscribe F1=0.36
# ---------------------------------------------------------------------------
OPTIMAL_THRESHOLD: float = 0.27

# ---------------------------------------------------------------------------
# Report filenames
# ---------------------------------------------------------------------------
MODEL_METRICS_CSV = REPORTS_DIR / "model_metrics.csv"
HYPOTHESIS_REPORT_MD = REPORTS_DIR / "hypothesis_testing_report.md"
AUTOML_RESULTS_CSV = REPORTS_DIR / "automl_results.csv"
EDA_REPORT_MD = REPORTS_DIR / "eda_report.md"
MODEL_COMPARISON_REPORT_MD = REPORTS_DIR / "model_comparison_report.md"
FINAL_BUSINESS_REPORT_MD = REPORTS_DIR / "final_business_report.md"

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Data separator
# ---------------------------------------------------------------------------
CSV_SEP: str = ";"
