# Implementation Plan: Bank Marketing ML Project

**Branch**: `001-bank-marketing-ml` | **Date**: 2026-05-16 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-bank-marketing-ml/spec.md`

## Summary

Build a professional end-to-end machine learning project that predicts bank term deposit subscription
using the `bank-additional-full.csv` dataset (41,188 rows, 21 features). The project delivers two
classical ML model tracks (Benchmark with `duration`; Realistic Business Model without `duration`),
full EDA, hypothesis testing, AutoML comparison via PyCaret, SHAP explainability, a 7-page
Streamlit app, pytest test suite, and Docker/Makefile deployment readiness.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**:
- Data: pandas >= 2.0, numpy >= 1.26
- ML: scikit-learn >= 1.4, imbalanced-learn >= 0.12
- Boosting (optional): xgboost >= 2.0, lightgbm >= 4.0, catboost >= 1.2
- AutoML (optional): pycaret >= 3.3
- Stats: scipy >= 1.12, statsmodels >= 0.14
- Explainability (optional): shap >= 0.45
- Viz: matplotlib >= 3.8, seaborn >= 0.13, plotly >= 5.20
- App: streamlit >= 1.33
- Persistence: joblib >= 1.3
- Testing: pytest >= 8.0

**Storage**: File-based only
- `data/raw/` — original CSVs (semicolon-separated)
- `data/processed/` — train.csv, test.csv, feature_metadata.json
- `models/` — .joblib files for pipelines and models
- `reports/` — markdown reports, figures/, automl_results.csv, model_metrics.csv

**Testing**: pytest with unit tests in `tests/`

**Target Platform**: macOS / Linux local; Streamlit Community Cloud for deployment

**Project Type**: ML pipeline + multi-page Streamlit web application

**Performance Goals**:
- Realistic Business Model: ROC-AUC >= 0.75, PR-AUC >= 0.40 on test set
- Top-decile lift >= 2.5x baseline
- Batch prediction of 1,000 rows completes in < 60 seconds
- All pytest tests pass in < 120 seconds

**Constraints**:
- Classical ML only — no deep learning
- `duration` excluded from Realistic Business Model and all app predictions
- Random seed = 42 everywhere
- SMOTE applied inside CV folds only
- Optional libraries must not break core modules if absent

**Scale/Scope**:
- 41,188 rows x 21 columns (main dataset)
- 9 Jupyter notebooks
- 14 src/ modules
- 7-page Streamlit app
- 5 pytest test files
- 3 deployment scripts

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Classical ML Only | PASS | Only scikit-learn, XGBoost, LightGBM, CatBoost used. No neural nets. |
| II. Two-Track Modelling | PASS | Feature Set A (with duration) and Feature Set B (without duration) explicitly defined in FR-003, SC-010, and all modeling phases. |
| III. Main Dataset bank-additional-full.csv | PASS | All final EDA, modeling, and app use this dataset per NFR-003. |
| IV. Reproducibility | PASS | random_state=42 used in all splits, CV, and model training per NFR-005. |
| V. Module-First Architecture | PASS | All logic in src/; notebooks call src/ only per NFR-004. |
| VI. Business-First Explainability | PASS | SHAP + permutation importance required per FR-015. |
| VII. Graceful Degradation | PASS | Optional libraries guarded per NFR requirements and FR-013 notes. |

**Gate result**: ALL PASS — no violations to justify.

## Project Structure

### Documentation (this feature)

```
specs/001-bank-marketing-ml/
├── plan.md                        # This file
├── research.md                    # Phase 0 output
├── data-model.md                  # Phase 1 output
├── quickstart.md                  # Phase 1 output
├── contracts/
│   ├── single-prediction.md       # Phase 1 output
│   ├── batch-prediction.md        # Phase 1 output
│   └── inference-schema.md        # Phase 1 output
├── checklists/
│   └── requirements.md
└── tasks.md                       # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```
bank-marketing-ml-project/
├── data/
│   ├── raw/                       # Original CSV files (semicolon-separated)
│   └── processed/                 # train.csv, test.csv, feature_metadata.json
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda_univariate.ipynb
│   ├── 03_eda_bivariate_multivariate.ipynb
│   ├── 04_hypothesis_testing.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_modeling_baselines.ipynb
│   ├── 07_automl_experiments.ipynb
│   ├── 08_model_selection_explainability.ipynb
│   └── 09_business_recommendations.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                  # Paths, constants, random seed
│   ├── data_loader.py             # load_dataset, summarize_dataset, compare_datasets
│   ├── preprocessing.py           # build_preprocessing_pipeline, target encoder
│   ├── features.py                # add_features, get_feature_lists
│   ├── eda.py                     # Univariate + bivariate EDA functions
│   ├── multivariate_analysis.py   # Correlation, VIF, Cramer's V, segment analysis
│   ├── hypothesis_tests.py        # Chi-square, Mann-Whitney, run_all_hypothesis_tests
│   ├── modeling.py                # get_baseline_models, train_model
│   ├── evaluation.py              # evaluate_binary_classifier, plots
│   ├── model_selection.py         # compare_models, select_best_model, threshold tuning
│   ├── explainability.py          # SHAP, permutation importance
│   ├── inference.py               # predict_single, predict_batch, schema validation
│   └── utils.py                   # Shared helpers
├── models/
│   ├── benchmark_model_with_duration.joblib
│   ├── realistic_model_without_duration.joblib
│   ├── preprocessing_pipeline.joblib
│   └── model_card.md
├── reports/
│   ├── figures/
│   ├── eda_report.md
│   ├── hypothesis_testing_report.md
│   ├── model_comparison_report.md
│   ├── automl_results.csv
│   ├── model_metrics.csv
│   └── final_business_report.md
├── app/
│   ├── streamlit_app.py           # Entry point; Project Overview page
│   └── pages/
│       ├── 1_EDA_Dashboard.py
│       ├── 2_Hypothesis_Testing.py
│       ├── 3_Model_Performance.py
│       ├── 4_Predict_New_Client.py
│       ├── 5_Batch_Prediction.py
│       └── 6_Business_Recommendations.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_inference.py
│   └── test_streamlit_inputs.py
├── scripts/
│   ├── train.py
│   ├── generate_reports.py
│   └── predict_batch.py
├── .streamlit/config.toml
├── .gitignore
├── README.md
├── requirements.txt
├── Dockerfile
└── Makefile
```

**Structure Decision**: Single-project layout with clean separation of concerns across
`src/` (library), `notebooks/` (exploration), `app/` (presentation), `scripts/`
(orchestration), and `tests/` (validation).

## Implementation Milestones

### Milestone 1 — Foundation (Phase 0)
Files: `src/config.py`, `src/utils.py`, `src/data_loader.py`, `tests/test_data_loader.py`,
`requirements.txt`, `.gitignore`, `README.md` (skeleton), `Makefile`

Acceptance: `pytest tests/test_data_loader.py` passes; all four CSVs load correctly.

### Milestone 2 — EDA (Phase 2-4)
Files: `src/eda.py`, `src/multivariate_analysis.py`, `notebooks/01–03`, `reports/eda_report.md`

Acceptance: All figures saved to `reports/figures/`; univariate, bivariate, multivariate
summaries returned as DataFrames.

### Milestone 3 — Hypothesis Testing (Phase 5)
Files: `src/hypothesis_tests.py`, `notebooks/04_hypothesis_testing.ipynb`,
`reports/hypothesis_testing_report.md`

Acceptance: `run_all_hypothesis_tests()` returns all 7 hypotheses with correct p-values,
reject_null flags, and effect sizes.

### Milestone 4 — Preprocessing & Features (Phase 6)
Files: `src/preprocessing.py`, `src/features.py`, `notebooks/05_feature_engineering.ipynb`,
`tests/test_preprocessing.py`, `tests/test_features.py`, `data/processed/`

Acceptance: Pipeline builds without error; feature engineering adds all 9 engineered features;
`data/processed/train.csv` and `test.csv` written.

### Milestone 5 — Baseline Modeling (Phase 7)
Files: `src/modeling.py`, `src/evaluation.py`, `notebooks/06_modeling_baselines.ipynb`,
`reports/model_metrics.csv`

Acceptance: All baseline models train and evaluate on both feature sets; metrics CSV written;
model .joblib files saved.

### Milestone 6 — AutoML (Phase 8)
Files: `notebooks/07_automl_experiments.ipynb`, `reports/automl_results.csv`

Acceptance: PyCaret compare_models() runs for both feature sets; results CSV has >= 8
model entries per set.

### Milestone 7 — Model Selection & Explainability (Phases 9-10)
Files: `src/model_selection.py`, `src/explainability.py`,
`notebooks/08_model_selection_explainability.ipynb`, `models/realistic_model_without_duration.joblib`,
`models/benchmark_model_with_duration.joblib`, `models/model_card.md`

Acceptance: Best model selected by average_precision; threshold tuned; SHAP summary and
waterfall plots saved; model_card.md filled.

### Milestone 8 — Inference & Streamlit App (Phases 11-12)
Files: `src/inference.py`, `tests/test_inference.py`, `app/streamlit_app.py`, `app/pages/*`,
`tests/test_streamlit_inputs.py`, `.streamlit/config.toml`

Acceptance: Single prediction form works; batch upload and download works; duration warning
visible; all 7 pages render without error.

### Milestone 9 — Deployment & Final Reports (Phase 12)
Files: `Dockerfile`, `scripts/train.py`, `scripts/generate_reports.py`,
`scripts/predict_batch.py`, `reports/final_business_report.md`, `README.md` (final),
`notebooks/09_business_recommendations.ipynb`

Acceptance: `docker build` succeeds; `streamlit run app/streamlit_app.py` runs; README
contains all required sections.
