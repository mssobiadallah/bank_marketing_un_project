# Bank Marketing ML Project - Constitution

## Core Principles

### I. Classical ML Only (NON-NEGOTIABLE)
Only classical ML models permitted. Deep learning is forbidden.
Allowed: Logistic Regression, Decision Trees, Random Forests, Extra Trees,
Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost, SVM, KNN, DummyClassifier.

### II. Two-Track Modelling (NON-NEGOTIABLE)
Every experiment runs twice: Benchmark Model (with duration) and Realistic Business Model
(without duration). The app and inference always use the Realistic Business Model.

### III. Main Dataset is bank-additional-full.csv
All final work uses bank-additional-full.csv (41,188 rows, 21 columns).

### IV. Reproducibility First
Random seed 42 everywhere. Reproducible via: python scripts/train.py

### V. Module-First Architecture
All reusable logic lives in src/. Notebooks call src/ functions only.
Every src/ module must have type hints and docstrings.

### VI. Business-First Explainability
Feature importance required for every model. SHAP plots required for final model.

### VII. Graceful Degradation
Optional libraries (PyCaret, XGBoost, LightGBM, CatBoost, SHAP) must not break core modules.

## Model Selection Policy

Primary metric: average_precision (PR-AUC). Secondary: roc_auc.
Threshold tuned after model selection.
Target: ROC-AUC >= 0.75, PR-AUC >= 0.40 on test set.

## Data Policy

- unknown in categorical columns kept as valid category.
- pdays=999 converted to was_previously_contacted=False.
- SMOTE inside CV folds only.
- Class weights preferred over SMOTE for baselines.

## Development Workflow

1. Branch: 001-bank-marketing-ml
2. Commit per milestone.
3. Tests pass before milestone is complete.

## Governance

This constitution supersedes all other practices.

**Version**: 1.0.0 | **Ratified**: 2026-05-16 | **Last Amended**: 2026-05-16
