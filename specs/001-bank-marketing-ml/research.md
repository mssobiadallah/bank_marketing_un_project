# Research: Bank Marketing ML Project

**Phase**: 0 — Outline & Research
**Date**: 2026-05-16
**Feature**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

All technical context in `plan.md` was fully resolvable from the spec and the existing
`bank_marketing_ml_graduation_project_plan.md`. No NEEDS CLARIFICATION markers remain.
This file documents the key decisions made and alternatives considered.

---

## Decision 1 — Primary Evaluation Metric

**Decision**: Use `average_precision` (PR-AUC) as the primary model selection metric.

**Rationale**: The dataset is severely class-imbalanced (~11.3% positive class in
`bank-additional-full.csv`). ROC-AUC can be misleadingly high for imbalanced datasets
because it accounts for true negatives. PR-AUC is sensitive to how well the model
retrieves the positive class and directly maps to the business goal of ranking customers
by subscription likelihood.

**Alternatives considered**:
- `roc_auc` — used as secondary metric; misleading as sole primary for imbalanced data.
- `f1` — threshold-dependent; requires a threshold choice before model selection.
- `recall` — ignores precision; would favour a model that predicts "yes" for everyone.

---

## Decision 2 — AutoML Framework

**Decision**: Use PyCaret Classification as the AutoML tool.

**Rationale**: PyCaret wraps scikit-learn, XGBoost, LightGBM, and CatBoost into a unified
`compare_models()` call, handles preprocessing, CV scoring, and result formatting automatically.
For a graduation project, it provides the clearest comparison table and the most professional
output with the least boilerplate. It is guarded as optional so its absence does not break
other modules.

**Alternatives considered**:
- FLAML — lighter, faster, but produces less readable output for a graduation presentation.
- AutoGluon Tabular — powerful but heavyweight; installation conflicts common on student machines.
- Manual grid search — not true AutoML; misses the point of the AutoML phase.

---

## Decision 3 — Class Imbalance Strategy

**Decision**: Use `class_weight='balanced'` as the default imbalance strategy for all
baseline models. SMOTE is available as an optional pipeline step, applied strictly inside
cross-validation folds using `imblearn.pipeline.Pipeline`.

**Rationale**: Class weights are simpler, require no additional sampling, are natively
supported by scikit-learn classifiers, and are less prone to overfitting. SMOTE is included
for completeness and comparison, but is not the default because it must only be applied inside
CV folds to avoid data leakage — a constraint easy to violate accidentally.

**Alternatives considered**:
- Oversampling (SMOTE only) — riskier; leakage if applied before train/test split.
- Undersampling — discards real data; only useful for very large majority class.
- Threshold tuning only — complementary, not a replacement for training-time rebalancing.

---

## Decision 4 — SHAP Explainer Type

**Decision**: Use `shap.TreeExplainer` for tree-based models (Random Forest, Gradient
Boosting, XGBoost, LightGBM, CatBoost); `shap.LinearExplainer` for Logistic Regression;
`shap.KernelExplainer` as a fallback for any other model type.

**Rationale**: `TreeExplainer` is exact (not an approximation) and fast for tree-based
models. `LinearExplainer` is exact for linear models. `KernelExplainer` is a model-agnostic
fallback but is slow — it should only be used when the model type is not tree or linear.

**Alternatives considered**:
- `KernelExplainer` for all models — model-agnostic but too slow for 41k rows.
- `shap.Explainer` auto-dispatch — convenient but less controllable for edge cases.

---

## Decision 5 — Classification Threshold Tuning Strategy

**Decision**: Support three strategies in `model_selection.tune_classification_threshold()`:
1. `max_f1` — threshold that maximises F1-score on the validation set (default).
2. `target_recall` — smallest threshold that achieves a user-specified recall floor.
3. `target_precision` — largest threshold that achieves a user-specified precision floor.

**Rationale**: The business use case is flexible. A marketing manager may want to maximise
F1 (balanced), or may have a fixed call-centre capacity (target recall = top-N%), or may
want to avoid too many wasted calls (target precision). Offering all three makes the project
genuinely useful.

**Alternatives considered**:
- Fixed 0.5 threshold — never appropriate for imbalanced data.
- Youden's J statistic — equivalent to max (TPR - FPR); less directly interpretable.

---

## Decision 6 — Encoding Strategy for Categorical Variables

**Decision**: Use `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for all
nominal categorical features. Use ordinal encoding only for `education` (if used as ordinal)
and `month` (calendar order). Apply inside the `ColumnTransformer` preprocessing pipeline.

**Rationale**: OHE is appropriate for nominal categoricals and is compatible with all
scikit-learn classifiers. `handle_unknown='ignore'` is essential for inference robustness
when the app receives unseen categories. Sparse output is disabled for easier downstream
handling.

**Alternatives considered**:
- TargetEncoder — introduces target leakage risk if not properly cross-fitted.
- OrdinalEncoder for all — loses information for nominal features in linear models.
- Leaving as strings — only works for tree models that natively support categoricals
  (LightGBM, CatBoost); breaks Logistic Regression and SVM.

---

## Decision 7 — `pdays` Handling

**Decision**: For `bank-additional-full.csv`, `pdays = 999` means "not previously contacted".
Create a binary feature `was_previously_contacted = (pdays != 999)`. Replace the original
`pdays` with the clipped value (set `pdays` to NaN or 0 after flag creation) or keep
`pdays` as-is with the flag as an additional feature. The flag must always be created.

**Rationale**: `999` is a sentinel value, not a real contact-day count. Without the flag,
models will incorrectly treat 999 as a real number of days. The original `bank-full.csv`
uses `-1` for the same meaning — handled identically with a different sentinel check.

**Alternatives considered**:
- Drop `pdays` entirely — loses information about previous contact timing for contacted clients.
- Impute 999 with median — misleading; median of contacted clients is not meaningful
  for non-contacted clients.

---

## Decision 8 — Streamlit App Model Loading

**Decision**: Load the Realistic Business Model (without `duration`) from
`models/realistic_model_without_duration.joblib` using `st.cache_resource`. Load the
preprocessing pipeline from `models/preprocessing_pipeline.joblib` separately. If any
model file is missing, display a clear `st.error()` message and `st.stop()` — do not crash.

**Rationale**: `st.cache_resource` prevents the model from being re-loaded on every
interaction. Separate loading of the pipeline allows it to be inspected and reused in
the EDA dashboard for feature name extraction. `st.stop()` on missing files is the correct
Streamlit pattern for graceful degradation.

**Alternatives considered**:
- Load both model tracks and let user switch — adds UI complexity; business page should
  only show the realistic model.
- Embed pipeline in model joblib — harder to inspect; prevents reuse in EDA pages.

---

## Resolved: No NEEDS CLARIFICATION Items

All items in the Technical Context were resolvable from:
1. `bank_marketing_ml_graduation_project_plan.md` (project source of truth)
2. `specs/001-bank-marketing-ml/spec.md` (specification)
3. `specs/001-bank-marketing-ml/plan.md` (implementation plan)
4. `.specify/memory/constitution.md` (project constitution)

Research phase complete. Proceeding to Phase 1 design.
