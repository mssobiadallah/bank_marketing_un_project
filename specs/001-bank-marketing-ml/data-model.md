# Data Model: Bank Marketing ML Project

**Phase**: 1 — Design & Contracts
**Date**: 2026-05-16
**Feature**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

---

## Entity 1 — RawRecord

Represents one row loaded directly from a CSV dataset (semicolon-separated).

**Source**: `data/raw/bank-additional-full.csv` (primary), other three CSV variants.

| Field | Type | Values / Range | Notes |
|-------|------|----------------|-------|
| age | int | 17–98 | Client age |
| job | str | 12 categories + unknown | Occupation |
| marital | str | married / single / divorced / unknown | |
| education | str | 8 levels + unknown | |
| default | str | yes / no / unknown | Has credit in default? |
| housing | str | yes / no / unknown | Has housing loan? |
| loan | str | yes / no / unknown | Has personal loan? |
| contact | str | cellular / telephone | Contact type |
| month | str | jan–dec | Last contact month |
| day_of_week | str | mon–fri | Last contact day |
| duration | int | 0–4918 s | Last contact duration. **EXCLUDED from Realistic Model.** |
| campaign | int | 1–56 | Contacts in this campaign |
| pdays | int | 0–999 | Days since last contact (999 = not contacted) |
| previous | int | 0–7 | Contacts before this campaign |
| poutcome | str | success / failure / nonexistent | Previous campaign outcome |
| emp.var.rate | float | -3.4 to 1.4 | Employment variation rate |
| cons.price.idx | float | 92.2–94.8 | Consumer price index |
| cons.conf.idx | float | -50.8 to -26.9 | Consumer confidence index |
| euribor3m | float | 0.634–5.045 | Euribor 3-month rate |
| nr.employed | float | 4963.6–5228.1 | Number of employees (quarterly) |
| y | str | yes / no | **Target variable** |

**Validation rules**:
- Separator must be `;` (not `,`)
- `y` must contain only `yes` or `no`
- `duration` >= 0
- `pdays` in [0, 999]
- No fully-null rows

---

## Entity 2 — ProcessedRecord

A `RawRecord` after target encoding, feature engineering, and preprocessing pipeline
transformation. Ready for model training and inference.

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| y_binary | int | y | 1 = yes, 0 = no |
| was_previously_contacted | int | pdays | 1 if pdays != 999, else 0 |
| campaign_intensity_group | str | campaign | low / medium / high buckets |
| age_group | str | age | young / middle / senior |
| economic_stress_index | float | euribor3m + emp.var.rate | Composite indicator |
| has_any_loan | int | housing + loan | 1 if either loan = yes |
| month_order | int | month | Calendar order 1–12 |
| previous_contact_success_flag | int | poutcome | 1 if poutcome = success |
| contact_is_cellular | int | contact | 1 if cellular |
| client_financial_pressure_flag | int | default + housing + loan | Composite |
| [all OHE features] | float | categorical cols | OneHotEncoder output |
| [scaled numeric features] | float | numeric cols | StandardScaler output (optional) |

---

## Entity 3 — FeatureSet

Defines which columns are included in a model training or inference run.

| Field | Type | Description |
|-------|------|-------------|
| name | str | "feature_set_a_with_duration" or "feature_set_b_without_duration" |
| numeric_cols | list[str] | All numerical feature names |
| categorical_cols | list[str] | All categorical feature names |
| exclude_duration | bool | False for Set A; True for Set B |
| target_col | str | Always "y" |

---

## Entity 4 — ModelArtifact

A trained model plus its associated metadata, saved to `models/`.

| Field | Type | Description |
|-------|------|-------------|
| model_name | str | Classifier class name |
| feature_set | str | "set_a" or "set_b" |
| pipeline | sklearn.Pipeline | Preprocessor + classifier |
| threshold | float | Classification threshold (default 0.5, tuned after selection) |
| metrics | ModelMetrics | Evaluated metrics on test set |
| file_path | str | Path to .joblib file |

**State transitions**:
```
untrained → trained → evaluated → selected → threshold_tuned → saved
```

---

## Entity 5 — ModelMetrics

Evaluation results for one model on one feature set.

| Field | Type | Description |
|-------|------|-------------|
| model_name | str | Classifier name |
| feature_set | str | "set_a" or "set_b" |
| accuracy | float | Standard accuracy |
| balanced_accuracy | float | Balanced accuracy (accounts for imbalance) |
| precision | float | Precision for class 1 |
| recall | float | Recall for class 1 |
| f1 | float | F1-score for class 1 |
| roc_auc | float | ROC-AUC |
| average_precision | float | PR-AUC (primary metric) |
| log_loss | float | Log loss |
| confusion_matrix | list[list[int]] | [[TN, FP], [FN, TP]] |

---

## Entity 6 — HypothesisTestResult

Result of one statistical hypothesis test.

| Field | Type | Description |
|-------|------|-------------|
| hypothesis_name | str | e.g., "H1 - Job and Subscription" |
| feature | str | Feature name tested |
| test_name | str | e.g., "chi_square", "mann_whitney" |
| statistic | float | Test statistic |
| p_value | float | p-value |
| alpha | float | Significance level (always 0.05) |
| reject_null | bool | True if p_value < alpha |
| effect_size | float | Cramer's V, rank-biserial, or Cohen's d |
| effect_size_type | str | Name of effect size metric |
| interpretation | str | Plain-English business explanation |

---

## Entity 7 — CustomerPrediction

Output of a single-customer inference request.

| Field | Type | Description |
|-------|------|-------------|
| input_features | dict | Raw feature values provided by user |
| predicted_class | int | 0 (no) or 1 (yes) |
| subscription_probability | float | P(y=1), range [0, 1] |
| risk_level | str | "High" / "Medium" / "Low" opportunity |
| top_features | list[dict] | Top 5 SHAP contributions [{feature, shap_value}] |
| model_version | str | Model file name used |
| duration_excluded | bool | Always True in app (Realistic Business Model) |

**Validation rules**:
- `duration` must not appear in `input_features` for Realistic Business Model
- All required columns must be present (schema validated in `src/inference.py`)
- `subscription_probability` in [0, 1]

---

## Entity 8 — BatchPredictionRequest

Input/output contract for a CSV batch prediction run.

| Field | Type | Description |
|-------|------|-------------|
| input_df | pd.DataFrame | N rows with same schema as RawRecord (minus target, minus duration) |
| output_df | pd.DataFrame | Input columns + predicted_class + subscription_probability + rank |
| n_rows | int | Number of rows processed |
| validation_errors | list[str] | Missing or invalid columns detected |

**Validation rules**:
- `duration` silently stripped if present (with warning displayed)
- Missing required columns raise `ValueError` with column list
- `rank` column sorts customers from highest to lowest probability
