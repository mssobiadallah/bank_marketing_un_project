# Feature Specification: Bank Marketing ML Project

**Feature Branch**: `001-bank-marketing-ml`

**Created**: 2026-05-16

**Status**: Draft

---

## Project Summary

A bank runs direct marketing campaigns through phone calls. The goal is to predict whether a client will subscribe to a term deposit (`y = yes/no`) before or during campaign planning, so the bank can prioritise high-potential customers, reduce wasted calls, and improve campaign conversion.

**Main dataset**: `bank-additional-full.csv` (41,188 rows, 21 columns, includes social and economic indicators)

**Two model tracks must be built**:

| Track | Description | `duration` included? |
|---|---|---|
| Benchmark Model | Shows upper-bound performance | ✅ Yes |
| Realistic Business Model | Pre-call customer targeting | ❌ No |

The final production-facing model is always the **Realistic Business Model** (without `duration`).

---

## User Scenarios & Testing

### User Story 1 — Data Scientist: Explore and Understand the Dataset (Priority: P1)

A data scientist loads all four dataset variants, reviews their structure, compares shapes and column types, identifies missing/unknown values, checks class imbalance, and produces a data understanding report.

**Why this priority**: All downstream work depends on understanding what the data contains and whether it is trustworthy.

**Independent Test**: Can be fully tested by running `src/data_loader.py` against `bank-additional-full.csv` and verifying a summary report is produced.

**Acceptance Scenarios**:

1. **Given** the raw CSV files exist in `data/raw/`, **When** `load_dataset()` is called for each file, **Then** a DataFrame is returned with the correct shape and column names.
2. **Given** a loaded DataFrame, **When** `summarize_dataset()` is called, **Then** it returns shape, dtypes, missing value counts, duplicate count, target distribution, and unknown-value counts per categorical column.
3. **Given** all four datasets, **When** `compare_datasets()` is called, **Then** a comparison table is returned showing rows, columns, and target distribution for each file.

---

### User Story 2 — Data Scientist: Perform Full EDA (Priority: P1)

A data scientist performs univariate, bivariate, and multivariate analysis on `bank-additional-full.csv` and produces saved figures and summary tables covering all numerical and categorical features.

**Why this priority**: EDA shapes all modelling decisions and is a mandatory graded component of the graduation project.

**Independent Test**: Can be fully tested by running EDA functions on `bank-additional-full.csv` and verifying figures are saved to `reports/figures/` and summary DataFrames are returned.

**Acceptance Scenarios**:

1. **Given** the full dataset, **When** univariate analysis functions are called for all numerical and categorical features, **Then** histograms, boxplots, and frequency tables are produced.
2. **Given** the full dataset, **When** bivariate analysis functions are called, **Then** conversion-rate tables and feature-vs-target plots are produced for all features.
3. **Given** the full dataset, **When** multivariate analysis functions are called, **Then** a correlation heatmap, Cramér's V matrix, VIF table, and segment-conversion table are produced.
4. **Given** a categorical feature and target `y`, **When** `conversion_rate_by_category()` is called, **Then** a DataFrame with subscription rate per category is returned sorted descending.

---

### User Story 3 — Data Scientist: Run Statistical Hypothesis Tests (Priority: P2)

A data scientist runs all seven planned hypothesis tests, receives a results table with test names, p-values, effect sizes, and plain-English interpretations, and saves a hypothesis testing report.

**Why this priority**: Hypothesis testing is a mandatory graded component.

**Independent Test**: Can be fully tested by calling `run_all_hypothesis_tests()` on the full dataset and verifying results contain all seven hypotheses with correct statistical outputs.

**Acceptance Scenarios**:

1. **Given** a categorical feature and binary target, **When** `chi_square_test()` is called, **Then** chi-square statistic, p-value, degrees of freedom, and Cramér's V are returned.
2. **Given** a numerical feature and binary target, **When** `mann_whitney_test()` is called, **Then** U-statistic, p-value, and rank-biserial correlation are returned.
3. **Given** `alpha = 0.05`, **When** any test is run, **Then** `reject_null` is correctly set to `True` if `p_value < alpha`.
4. **Given** the full dataset and hypothesis config, **When** `run_all_hypothesis_tests()` is called, **Then** a DataFrame with all seven hypothesis results is returned and saved to `reports/hypothesis_testing_report.md`.

---

### User Story 4 — Data Scientist: Build and Evaluate Baseline ML Models (Priority: P1)

A data scientist builds a preprocessing pipeline, trains all baseline classical ML models for both feature sets (with and without `duration`), evaluates all models using the same metrics, and saves results.

**Why this priority**: Building and comparing models is the core deliverable of the graduation project.

**Independent Test**: Can be fully tested by running `scripts/train.py` and verifying that model files are saved to `models/` and a metrics CSV is saved to `reports/model_metrics.csv`.

**Acceptance Scenarios**:

1. **Given** the full dataset, **When** `build_preprocessing_pipeline()` is called, **Then** a fitted scikit-learn `Pipeline` is returned that handles numerical scaling and categorical one-hot encoding.
2. **Given** a fitted pipeline and model, **When** `train_model()` is called, **Then** the model trains without error and cross-validation scores are returned.
3. **Given** a trained model and test set, **When** `evaluate_binary_classifier()` is called, **Then** ROC-AUC, PR-AUC, precision, recall, F1, balanced accuracy, and confusion matrix are returned.
4. **Given** two feature sets (with/without `duration`), **When** all baseline models are trained, **Then** separate result tables exist for each feature set.
5. **Given** a trained model, **When** `save_model()` is called, **Then** a `.joblib` file is created in `models/`.

---

### User Story 5 — Data Scientist: Run AutoML Comparison (Priority: P2)

A data scientist runs a PyCaret classification experiment for both feature sets, compares all model performances, tunes the best model, and exports results.

**Why this priority**: AutoML comparison elevates the project above a basic modelling exercise and is required per the project plan.

**Independent Test**: Can be fully tested by running `notebooks/07_automl_experiments.ipynb` end-to-end and verifying `reports/automl_results.csv` is produced.

**Acceptance Scenarios**:

1. **Given** the prepared dataset, **When** PyCaret `compare_models()` is run, **Then** a ranked comparison table is returned sorted by ROC-AUC.
2. **Given** the best model from AutoML, **When** `tune_model()` is run, **Then** improved or equal ROC-AUC is returned.
3. **Given** both experiments (with/without `duration`), **When** results are exported, **Then** `reports/automl_results.csv` contains all model names and metrics for both runs.

---

### User Story 6 — Data Scientist: Select Final Model and Explain Predictions (Priority: P1)

A data scientist selects the final realistic model (without `duration`), tunes its classification threshold, and generates global and local explanations using feature importance and SHAP.

**Why this priority**: Model explainability converts this from a raw ML project into a business decision-support tool.

**Independent Test**: Can be fully tested by calling `shap_summary()` and `permutation_importance_table()` on the selected model and verifying figures are saved and DataFrames are returned.

**Acceptance Scenarios**:

1. **Given** candidate models and a results DataFrame, **When** `select_best_model()` is called with `primary_metric="average_precision"`, **Then** the model with the highest average precision is returned.
2. **Given** the selected model and test probabilities, **When** `tune_classification_threshold()` is called, **Then** an optimal threshold value is returned with supporting metrics at that threshold.
3. **Given** the selected model and a test sample, **When** `shap_summary()` is called, **Then** a SHAP summary plot is saved to `reports/figures/`.
4. **Given** a single input row, **When** `shap_single_prediction()` is called, **Then** a SHAP waterfall plot showing feature contributions for that prediction is produced.

---

### User Story 7 — Marketing Analyst: Use the Streamlit App (Priority: P1)

A non-technical marketing analyst opens the Streamlit app, explores EDA charts, views model performance metrics, enters a single customer's attributes, receives a subscription probability and risk level, and downloads a batch prediction file for a list of customers.

**Why this priority**: The Streamlit app is the final deliverable and graduation demonstration piece.

**Independent Test**: Can be fully tested by running `streamlit run app/streamlit_app.py`, navigating all pages, submitting the prediction form, and verifying a CSV download works.

**Acceptance Scenarios**:

1. **Given** the app is running, **When** the user navigates to the EDA Dashboard page, **Then** target distribution, feature distributions, conversion-rate charts, and correlation heatmap are displayed.
2. **Given** the app is running, **When** the user navigates to the Hypothesis Testing page, **Then** a formatted table of all seven hypothesis results with p-values, effect sizes, and plain-English interpretations is displayed.
3. **Given** the app is running, **When** the user navigates to Model Performance, **Then** a model comparison table, ROC curve, PR curve, confusion matrix, and lift chart are displayed.
4. **Given** the user fills in all customer fields on the Predict New Client page, **When** "Predict" is clicked, **Then** a predicted class, subscription probability, risk level, and top explanation features are displayed.
5. **Given** the user uploads a valid CSV on the Batch Prediction page, **When** predictions are generated, **Then** a downloadable CSV with predicted probabilities and ranked customers is provided.
6. **Given** the Predict New Client page, **When** the page loads, **Then** a visible warning states that `duration` is excluded from the realistic model because it is not available before the call.
7. **Given** the app is running, **When** the model or data files are missing from `models/`, **Then** a clear, user-friendly error message is shown and the app does not crash.

---

### Edge Cases

- What happens when `bank-additional-full.csv` contains `pdays = 999`? → `was_previously_contacted = False` flag must be created; `999` treated as not-contacted.
- What happens when categorical columns contain `unknown` values? → Kept as a valid category by default; handled by `OneHotEncoder(handle_unknown="ignore")`.
- What happens when the batch upload CSV is missing required columns? → A clear validation error is shown listing the missing columns.
- What happens when the uploaded batch CSV contains `duration`? → The app strips it silently for the realistic model and displays a warning.
- What happens when a model trained on the benchmark feature set (with `duration`) is accidentally used on the realistic feature set? → Schema validation in `src/inference.py` must raise a descriptive error.
- What happens when the class imbalance causes all predictions to be the majority class? → Class weights are applied by default; DummyClassifier baseline is always included for reference.
- What happens when PyCaret is not installed? → AutoML notebook must fail gracefully with a clear install instruction; all other modules must remain functional.

---

## Requirements

### Functional Requirements

- **FR-001**: The system MUST load all four CSV dataset variants using semicolon (`;`) as separator.
- **FR-002**: The system MUST produce summary statistics including shape, dtypes, missing values, duplicate count, target distribution, and unknown-value counts per categorical column.
- **FR-003**: The system MUST build two separate feature sets: Feature Set A (with `duration`) and Feature Set B (without `duration`).
- **FR-004**: The system MUST perform univariate analysis for all numerical and categorical features in `bank-additional-full.csv`.
- **FR-005**: The system MUST perform bivariate analysis producing conversion-rate tables and feature-vs-target plots for all features.
- **FR-006**: The system MUST perform multivariate analysis including correlation matrix, Cramér's V matrix, point-biserial correlation, VIF table, and segment-conversion table.
- **FR-007**: The system MUST run all seven planned hypothesis tests (H1–H7) and produce a results table with statistic, p-value, alpha, reject_null, effect size, and business interpretation for each.
- **FR-008**: The system MUST create a reusable scikit-learn `ColumnTransformer` preprocessing pipeline with OneHotEncoder for categorical features and optional StandardScaler for numerical features.
- **FR-009**: The system MUST engineer the following features: `was_previously_contacted`, `campaign_intensity_group`, `age_group`, `economic_stress_index`, `has_any_loan`, `month_order`, `previous_contact_success_flag`, `contact_is_cellular`, `client_financial_pressure_flag`.
- **FR-010**: The system MUST train and evaluate at minimum: DummyClassifier, Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, and HistGradientBoosting using stratified cross-validation.
- **FR-011**: The system MUST report: ROC-AUC, PR-AUC, precision, recall, F1, balanced accuracy, log loss, and confusion matrix for every trained model.
- **FR-012**: The system MUST save all trained models as `.joblib` files in `models/`.
- **FR-013**: The system MUST run a PyCaret AutoML experiment for both feature sets and export results to `reports/automl_results.csv`.
- **FR-014**: The system MUST tune the classification threshold of the final model using at least one strategy (max-F1, target-recall, or target-precision).
- **FR-015**: The system MUST produce a SHAP summary plot, SHAP waterfall plot for a single prediction, and a permutation importance table for the final selected model.
- **FR-016**: The system MUST expose a multi-page Streamlit app with: Project Overview, EDA Dashboard, Hypothesis Testing, Model Performance, Predict New Client, Batch Prediction, and Business Recommendations pages.
- **FR-017**: The Streamlit app MUST display a visible, prominent warning on any prediction page that `duration` is excluded from the realistic model.
- **FR-018**: The Streamlit app MUST support CSV upload for batch predictions and provide a download button for results.
- **FR-019**: The system MUST include pytest unit tests covering: data loading, column validation, target encoding, feature engineering, preprocessing pipeline, model inference, and batch prediction.
- **FR-020**: The project MUST include `requirements.txt`, `Makefile`, `Dockerfile`, `.gitignore`, `README.md`, and `reports/model_card.md`.

### Non-Functional Requirements

- **NFR-001**: All code MUST use Python 3.11+.
- **NFR-002**: Only classical ML models are permitted. Deep learning and neural networks are explicitly out of scope.
- **NFR-003**: The main dataset for all final modelling MUST be `bank-additional-full.csv`.
- **NFR-004**: All modules under `src/` MUST have type hints and docstrings.
- **NFR-005**: Random seed MUST be set to `42` across all experiments for reproducibility.
- **NFR-006**: The project MUST be runnable with `streamlit run app/streamlit_app.py` after `pip install -r requirements.txt`.
- **NFR-007**: The project MUST be trainable with `python scripts/train.py` from the root directory.
- **NFR-008**: All figures MUST be saved to `reports/figures/` when `output_path` is provided.
- **NFR-009**: The Streamlit app MUST use `st.cache_data` and `st.cache_resource` to cache data and model loading.

---

## Key Entities

- **Dataset**: One of four CSV files loaded from `data/raw/`. Primary entity for all analysis.
- **Feature Set A**: All features including `duration`. Used for the benchmark model only.
- **Feature Set B**: All features excluding `duration`. Used for the realistic business model and Streamlit app.
- **Benchmark Model**: A trained binary classifier using Feature Set A. Represents performance upper-bound.
- **Realistic Business Model**: A trained binary classifier using Feature Set B. The production-facing model.
- **Preprocessing Pipeline**: A scikit-learn `ColumnTransformer` + `Pipeline` object saved as a `.joblib` file.
- **Hypothesis Test Result**: A row in the results DataFrame with `hypothesis_name`, `feature`, `test_name`, `statistic`, `p_value`, `alpha`, `reject_null`, `effect_size`, `interpretation`.
- **Model Metrics**: A row in `reports/model_metrics.csv` with model name, feature set, and all evaluation metrics.
- **Customer Prediction**: A single row input yielding `predicted_class`, `subscription_probability`, `risk_level`, top SHAP features.
- **Batch Prediction**: A DataFrame of N rows yielding predictions ranked by subscription probability.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: The realistic business model achieves ROC-AUC ≥ 0.75 on the held-out test set.
- **SC-002**: The realistic business model achieves PR-AUC ≥ 0.40 on the held-out test set (reflecting class imbalance of ~11% positive).
- **SC-003**: The top-decile lift of the final model is ≥ 2.5× baseline random targeting.
- **SC-004**: All seven hypothesis tests produce results with correct p-values, effect sizes, and reject_null flags.
- **SC-005**: All pytest tests pass with `pytest tests/` from the project root.
- **SC-006**: The Streamlit app loads and all seven pages render without error after `pip install -r requirements.txt`.
- **SC-007**: A non-technical user can obtain a single-customer subscription prediction within 30 seconds of opening the app.
- **SC-008**: Batch prediction for 1,000 rows completes in under 60 seconds.
- **SC-009**: EDA notebooks produce all required figures in `reports/figures/` when run end-to-end.
- **SC-010**: AutoML experiment produces `reports/automl_results.csv` with at minimum 8 model entries per feature set.

---

## Assumptions

- The `bank-additional-full.csv` dataset is the authoritative source for all modelling and EDA. The smaller variants are used only for fast experimentation.
- `pdays = 999` in the additional dataset means the client was not previously contacted and must be converted to a binary flag.
- `unknown` values in categorical columns are treated as a valid category and handled by the encoder rather than being imputed.
- XGBoost, LightGBM, and CatBoost are included if the environment supports them; if not, the project remains complete without them.
- PyCaret is installed in a separate environment step if it causes dependency conflicts, and its absence does not break any non-AutoML module.
- SMOTE, if used for class imbalance, is applied only inside cross-validation folds, never on the full training set before splitting.
- The project is intended for local and Streamlit Community Cloud deployment; paid cloud infrastructure is out of scope.
- The Streamlit app always loads the **Realistic Business Model** (Feature Set B, without `duration`) for all user-facing predictions.
- `duration` may be shown in the EDA Dashboard for analytical purposes but must never be used as a model input in the app.
- All random operations use `random_state=42` for full reproducibility.

---

## Project File & Folder Structure

```text
bank-marketing-ml-project/
│
├── data/
│   ├── raw/
│   │   ├── bank.csv
│   │   ├── bank-full.csv
│   │   ├── bank-additional.csv
│   │   └── bank-additional-full.csv
│   └── processed/
│       ├── train.csv
│       ├── test.csv
│       └── feature_metadata.json
│
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
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── eda.py
│   ├── multivariate_analysis.py
│   ├── hypothesis_tests.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── model_selection.py
│   ├── explainability.py
│   ├── inference.py
│   └── utils.py
│
├── models/
│   ├── benchmark_model_with_duration.joblib
│   ├── realistic_model_without_duration.joblib
│   ├── preprocessing_pipeline.joblib
│   └── model_card.md
│
├── reports/
│   ├── figures/
│   ├── eda_report.md
│   ├── hypothesis_testing_report.md
│   ├── model_comparison_report.md
│   ├── automl_results.csv
│   ├── model_metrics.csv
│   └── final_business_report.md
│
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 1_EDA_Dashboard.py
│   │   ├── 2_Hypothesis_Testing.py
│   │   ├── 3_Model_Performance.py
│   │   ├── 4_Predict_New_Client.py
│   │   ├── 5_Batch_Prediction.py
│   │   └── 6_Business_Recommendations.py
│   └── assets/
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_inference.py
│   └── test_streamlit_inputs.py
│
├── scripts/
│   ├── train.py
│   ├── generate_reports.py
│   └── predict_batch.py
│
├── specs/
│   └── 001-bank-marketing-ml/
│       ├── spec.md  ← this file
│       └── checklists/
│           └── requirements.md
│
├── .streamlit/
│   └── config.toml
│
├── .gitignore
├── README.md
├── requirements.txt
├── Dockerfile
├── Makefile
└── spec.md  (root alias, points to specs/001-bank-marketing-ml/spec.md)
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11+ |
| Data | Pandas, NumPy |
| Statistics | SciPy, Statsmodels |
| Visualisation | Matplotlib, Seaborn, Plotly |
| ML Core | Scikit-learn |
| Gradient Boosting | XGBoost, LightGBM, CatBoost (optional) |
| Imbalance | imbalanced-learn (SMOTE) |
| AutoML | PyCaret Classification |
| Explainability | SHAP |
| App | Streamlit, Plotly |
| Persistence | Joblib |
| Testing | Pytest |
| Deployment | Docker, Streamlit Community Cloud |
