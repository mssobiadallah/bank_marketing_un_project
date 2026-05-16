# Tasks: Bank Marketing ML Project

**Input**: Design documents from `specs/001-bank-marketing-ml/`
**Branch**: `001-bank-marketing-ml`
**Date**: 2026-05-16

**Prerequisites**: plan.md ✅ | spec.md ✅ | research.md ✅ | data-model.md ✅ | contracts/ ✅ | quickstart.md ✅

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Create all project scaffolding, configuration, and environment files needed before any code is written.

- [ ] T001 Create `requirements.txt` with all dependencies (pandas, numpy, scikit-learn, scipy, statsmodels, matplotlib, seaborn, plotly, streamlit, joblib, shap, pycaret, xgboost, lightgbm, catboost, imbalanced-learn, pytest)
- [ ] T002 [P] Create `.gitignore` covering venv/, __pycache__/, .ipynb_checkpoints/, models/*.joblib, data/processed/, reports/figures/, .DS_Store
- [ ] T003 [P] Create `Makefile` with targets: install, test, train, app, docker-build, docker-run per plan.md
- [ ] T004 [P] Create `Dockerfile` using python:3.11-slim, WORKDIR /app, COPY requirements.txt, RUN pip install, COPY ., EXPOSE 8501, CMD streamlit run
- [ ] T005 [P] Create `.streamlit/config.toml` with theme and server settings for deployment
- [ ] T006 [P] Create `README.md` skeleton with all sections from plan.md section 10 (Problem, Dataset, Architecture, EDA, Hypothesis Testing, Modeling, AutoML, Final Model, App, Run Locally, Deploy, Limitations, References)
- [ ] T007 [P] Create empty `__init__.py` files in `src/` and `tests/`
- [ ] T008 [P] Create placeholder directories: `data/processed/`, `models/`, `reports/figures/`, `app/assets/`

**Checkpoint**: All scaffolding in place. `make install` completes without error.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core shared infrastructure that ALL user stories depend on. Must be complete before any user story begins.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T009 Create `src/config.py` defining: DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RANDOM_SEED=42, TARGET_COL="y", MAIN_DATASET filename, NUMERIC_COLS list, CATEGORICAL_COLS list, FEATURE_SET_A_COLS (with duration), FEATURE_SET_B_COLS (without duration), ENGINEERED_FEATURE_NAMES list
- [ ] T010 Create `src/utils.py` with: `ensure_dir(path)`, `save_figure(fig, path)`, `save_dataframe(df, path)`, `set_random_seed(seed)`, `timer` decorator, `get_logger(name)` — all with type hints and docstrings
- [ ] T011 [P] Create `tests/conftest.py` with pytest fixtures: `sample_df` (10-row DataFrame with all 21 columns), `full_df` (loads bank-additional-full.csv), `processed_df`, `fitted_pipeline`

**Checkpoint**: Foundation ready — `pytest tests/conftest.py` shows fixtures load without error.

---

## Phase 3: User Story 1 — Data Understanding (Priority: P1) 🎯 MVP Start

**Goal**: Load, compare, and summarise all four datasets; produce data understanding outputs that validate the data for downstream work.

**Independent Test**: Run `python -c "from src.data_loader import load_dataset, summarize_dataset; df = load_dataset('data/raw/bank-additional-full.csv'); print(summarize_dataset(df))"` — verify shape (41188, 21) and target distribution printed.

- [ ] T012 [US1] Implement `src/data_loader.py`: `load_dataset(path: str, sep: str = ";") -> pd.DataFrame` — reads CSV with semicolon separator, validates non-empty result, raises ValueError on wrong separator
- [ ] T013 [US1] Add `summarize_dataset(df: pd.DataFrame) -> dict` to `src/data_loader.py` — returns shape, column names, dtypes, missing value counts, duplicate count, target distribution if `y` exists, unknown-value counts per categorical column
- [ ] T014 [US1] Add `compare_datasets(paths: dict[str, str]) -> pd.DataFrame` to `src/data_loader.py` — loads each path, returns comparison table with dataset name, rows, columns, target distribution (yes%, no%), unknown counts
- [ ] T015 [US1] Add `validate_required_columns(df: pd.DataFrame, feature_set: str) -> list[str]` to `src/data_loader.py` — returns list of missing required columns for "set_a" or "set_b" per contracts/inference-schema.md
- [ ] T016 [P] [US1] Create `tests/test_data_loader.py`: test `load_dataset` loads bank-additional-full.csv with shape (41188, 21); test semicolon separator detected; test `summarize_dataset` keys; test `compare_datasets` returns 4-row DataFrame; test `validate_required_columns` catches missing `age` column; test missing file raises FileNotFoundError
- [ ] T017 [US1] Create `notebooks/01_data_understanding.ipynb`: load all 4 datasets, call `compare_datasets`, show `summarize_dataset` output for main dataset, display target distribution chart, display unknown-value summary table, write observations as markdown cells

**Checkpoint**: `pytest tests/test_data_loader.py` all pass. Notebook 01 runs end-to-end.

---

## Phase 4: User Story 2 — Full EDA (Priority: P1)

**Goal**: Univariate, bivariate, and multivariate analysis of `bank-additional-full.csv`; all figures saved to `reports/figures/`; summary DataFrames returned.

**Independent Test**: Run `python -c "import pandas as pd; from src.eda import get_numeric_summary, get_categorical_summary; df = pd.read_csv('data/raw/bank-additional-full.csv', sep=';'); print(get_numeric_summary(df, ['age','campaign']))"` — verify summary DataFrame returned.

- [ ] T018 [US2] Create `src/eda.py` with univariate functions:
  - `get_numeric_summary(df, numeric_cols) -> pd.DataFrame` — count, mean, std, min, 25%, 50%, 75%, max, skew, kurtosis
  - `get_categorical_summary(df, categorical_cols) -> pd.DataFrame` — count per category, frequency%, unknown count
  - `calculate_unknown_counts(df) -> pd.DataFrame` — per-column unknown count and percentage
  - `calculate_outliers_iqr(df, numeric_cols) -> pd.DataFrame` — lower fence, upper fence, outlier count, outlier%
  - `plot_target_distribution(df, target="y", output_path=None)` — bar chart of yes/no counts and percentages
- [ ] T019 [US2] Add univariate plot functions to `src/eda.py`:
  - `plot_numeric_distribution(df, column, output_path=None)` — histogram + KDE + boxplot side by side
  - `plot_categorical_distribution(df, column, output_path=None)` — horizontal bar chart sorted by frequency
  - `plot_all_numeric_distributions(df, numeric_cols, output_dir)` — saves one figure per column
  - `plot_all_categorical_distributions(df, categorical_cols, output_dir)` — saves one figure per column
- [ ] T020 [US2] Add bivariate analysis functions to `src/eda.py`:
  - `conversion_rate_by_category(df, category_col, target="y") -> pd.DataFrame` — count, yes_count, conversion_rate%, sorted descending
  - `numeric_summary_by_target(df, numeric_col, target="y") -> pd.DataFrame` — mean, median, std per target class
  - `plot_conversion_rate(df, category_col, target="y", output_path=None)` — horizontal bar chart by conversion rate
  - `plot_numeric_by_target(df, numeric_col, target="y", output_path=None)` — overlapping KDE or boxplot by target
  - `plot_all_conversion_rates(df, categorical_cols, output_dir)` — saves one figure per column
  - `plot_all_numerics_by_target(df, numeric_cols, output_dir)` — saves one figure per column
- [ ] T021 [US2] Create `src/multivariate_analysis.py`:
  - `correlation_matrix(df, numeric_cols) -> pd.DataFrame`
  - `plot_correlation_heatmap(df, numeric_cols, output_path=None)`
  - `cramers_v(x: pd.Series, y: pd.Series) -> float`
  - `cramers_v_matrix(df, categorical_cols) -> pd.DataFrame`
  - `point_biserial_table(df, numeric_cols, target="y") -> pd.DataFrame` — correlation, p-value per feature
  - `calculate_vif(df, numeric_cols) -> pd.DataFrame` — feature, VIF score
  - `segment_conversion_table(df, group_cols, target="y", min_count=50) -> pd.DataFrame`
  - `plot_cramers_v_heatmap(df, categorical_cols, output_path=None)`
- [ ] T022 [P] [US2] Create `notebooks/02_eda_univariate.ipynb`: load main dataset, call all univariate functions, display all numeric distribution plots, display all categorical distribution plots, display target distribution, display unknown-value summary, outlier table; write business observations as markdown
- [ ] T023 [P] [US2] Create `notebooks/03_eda_bivariate_multivariate.ipynb`: bivariate analysis for all features vs target, multivariate section with correlation heatmap, Cramér's V heatmap, VIF table, segment analysis for high/low conversion segments; answer all business questions from spec Phase 3; write EDA findings

**Checkpoint**: `reports/figures/` contains at minimum 30 saved figures. Notebooks 02 and 03 run end-to-end.

---

## Phase 5: User Story 4 — Preprocessing, Features & Baseline Modeling (Priority: P1)

**Goal**: Build reusable preprocessing pipeline, engineer all 9 features, train all baseline models on both feature sets, save metrics and model artifacts.

**Independent Test**: Run `python scripts/train.py` — verify `models/benchmark_model_with_duration.joblib`, `models/realistic_model_without_duration.joblib`, `models/preprocessing_pipeline.joblib`, and `reports/model_metrics.csv` all created.

- [ ] T024 [US4] Create `src/features.py`:
  - `encode_target(df: pd.DataFrame, target="y") -> pd.DataFrame` — maps yes→1, no→0 in-place
  - `add_features(df: pd.DataFrame, dataset_type="additional") -> pd.DataFrame` — adds all 9 engineered features: `was_previously_contacted` (pdays!=999), `campaign_intensity_group` (low/medium/high), `age_group` (young/middle/senior), `economic_stress_index` (euribor3m + emp.var.rate composite), `has_any_loan` (housing==yes OR loan==yes), `month_order` (jan=1…dec=12), `previous_contact_success_flag` (poutcome==success), `contact_is_cellular` (contact==cellular), `client_financial_pressure_flag` (default==yes OR has_any_loan)
  - `get_feature_lists(df, target="y", exclude_duration=False) -> dict` — returns {"numeric": [...], "categorical": [...], "target": "y"}
- [ ] T025 [US4] Create `src/preprocessing.py`:
  - `build_preprocessing_pipeline(numeric_cols, categorical_cols, scale_numeric=False) -> ColumnTransformer` — OneHotEncoder(handle_unknown="ignore", sparse_output=False) for categoricals; StandardScaler (optional) for numerics; passthrough remainder
  - `split_data(df, target="y", test_size=0.2, random_state=42) -> tuple` — stratified train/test split; returns X_train, X_test, y_train, y_test
  - `save_pipeline(pipeline, path: str)` — joblib.dump
  - `load_pipeline(path: str)` — joblib.load with FileNotFoundError check
- [ ] T026 [P] [US4] Create `tests/test_features.py`: test `encode_target` maps yes→1 no→0; test `add_features` adds exactly 9 new columns; test `was_previously_contacted`=0 when pdays=999; test `was_previously_contacted`=1 when pdays<999; test `get_feature_lists` returns correct column separation; test no column is duplicated
- [ ] T027 [P] [US4] Create `tests/test_preprocessing.py`: test `build_preprocessing_pipeline` returns ColumnTransformer; test fit/transform produces correct output shape; test OneHotEncoder handles unknown category; test `split_data` stratification (target ratio preserved ±1%); test `save_pipeline`/`load_pipeline` round-trip
- [ ] T028 [US4] Create `src/modeling.py`:
  - `get_baseline_models(random_state=42) -> dict` — returns dict of {name: estimator} for: DummyClassifier, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, KNeighborsClassifier; all with class_weight="balanced" where supported
  - `train_model(model, X_train, y_train, preprocessor, cv=5) -> tuple` — returns (fitted_pipeline, cv_scores_dict)
  - `save_model(pipeline, path: str)` — joblib.dump with directory creation
  - `load_model(path: str)` — joblib.load with FileNotFoundError check
- [ ] T029 [US4] Create `src/evaluation.py`:
  - `evaluate_binary_classifier(model, X_test, y_test, threshold=0.5) -> dict` — returns accuracy, balanced_accuracy, precision, recall, f1, roc_auc, average_precision, log_loss, confusion_matrix
  - `plot_confusion_matrix(y_true, y_pred, output_path=None)`
  - `plot_roc_curve(y_true, y_proba, model_name="", output_path=None)`
  - `plot_pr_curve(y_true, y_proba, model_name="", output_path=None)`
  - `save_metrics_csv(results: list[dict], path: str)` — appends rows to CSV
  - `plot_model_comparison(results_df, metric="average_precision", output_path=None)` — horizontal bar chart
- [ ] T030 [US4] Create `notebooks/05_feature_engineering.ipynb`: load main dataset, show raw vs engineered feature comparison, display all 9 new features with value counts and distributions, show `get_feature_lists` output for both feature sets, save processed train/test to `data/processed/`
- [ ] T031 [US4] Create `notebooks/06_modeling_baselines.ipynb`: load processed data, train all baseline models on Feature Set A and B, display cross-validation scores table, display test metrics table, plot ROC curves, PR curves, confusion matrices for top 3 models per feature set; highlight best model per metric; save `reports/model_metrics.csv`
- [ ] T032 [US4] Create `scripts/train.py`: full training pipeline — load `bank-additional-full.csv`, encode target, add features, get feature lists, build pipeline, split data, train all baseline models for both feature sets, evaluate, save metrics CSV, save best models and preprocessing pipeline to `models/`

**Checkpoint**: `pytest tests/test_features.py tests/test_preprocessing.py` all pass. `python scripts/train.py` completes and creates all 3 model files.

---

## Phase 6: User Story 6 — Model Selection & Explainability (Priority: P1)

**Goal**: Select the best realistic model by average_precision, tune its threshold, produce SHAP summary and waterfall plots, permutation importance table, and `model_card.md`.

**Independent Test**: Run `python -c "from src.model_selection import select_best_model; import pandas as pd; df = pd.read_csv('reports/model_metrics.csv'); print(select_best_model(df))"` — verify best model name returned.

- [ ] T033 [US6] Create `src/model_selection.py`:
  - `compare_model_results(results_df: pd.DataFrame) -> pd.DataFrame` — sorted by average_precision descending; includes rank column
  - `select_best_model(results_df: pd.DataFrame, primary_metric="average_precision") -> str` — returns model name with highest primary_metric for feature_set="set_b"
  - `tune_classification_threshold(y_true, y_proba, strategy="max_f1", target_value=None) -> tuple[float, dict]` — returns (optimal_threshold, metrics_at_threshold); supports "max_f1", "target_recall", "target_precision"
  - `evaluate_at_threshold(y_true, y_proba, threshold: float) -> dict` — precision, recall, f1 at given threshold
  - `create_lift_table(y_true, y_proba, n_bins=10) -> pd.DataFrame` — decile, n_customers, n_subscribers, conversion_rate, lift columns
  - `plot_lift_chart(lift_df: pd.DataFrame, output_path=None)`
  - `plot_threshold_curve(y_true, y_proba, output_path=None)` — F1/precision/recall vs threshold
- [ ] T034 [US6] Create `src/explainability.py`:
  - `permutation_importance_table(model, X_test, y_test, n_repeats=10) -> pd.DataFrame` — feature, importance_mean, importance_std sorted descending
  - `plot_feature_importance(model, feature_names, output_path=None)` — handles tree-based `.feature_importances_` and linear `.coef_`
  - `get_shap_explainer(model, X_sample)` — auto-selects TreeExplainer / LinearExplainer / KernelExplainer per research.md Decision 4
  - `shap_summary(model, X_sample, output_path=None)` — saves beeswarm summary plot to output_path
  - `shap_single_prediction(model, X_row: pd.DataFrame, output_path=None) -> list[dict]` — returns top 5 [{feature, shap_value}] sorted by abs value; saves waterfall plot
  - `generate_business_insights(feature_importance_df: pd.DataFrame, eda_summary: dict) -> list[str]` — returns list of plain-English insight strings
- [ ] T035 [US6] Create `notebooks/08_model_selection_explainability.ipynb`: load metrics CSV, show `compare_model_results` table, call `select_best_model`, load selected model, plot threshold curve, tune threshold with all three strategies, display lift table and lift chart, run `shap_summary`, run `shap_single_prediction` on one example, run `permutation_importance_table`, write business insight section as markdown
- [ ] T036 [US6] Create `models/model_card.md` filled with: model name, model type, target, dataset, features used (Feature Set B), features excluded (duration), main metrics (filled from notebook), limitations, ethical considerations — per the template in plan section 14

**Checkpoint**: SHAP summary plot saved to `reports/figures/`. Lift table shows top-decile lift. `models/model_card.md` filled with real metric values.

---

## Phase 7: User Story 7 — Streamlit App (Priority: P1)

**Goal**: 7-page multi-page Streamlit app that loads the Realistic Business Model, displays all EDA/hypothesis/model results, accepts single and batch predictions, and handles missing model files gracefully.

**Independent Test**: Run `streamlit run app/streamlit_app.py` — all 7 pages load without error; prediction form returns a result; CSV upload and download works; duration warning is visible.

- [ ] T037 [US7] Create `src/inference.py`:
  - `validate_input_schema(input_df, feature_set="set_b") -> list[str]` — returns list of error messages; strips `duration` with warning if present in set_b
  - `predict_single(model, pipeline, input_dict: dict) -> dict` — validates, preprocesses, predicts, returns CustomerPrediction dict per contracts/single-prediction.md
  - `predict_batch(model, pipeline, df: pd.DataFrame) -> pd.DataFrame` — validates, preprocesses, predicts all rows, adds predicted_class + subscription_probability + rank columns, sorted by probability descending
  - `load_model_and_pipeline(model_path: str, pipeline_path: str) -> tuple` — joblib.load both; raises FileNotFoundError with clear message if either missing
  - `risk_level(probability: float) -> str` — "High" if >=0.6, "Medium" if >=0.3, "Low" otherwise
- [ ] T038 [P] [US7] Create `tests/test_inference.py`: test `validate_input_schema` returns empty list for valid input; test it catches missing column; test `duration` is stripped with warning; test `predict_single` returns all required keys; test `predict_batch` returns DataFrame with rank column sorted correctly; test `load_model_and_pipeline` raises FileNotFoundError on missing file; test `risk_level` thresholds
- [ ] T039 [P] [US7] Create `tests/test_streamlit_inputs.py`: test `validate_input_schema` for all 19 required fields; test batch CSV with extra columns (duration stripped); test batch CSV missing required column returns error list; test empty DataFrame raises ValueError
- [ ] T040 [US7] Create `app/streamlit_app.py`: main entry point — sidebar navigation, project title/subtitle, Project Overview page (business problem, dataset table, target variable, important duration warning, main metrics summary), load model/pipeline with `st.cache_resource`, display `st.error` + `st.stop()` if model files missing
- [ ] T041 [US7] Create `app/pages/1_EDA_Dashboard.py`: `st.cache_data` data loading, tabs for Target Distribution / Numeric Features / Categorical Features / Bivariate Analysis / Multivariate; display all pre-saved figures from `reports/figures/` using `st.image`; display conversion-rate tables; display correlation heatmap; display unknown-value summary
- [ ] T042 [US7] Create `app/pages/2_Hypothesis_Testing.py`: display formatted table of all 7 hypothesis results from `reports/hypothesis_testing_report.md` or recomputed; columns: Hypothesis, Feature, Test, Statistic, p-value, Reject H0, Effect Size, Interpretation; highlight rejected nulls in colour
- [ ] T043 [US7] Create `app/pages/3_Model_Performance.py`: display model comparison table from `reports/model_metrics.csv`; Plotly ROC curves for all models; Plotly PR curves; confusion matrix for selected model; lift chart; threshold tuning slider; display selected model metrics at chosen threshold
- [ ] T044 [US7] Create `app/pages/4_Predict_New_Client.py`: prominent `st.warning` about duration exclusion; form with all 19 input fields with sensible defaults and help text; "Predict" button calls `predict_single`; display predicted class, probability, risk level badge; display top-5 SHAP features as horizontal bar chart
- [ ] T045 [US7] Create `app/pages/5_Batch_Prediction.py`: `st.file_uploader` for CSV; validate schema; call `predict_batch`; display preview of top 10 ranked customers; `st.download_button` for full predictions CSV; display warning if duration column was present and stripped
- [ ] T046 [US7] Create `app/pages/6_Business_Recommendations.py`: display top customer segments from segment analysis; contact strategy recommendations; campaign efficiency estimates (top decile lift); model limitations section; ethical considerations from model_card.md

**Checkpoint**: All 7 pages render. Prediction form returns result. CSV download works. Duration warning visible on pages 4 and 5.

---

## Phase 8: User Story 3 — Hypothesis Testing (Priority: P2)

**Goal**: Run all 7 hypothesis tests (H1–H7), produce a results DataFrame with all required fields, save report to `reports/hypothesis_testing_report.md`.

**Independent Test**: Run `python -c "import pandas as pd; from src.hypothesis_tests import run_all_hypothesis_tests; df = pd.read_csv('data/raw/bank-additional-full.csv', sep=';'); results = run_all_hypothesis_tests(df, {}); print(results[['hypothesis_name','reject_null','p_value']])"` — verify 7 rows returned.

- [ ] T047 [US3] Create `src/hypothesis_tests.py`:
  - `chi_square_test(df, feature, target="y") -> dict` — chi2 statistic, p_value, degrees_of_freedom, cramers_v, reject_null; validates at least 5 expected counts per cell
  - `mann_whitney_test(df, numeric_feature, target="y") -> dict` — U statistic, p_value, rank_biserial_correlation, reject_null
  - `t_test_feature(df, numeric_feature, target="y") -> dict` — t statistic, p_value, cohens_d, reject_null
  - `normality_check(df, numeric_feature, target="y") -> dict` — Shapiro-Wilk result per class (on sample if n>5000)
  - `cramers_v_effect_size(contingency_table) -> float`
  - `cohens_d(group1: pd.Series, group2: pd.Series) -> float`
  - `run_all_hypothesis_tests(df: pd.DataFrame, config: dict) -> pd.DataFrame` — runs H1–H7 (job, education, housing, poutcome chi-square; age and campaign Mann-Whitney; euribor3m/emp.var.rate/nr.employed/cons.price.idx/cons.conf.idx Mann-Whitney); returns DataFrame with columns: hypothesis_name, feature, test_name, statistic, p_value, alpha, reject_null, effect_size, effect_size_type, interpretation; saves report to `reports/hypothesis_testing_report.md`
- [ ] T048 [US3] Create `notebooks/04_hypothesis_testing.ipynb`: load main dataset, run `run_all_hypothesis_tests`, display formatted results table, display p-value bar chart, display effect sizes chart, write plain-English business interpretation for each hypothesis as markdown cells

**Checkpoint**: `run_all_hypothesis_tests` returns 7-row DataFrame. `reports/hypothesis_testing_report.md` written. Notebook 04 runs end-to-end.

---

## Phase 9: User Story 5 — AutoML Comparison (Priority: P2)

**Goal**: PyCaret classification experiments for both feature sets; ranked comparison table; tuned best model; `reports/automl_results.csv` with ≥8 model entries per feature set.

**Independent Test**: Open `notebooks/07_automl_experiments.ipynb`, run all cells — verify `reports/automl_results.csv` exists and contains ≥8 rows per feature set (or notebook exits gracefully with install instructions if PyCaret missing).

- [ ] T049 [US5] Create `notebooks/07_automl_experiments.ipynb`: section 1 — try/except PyCaret import with clear install message if absent; section 2 — load main dataset, encode target, add features; section 3 — PyCaret `setup()` with Feature Set B (no duration), `compare_models(sort="AUC")`, display leaderboard, tune best model, `evaluate_model`, `save_model`; section 4 — repeat for Feature Set A (with duration); section 5 — combine both leaderboards, export to `reports/automl_results.csv`; section 6 — comparison table side by side; add markdown cells with business-language result interpretation

**Checkpoint**: `reports/automl_results.csv` created with ≥8 model rows per feature set (or notebook shows graceful error if PyCaret not installed).

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Finalize deployment artifacts, scripts, remaining notebooks, and reports.

- [ ] T050 [P] Create `scripts/generate_reports.py`: runs all EDA functions on main dataset, saves all figures to `reports/figures/`, writes `reports/eda_report.md` with key findings and embedded figure paths, writes `reports/model_comparison_report.md` from `model_metrics.csv`
- [ ] T051 [P] Create `scripts/predict_batch.py`: CLI script with argparse `--input` and `--output`; loads model and pipeline; validates schema; runs `predict_batch`; saves sorted CSV; prints summary (n rows, top-10 customers)
- [ ] T052 [P] Create `notebooks/09_business_recommendations.ipynb`: load final model, load segment analysis results, display top 5 customer profiles with high conversion rate, campaign efficiency section (top decile lift, estimated savings), contact strategy recommendations, economic indicator context, write final business narrative as markdown
- [ ] T053 Finalise `README.md`: fill all skeleton sections with real content — add dataset description, architecture diagram (text), EDA summary (3 bullet points), hypothesis summary (3 key findings), modeling summary (best model + metrics), how to run locally, how to deploy, limitations, references per plan.md section 16
- [ ] T054 [P] Fill `reports/final_business_report.md`: executive summary, dataset description, methodology (CRISP-DM steps), EDA key findings, hypothesis testing results, model comparison, final model metrics, business recommendations, limitations, future work
- [ ] T055 [P] Run `pytest tests/` — fix any failing tests; ensure all 5 test files pass; check coverage covers `src/data_loader.py`, `src/features.py`, `src/preprocessing.py`, `src/inference.py`
- [ ] T056 [P] Final `docker build -t bank-marketing-ml .` — fix any build errors; verify `docker run -p 8501:8501 bank-marketing-ml` launches the app
- [ ] T057 Commit final state: `git add -A && git commit -m "feat: complete Bank Marketing ML project (all milestones)"`

**Checkpoint**: `pytest tests/` all pass. `docker build` succeeds. `streamlit run app/streamlit_app.py` runs. All report files exist.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all user stories
- **Phase 3 (US1)**: Depends on Phase 2 — can start as soon as Phase 2 done
- **Phase 4 (US2)**: Depends on Phase 2 — can start in parallel with Phase 3
- **Phase 5 (US4)**: Depends on Phase 3 (needs data_loader) and Phase 2
- **Phase 6 (US6)**: Depends on Phase 5 (needs trained models and metrics CSV)
- **Phase 7 (US7)**: Depends on Phase 5 (needs models) and Phase 6 (needs inference schema)
- **Phase 8 (US3)**: Depends on Phase 2 — can run in parallel with Phases 4–7
- **Phase 9 (US5)**: Depends on Phase 5 (needs processed data) — can run in parallel with Phases 6–7
- **Phase 10 (Polish)**: Depends on all phases complete

### User Story Dependencies

| Story | Depends On | Can Parallel With |
|-------|-----------|------------------|
| US1 (Phase 3) | Phase 2 | US2, US3 |
| US2 (Phase 4) | Phase 2 | US1, US3 |
| US3 (Phase 8) | Phase 2 | US1, US2 |
| US4 (Phase 5) | US1 (data_loader) | US2, US3 |
| US5 (Phase 9) | US4 (processed data) | US6, US7 |
| US6 (Phase 6) | US4 (trained models) | US7 |
| US7 (Phase 7) | US4 (models) + US6 (inference) | US5 |

### Parallel Opportunities

- T002, T003, T004, T005, T006, T007, T008 — all Phase 1 in parallel
- T011, T016, T026, T027, T038, T039 — test files in parallel with implementation
- T018, T019, T020, T021 — all US2 src/ functions can be written in parallel
- T022, T023 — US2 notebooks in parallel
- T024, T025 — features.py and preprocessing.py in parallel
- T026, T027 — test files in parallel
- T028, T029 — modeling.py and evaluation.py in parallel
- T033, T034 — model_selection.py and explainability.py in parallel
- T040–T046 — Streamlit pages in parallel (different files, no cross-dependencies)
- T050, T051, T052, T053, T054, T055, T056 — all Phase 10 mostly parallel

---

## Implementation Strategy

### MVP Scope (P1 Stories Only — Phases 1–7)

1. Phase 1: Setup → Phase 2: Foundational
2. Phase 3: US1 (data loading) → validate independently
3. Phase 5: US4 (preprocessing + modeling) → `python scripts/train.py` works
4. Phase 6: US6 (model selection) → final model selected
5. Phase 7: US7 (Streamlit app) → **app is live and functional**

MVP delivers: a working Streamlit app with real predictions from a properly trained
classical ML model, with SHAP explanations and EDA charts displayed.

### Incremental Delivery (Add P2 Stories After MVP)

- Phase 4 (US2 EDA): Add full EDA module and notebooks → richer EDA Dashboard page
- Phase 8 (US3 Hypothesis Testing): Add stats module → Hypothesis Testing page populated
- Phase 9 (US5 AutoML): Add PyCaret notebook → Model Performance page shows AutoML results
- Phase 10: Polish, deployment, final reports

### Notes

- `[P]` tasks operate on different files — safe to run in parallel
- `[USn]` label enables traceability back to user stories and acceptance scenarios
- Commit after each phase checkpoint at minimum
- Run `pytest tests/` before advancing to the next phase
- Avoid implementing app pages before `src/inference.py` is complete (Phase 7 depends on T037)
