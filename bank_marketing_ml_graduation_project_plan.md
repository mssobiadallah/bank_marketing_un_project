# Bank Marketing ML Graduation Project — Spec Kit Implementation Plan

## 1. Project Summary

### Project Title
**Predicting Bank Term Deposit Subscription Using Machine Learning**

### Business Problem
A bank runs direct marketing campaigns through phone calls. The goal is to predict whether a client will subscribe to a term deposit (`y = yes/no`) before or during campaign planning, so the bank can prioritize high-potential customers, reduce wasted calls, and improve campaign conversion.

### Dataset Family
This project uses the public **Bank Marketing** datasets:

| Dataset | Rows | Columns | Recommended Usage |
|---|---:|---:|---|
| `bank.csv` | 4,521 | 17 | Fast experimentation and heavy algorithms |
| `bank-full.csv` | 45,211 | 17 | Main dataset without economic indicators |
| `bank-additional.csv` | 4,119 | 21 | Fast experimentation with economic indicators |
| `bank-additional-full.csv` | 41,188 | 21 | Recommended final dataset |

### Recommended Main Dataset
Use **`bank-additional-full.csv`** as the main dataset because it includes additional social and economic context features:
- `emp.var.rate`
- `cons.price.idx`
- `cons.conf.idx`
- `euribor3m`
- `nr.employed`

These features provide stronger business insight and usually improve predictive performance.

### Classification Target
The target variable is:

```text
y
```

Values:
- `yes`: client subscribed to a term deposit
- `no`: client did not subscribe

### Important Modeling Warning
The feature `duration` is the call duration. It is highly predictive, but it is usually **not available before the call happens**. Therefore, the project should build two model versions:

1. **Benchmark Model**
   - Includes `duration`
   - Shows upper-bound performance

2. **Realistic Business Model**
   - Excludes `duration`
   - Used for realistic pre-call customer targeting

The final Streamlit app should make this clear.

---

## 2. Project Goals

### Technical Goals
- Build a full machine learning pipeline using only classical ML models.
- Perform professional EDA:
  - Univariate analysis
  - Bivariate analysis
  - Multivariate analysis
  - Outlier analysis
  - Missing/unknown value analysis
  - Class imbalance analysis
- Perform hypothesis testing.
- Build multiple machine learning models.
- Use AutoML for comparison and model selection.
- Explain model behavior using feature importance and SHAP.
- Build a clean Streamlit frontend.
- Prepare the project for deployment.

### Business Goals
- Identify customer profiles more likely to subscribe.
- Understand which factors affect campaign success.
- Provide recommendations for marketing teams.
- Build an app that allows users to:
  - Upload data
  - Run predictions
  - View model confidence
  - View insights
  - Explore EDA charts

---

## 3. Success Metrics

### Model Metrics
Because the dataset is imbalanced, accuracy alone is not enough.

Primary metrics:
- ROC-AUC
- PR-AUC / Average Precision
- F1-score for the positive class `yes`
- Recall for `yes`
- Precision for `yes`

Secondary metrics:
- Accuracy
- Confusion matrix
- Balanced accuracy
- Log loss
- Calibration curve

### Business Metrics
- Top-decile lift
- Conversion rate in top 10% predicted customers
- Number of calls saved
- Expected campaign efficiency improvement

Example:
> If the model ranks customers by probability of subscription, the marketing team can contact only the top 20% highest-probability customers and compare the expected conversion rate against random targeting.

---

## 4. Project Scope

### In Scope
- Data understanding
- Data cleaning
- EDA
- Hypothesis testing
- Feature engineering
- Classical ML modeling
- AutoML comparison
- Model explainability
- Streamlit app
- Deployment readiness
- Documentation
- GitHub Copilot / Spec Kit implementation prompts

### Out of Scope
- Deep learning models
- Neural networks
- Real-time production banking integration
- Paid cloud deployment dependency
- Customer-sensitive private banking data

---

## 5. Recommended Tech Stack

### Core
- Python 3.11+
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Statsmodels

### Visualization
- Matplotlib
- Seaborn
- Plotly

### Modeling
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- XGBoost, if allowed
- LightGBM, if allowed
- CatBoost, if allowed
- Support Vector Machine, mainly on the smaller dataset
- KNN, mainly as a baseline

### AutoML Options
Recommended:
- PyCaret Classification

Alternative:
- FLAML
- AutoGluon Tabular

For a graduation project, PyCaret is the easiest to present, while FLAML is lightweight and practical.

### Explainability
- SHAP
- Permutation importance
- Partial dependence plots

### App
- Streamlit
- Joblib
- Plotly

### Optional Deployment
- Streamlit Community Cloud
- Hugging Face Spaces
- Docker
- Render
- Railway

---

## 6. Proposed Repository Structure

```text
bank-marketing-ml-project/
│
├── data/
│   ├── raw/
│   │   ├── bank.csv
│   │   ├── bank-full.csv
│   │   ├── bank-additional.csv
│   │   └── bank-additional-full.csv
│   │
│   ├── processed/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── feature_metadata.json
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
│   ├── hypothesis_tests.py
│   ├── modeling.py
│   ├── evaluation.py
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
│   └── final_business_report.md
│
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 1_EDA_Dashboard.py
│   │   ├── 2_Model_Performance.py
│   │   ├── 3_Predict_New_Client.py
│   │   └── 4_Business_Insights.py
│   └── assets/
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_inference.py
│   └── test_streamlit_inputs.py
│
├── .gitignore
├── README.md
├── requirements.txt
├── Dockerfile
├── Makefile
└── spec.md
```

---

## 7. Implementation Phases

## Phase 0 — Project Setup

### Objectives
Prepare the repository, environment, dependency files, and project conventions.

### Tasks
1. Create GitHub repository.
2. Create folder structure.
3. Add raw datasets into `data/raw/`.
4. Create virtual environment.
5. Install required packages.
6. Create `requirements.txt`.
7. Create `.gitignore`.
8. Create `README.md`.
9. Create `Makefile`.
10. Add basic tests.

### Suggested `requirements.txt`

```text
pandas
numpy
scikit-learn
scipy
statsmodels
matplotlib
seaborn
plotly
streamlit
joblib
shap
pycaret
xgboost
lightgbm
catboost
imbalanced-learn
pytest
python-dotenv
```

If PyCaret causes dependency problems, move it to a separate file:

```text
requirements-automl.txt
```

### Copilot / Spec Kit Prompt

```text
You are implementing a professional machine learning graduation project.

Create a Python project structure for a Bank Marketing classification project.
Use the folder structure described in spec.md.
Add clean, modular files under src/.
Add README.md, requirements.txt, .gitignore, Makefile, and tests/.
Do not implement deep learning.
Use only classical machine learning models.
Make the project production-like but simple enough for university students.
```

---

## Phase 1 — Data Understanding

### Objectives
Understand the dataset, target, feature types, missing values, imbalance, and business context.

### Tasks
1. Load all four CSV files.
2. Confirm separator is `;`.
3. Compare shapes:
   - `bank-full.csv`
   - `bank.csv`
   - `bank-additional-full.csv`
   - `bank-additional.csv`
4. Identify columns and data types.
5. Identify categorical and numerical features.
6. Analyze target distribution.
7. Detect `unknown` categories.
8. Check duplicate rows.
9. Check impossible values.
10. Write a data dictionary.

### Expected Outputs
- `reports/data_understanding.md`
- Data dictionary table
- Target distribution chart
- Missing/unknown value summary
- Dataset comparison table

### Important Data Checks
For `bank-additional-full.csv`:
- `pdays = 999` means client was not previously contacted.
- `unknown` is used in categorical variables.
- `duration` should be handled carefully.

For `bank-full.csv`:
- `pdays = -1` means client was not previously contacted.
- `unknown` exists in fields like `contact`, `poutcome`, and `education`.

### Copilot / Spec Kit Prompt

```text
Implement src/data_loader.py.

Requirements:
1. Create load_dataset(path: str, sep: str = ";") -> pandas.DataFrame.
2. Create summarize_dataset(df) that returns:
   - shape
   - column names
   - dtypes
   - missing values
   - duplicate count
   - target distribution if column y exists
   - unknown counts for categorical columns
3. Create compare_datasets(paths: dict) -> pandas.DataFrame.
4. Add clear docstrings and type hints.
5. Add tests in tests/test_data_loader.py.
```

---

## Phase 2 — EDA: Univariate Analysis

### Objectives
Analyze each feature individually.

### Numerical Features
For `bank-additional-full.csv`:
- `age`
- `duration`
- `campaign`
- `pdays`
- `previous`
- `emp.var.rate`
- `cons.price.idx`
- `cons.conf.idx`
- `euribor3m`
- `nr.employed`

### Categorical Features
- `job`
- `marital`
- `education`
- `default`
- `housing`
- `loan`
- `contact`
- `month`
- `day_of_week`
- `poutcome`

### Tasks
1. Summary statistics for numerical features.
2. Distribution plots.
3. Boxplots.
4. Outlier detection using IQR.
5. Frequency tables for categorical variables.
6. Bar charts for categorical variables.
7. Unknown category analysis.
8. Target distribution analysis.

### Expected Outputs
- Histograms
- Boxplots
- Count plots
- Outlier table
- Unknown-value summary
- EDA observations

### Example Questions
- What is the age distribution?
- Are most clients contacted once or many times?
- Which categorical features have many `unknown` values?
- How imbalanced is the target variable?
- Are economic indicators concentrated in specific ranges?

### Copilot / Spec Kit Prompt

```text
Implement src/eda.py with reusable EDA functions.

Functions:
1. get_numeric_summary(df, numeric_cols)
2. get_categorical_summary(df, categorical_cols)
3. calculate_unknown_counts(df)
4. calculate_outliers_iqr(df, numeric_cols)
5. plot_numeric_distribution(df, column, output_path=None)
6. plot_categorical_distribution(df, column, output_path=None)
7. plot_target_distribution(df, target="y", output_path=None)

Use matplotlib/seaborn/plotly where appropriate.
Save figures to reports/figures when output_path is provided.
Return clean DataFrames for summaries.
```

---

## Phase 3 — EDA: Bivariate Analysis

### Objectives
Analyze the relationship between each feature and the target variable.

### Tasks
1. Numerical features vs target:
   - Boxplots by `y`
   - KDE/histograms by `y`
   - Mean/median comparison
2. Categorical features vs target:
   - Conversion rate by category
   - Stacked bar charts
   - Count and percentage tables
3. Rank categories by subscription rate.
4. Identify strong predictors.

### Important Business Questions
- Which job types have higher subscription rates?
- Does education affect subscription probability?
- Are customers without loans more likely to subscribe?
- Does previous campaign success increase current success?
- Which months show better campaign results?
- Does economic context affect subscription probability?

### Expected Outputs
- Conversion-rate tables
- Feature-target plots
- Business observations

### Copilot / Spec Kit Prompt

```text
Extend src/eda.py.

Add functions:
1. conversion_rate_by_category(df, category_col, target="y")
2. numeric_summary_by_target(df, numeric_col, target="y")
3. plot_conversion_rate(df, category_col, target="y", output_path=None)
4. plot_numeric_by_target(df, numeric_col, target="y", output_path=None)

Target y is categorical yes/no.
Convert y to binary where needed:
yes = 1
no = 0.
```

---

## Phase 4 — EDA: Multivariate Analysis

### Objectives
Study interactions between multiple variables and reduce the risk of misleading one-variable conclusions.

### Tasks
1. Correlation matrix for numerical features.
2. Cramér's V for categorical-categorical relationships.
3. Point-biserial correlation for numerical features vs binary target.
4. Pairwise analysis of important features:
   - `age` + `job` + `y`
   - `education` + `job` + `y`
   - `poutcome` + `previous` + `y`
   - `contact` + `month` + `y`
   - `euribor3m` + `nr.employed` + `y`
5. Multicollinearity check:
   - Correlation threshold
   - VIF for numerical variables
6. Segment analysis:
   - High conversion customer segments
   - Low conversion customer segments

### Expected Outputs
- Correlation heatmap
- Cramér's V heatmap
- VIF table
- Segment analysis table
- Feature interaction observations

### Copilot / Spec Kit Prompt

```text
Implement src/multivariate_analysis.py.

Functions:
1. correlation_matrix(df, numeric_cols)
2. plot_correlation_heatmap(df, numeric_cols, output_path=None)
3. cramers_v(x, y)
4. cramers_v_matrix(df, categorical_cols)
5. point_biserial_table(df, numeric_cols, target="y")
6. calculate_vif(df, numeric_cols)
7. segment_conversion_table(df, group_cols, target="y", min_count=50)

Use robust error handling.
Return DataFrames.
Save plots when output_path is provided.
```

---

## Phase 5 — Hypothesis Testing

### Objectives
Use statistical tests to validate whether observed differences are statistically meaningful.

### General Rules
- Define null hypothesis `H0`.
- Define alternative hypothesis `H1`.
- Use significance level `alpha = 0.05`.
- Report p-value.
- Report test statistic.
- Report effect size where possible.
- Explain result in business language.

### Recommended Hypotheses

#### H1 — Job Type and Subscription
- **H0:** Subscription is independent of job type.
- **H1:** Subscription is associated with job type.
- Test: Chi-square test of independence.
- Effect size: Cramér's V.

#### H2 — Education and Subscription
- **H0:** Subscription is independent of education level.
- **H1:** Subscription is associated with education level.
- Test: Chi-square test.
- Effect size: Cramér's V.

#### H3 — Housing Loan and Subscription
- **H0:** Subscription is independent of housing loan status.
- **H1:** Housing loan status is associated with subscription.
- Test: Chi-square test.
- Effect size: Cramér's V.

#### H4 — Previous Campaign Outcome and Subscription
- **H0:** Previous campaign outcome is independent of current subscription.
- **H1:** Previous campaign outcome is associated with current subscription.
- Test: Chi-square test.
- Effect size: Cramér's V.

#### H5 — Age Difference Between Subscribers and Non-Subscribers
- **H0:** Average age is the same for subscribers and non-subscribers.
- **H1:** Average age differs between subscribers and non-subscribers.
- Test:
  - First check normality.
  - If non-normal, use Mann-Whitney U test.
  - If approximately normal, use t-test.
- Effect size:
  - Cohen's d, or rank-biserial correlation.

#### H6 — Campaign Contacts and Subscription
- **H0:** Number of campaign contacts is the same across target classes.
- **H1:** Number of campaign contacts differs between target classes.
- Test: Mann-Whitney U test.
- Effect size: rank-biserial correlation.

#### H7 — Economic Indicators and Subscription
Test each of:
- `euribor3m`
- `emp.var.rate`
- `nr.employed`
- `cons.price.idx`
- `cons.conf.idx`

Use:
- Mann-Whitney U test
- Effect size
- Business interpretation

### Expected Outputs
- `reports/hypothesis_testing_report.md`
- Statistical test results table
- Plain-English interpretation

### Copilot / Spec Kit Prompt

```text
Implement src/hypothesis_tests.py.

Functions:
1. chi_square_test(df, feature, target="y")
2. mann_whitney_test(df, numeric_feature, target="y")
3. t_test_feature(df, numeric_feature, target="y")
4. normality_check(df, numeric_feature, target="y")
5. cramers_v_effect_size(contingency_table)
6. cohens_d(group1, group2)
7. run_all_hypothesis_tests(df, config)

Return a DataFrame with:
- hypothesis_name
- feature
- test_name
- statistic
- p_value
- alpha
- reject_null
- effect_size
- interpretation
```

---

## Phase 6 — Data Preprocessing and Feature Engineering

### Objectives
Create a clean, reusable preprocessing pipeline for modeling and inference.

### Preprocessing Tasks
1. Convert target:
   - `yes` → 1
   - `no` → 0
2. Handle categorical `unknown` values:
   - Option 1: keep as category
   - Option 2: replace with `Unknown`
   - Option 3: impute based on most frequent category
3. Handle `pdays`:
   - For additional dataset: `999` means not previously contacted.
   - Create feature: `was_previously_contacted`.
   - Replace `999` with `NaN` or keep after adding flag.
4. Encode categorical variables:
   - OneHotEncoder for linear/tree models
   - Ordinal encoding only if logically ordered
5. Scale numerical features:
   - StandardScaler for Logistic Regression, SVM, KNN
   - Not necessary for tree-based models
6. Handle imbalance:
   - Class weights
   - SMOTE only inside cross-validation pipeline
   - Threshold tuning
7. Split data:
   - Train/validation/test
   - Stratified split by target
   - Optional time-aware split because full datasets are ordered by date

### Feature Engineering Ideas
- `was_previously_contacted`
- `campaign_intensity_group`
- `age_group`
- `economic_stress_index`
- `has_any_loan`
- `month_order`
- `previous_contact_success_flag`
- `contact_is_cellular`
- `client_financial_pressure_flag`

### Duration Strategy
Create two feature sets:

#### Feature Set A — Benchmark
Includes:
- All features including `duration`

#### Feature Set B — Realistic
Excludes:
- `duration`

The final selected production-like model should use Feature Set B.

### Copilot / Spec Kit Prompt

```text
Implement src/preprocessing.py and src/features.py.

Requirements:
1. Create target encoder yes/no to 1/0.
2. Create add_features(df, dataset_type="additional") function.
3. Create get_feature_lists(df, target="y", exclude_duration=False).
4. Create build_preprocessing_pipeline(numeric_cols, categorical_cols, scale_numeric=False).
5. Use sklearn ColumnTransformer.
6. Use OneHotEncoder(handle_unknown="ignore").
7. Add option to exclude duration.
8. Save preprocessing pipeline using joblib.
9. Add tests for feature generation and preprocessing.
```

---

## Phase 7 — Baseline Modeling

### Objectives
Build baseline models before advanced tuning.

### Baseline Models
1. DummyClassifier
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. Extra Trees
6. Gradient Boosting
7. HistGradientBoosting
8. KNN
9. SVM on small dataset only

Optional if environment supports:
10. XGBoost
11. LightGBM
12. CatBoost

### Evaluation Setup
Use:
- Stratified train/test split
- Cross-validation
- Same preprocessing logic for all models
- Same random seed
- Metrics focused on positive class

### Required Evaluation Outputs
- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
- ROC curve
- Precision-recall curve

### Copilot / Spec Kit Prompt

```text
Implement src/modeling.py and src/evaluation.py.

Requirements:
1. Create get_baseline_models(random_state=42).
2. Create train_model(model, X_train, y_train, preprocessor).
3. Create evaluate_binary_classifier(model, X_test, y_test).
4. Return metrics:
   - accuracy
   - balanced_accuracy
   - precision
   - recall
   - f1
   - roc_auc
   - average_precision
5. Plot confusion matrix, ROC curve, and PR curve.
6. Save metrics to reports/model_metrics.csv.
7. Save trained models to models/.
8. Support two experiments:
   - with duration
   - without duration
```

---

## Phase 8 — AutoML Experimentation

### Objectives
Use AutoML to compare many classical ML models and identify strong candidates.

### Recommended AutoML Approach
Use PyCaret Classification.

### AutoML Tasks
1. Run AutoML with `duration`.
2. Run AutoML without `duration`.
3. Compare top models.
4. Tune best models.
5. Blend/stack only if explainable and not too complex.
6. Export best AutoML model.
7. Compare AutoML result with manual models.

### PyCaret Example Flow
```python
from pycaret.classification import *

setup(
    data=df,
    target="y",
    session_id=42,
    normalize=True,
    fix_imbalance=True,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.9
)

best = compare_models(sort="AUC")
tuned = tune_model(best, optimize="AUC")
evaluate_model(tuned)
save_model(tuned, "models/pycaret_best_model")
```

### AutoML Deliverables
- `reports/automl_results.csv`
- Best model name
- Best model metrics
- Comparison with manual models
- Explanation of why final model was selected

### Copilot / Spec Kit Prompt

```text
Create notebooks/07_automl_experiments.ipynb.

Use PyCaret Classification to:
1. Load bank-additional-full.csv.
2. Run an experiment with duration.
3. Run an experiment without duration.
4. Compare models using ROC-AUC and average precision.
5. Tune the best model.
6. Save the best model.
7. Export model comparison tables to reports/automl_results.csv.
8. Add markdown cells explaining results in business language.

Do not use deep learning models.
```

---

## Phase 9 — Model Selection

### Objectives
Choose the final model based on performance, interpretability, business usefulness, and deployment simplicity.

### Selection Criteria
| Criterion | Explanation |
|---|---|
| ROC-AUC | Ranking quality |
| PR-AUC | Better for imbalanced data |
| Recall | Captures more potential subscribers |
| Precision | Avoids wasting calls |
| Interpretability | Can explain decisions |
| Stability | Similar performance in cross-validation and test |
| Deployment simplicity | Easy to save and load |
| Business acceptance | Marketing team can understand it |

### Recommended Final Candidates
1. Logistic Regression
   - Strong baseline
   - Highly explainable
2. Random Forest
   - Good nonlinear baseline
   - Feature importance
3. LightGBM / XGBoost / CatBoost
   - Usually strong performance
   - Good feature importance and SHAP compatibility
4. HistGradientBoosting
   - Scikit-learn native option

### Final Recommendation Strategy
Use:
- **Best explainable model** as the main production-like model.
- **Best performance model** as a benchmark.
- Compare both in the final report.

### Threshold Tuning
Default threshold `0.5` may not be best.

Tune threshold based on:
- Max F1
- Desired recall
- Desired precision
- Business capacity, for example contacting top 20% customers

### Copilot / Spec Kit Prompt

```text
Implement model selection logic.

Create src/model_selection.py.

Functions:
1. compare_model_results(results_df)
2. select_best_model(results_df, primary_metric="average_precision")
3. tune_classification_threshold(y_true, y_proba, strategy)
4. evaluate_at_threshold(y_true, y_proba, threshold)
5. create_lift_table(y_true, y_proba, n_bins=10)

Save:
- final selected model
- selected threshold
- model comparison report
- lift table
```

---

## Phase 10 — Explainability and Insight Generation

### Objectives
Explain why the model predicts a customer as likely or unlikely to subscribe.

### Explainability Methods
1. Global feature importance
2. Permutation importance
3. SHAP summary plot
4. SHAP waterfall plot for single prediction
5. Partial dependence plots
6. Logistic Regression coefficients if using linear model

### Business Insights to Extract
- Which features increase subscription probability?
- Which features decrease subscription probability?
- Which customer segments have high conversion?
- How does previous campaign success affect subscription?
- How do economic conditions affect subscription?
- Are repeated contacts useful or harmful?
- Which months or contact channels perform better?

### Copilot / Spec Kit Prompt

```text
Implement src/explainability.py.

Requirements:
1. permutation_importance_table(model, X_test, y_test)
2. plot_feature_importance(model, feature_names)
3. shap_summary(model, X_sample)
4. shap_single_prediction(model, X_row)
5. generate_business_insights(feature_importance_df, eda_results)

Make functions robust for:
- sklearn pipelines
- tree-based models
- logistic regression
```

---

## Phase 11 — Streamlit Frontend

### Objectives
Build an interactive app that presents the project professionally.

### App Pages

#### Page 1 — Project Overview
Show:
- Project problem
- Dataset explanation
- Target variable
- Main metrics
- Important warning about `duration`

#### Page 2 — EDA Dashboard
Show:
- Target distribution
- Numeric distributions
- Categorical distributions
- Conversion rate by category
- Correlation heatmap
- Unknown values summary

#### Page 3 — Hypothesis Testing
Show:
- Hypothesis table
- p-values
- effect sizes
- business interpretation

#### Page 4 — Model Performance
Show:
- Model comparison table
- ROC curve
- PR curve
- Confusion matrix
- Lift chart
- Threshold tuning

#### Page 5 — Predict New Client
Input fields:
- age
- job
- marital
- education
- default
- housing
- loan
- contact
- month
- day_of_week
- campaign
- pdays
- previous
- poutcome
- economic indicators

Output:
- predicted class
- probability of subscription
- risk/opportunity level
- top explanation features

#### Page 6 — Batch Prediction
Allow:
- CSV upload
- prediction download
- probability ranking
- top customers to contact

#### Page 7 — Business Recommendations
Show:
- Top customer segments
- Campaign recommendations
- Contact strategy
- Model limitations

### Streamlit UX Requirements
- Sidebar navigation
- Clean title and subtitle
- Tabs for charts
- Download buttons
- Clear warnings
- No technical overload on business pages

### Copilot / Spec Kit Prompt

```text
Create a professional Streamlit app in app/streamlit_app.py.

Requirements:
1. Multi-page Streamlit app.
2. Load trained model and preprocessing pipeline from models/.
3. Include project overview, EDA dashboard, hypothesis testing, model performance, single prediction, batch prediction, and business insights.
4. Use Plotly for interactive charts.
5. Use caching with st.cache_data and st.cache_resource.
6. Add file uploader for batch prediction.
7. Add download button for predictions.
8. Add clear warning that duration should not be used in realistic pre-call prediction.
9. Keep UI clean and graduation-project professional.
```

---

## Phase 12 — Deployment Readiness

### Objectives
Make the project easy to run locally and deploy.

### Local Run Commands

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
```

### Makefile

```makefile
install:
	pip install -r requirements.txt

test:
	pytest tests/

train:
	python scripts/train.py

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t bank-marketing-ml .

docker-run:
	docker run -p 8501:8501 bank-marketing-ml
```

### Deployment Options
1. Streamlit Community Cloud
2. Hugging Face Spaces
3. Render
4. Railway
5. Docker on VPS

### Copilot / Spec Kit Prompt

```text
Make the project deployment-ready.

Add:
1. Dockerfile
2. Makefile
3. .streamlit/config.toml
4. README deployment section
5. scripts/train.py
6. scripts/generate_reports.py

Ensure the app can run with:
streamlit run app/streamlit_app.py

Ensure the training can run with:
python scripts/train.py
```

---

## 8. Testing Plan

### Unit Tests
Test:
- Data loading
- Column validation
- Target encoding
- Feature engineering
- Preprocessing pipeline
- Model inference
- Batch prediction

### Data Validation Tests
Check:
- Required columns exist
- Target contains only yes/no
- No duplicated feature names
- Input schema matches training schema
- Batch CSV upload contains required columns

### App Tests
Check:
- Model loads successfully
- Single prediction works
- Batch prediction works
- App does not crash with missing optional columns

### Copilot / Spec Kit Prompt

```text
Create pytest tests for the project.

Tests should cover:
1. Loading semicolon-separated CSV files.
2. Validating required columns.
3. Encoding y yes/no to 1/0.
4. Creating engineered features.
5. Building preprocessing pipeline.
6. Running model prediction on one row.
7. Running batch prediction on a sample DataFrame.
8. Handling missing columns with clear errors.
```

---

## 9. Final Report Structure

### Suggested Final Graduation Report

```text
1. Abstract
2. Introduction
3. Business Problem
4. Dataset Description
5. Related Work
6. Methodology
   6.1 CRISP-DM
   6.2 Data Understanding
   6.3 EDA
   6.4 Hypothesis Testing
   6.5 Feature Engineering
   6.6 Modeling
   6.7 AutoML
   6.8 Model Selection
   6.9 Explainability
7. Results
8. Business Recommendations
9. Streamlit Application
10. Deployment
11. Limitations
12. Future Work
13. References
```

### Future Work
- Add cost-sensitive learning
- Add campaign budget optimization
- Add lead scoring workflow
- Add monitoring dashboard
- Add model drift detection
- Add time-based validation
- Add A/B testing plan

---

## 10. README Structure

```text
# Bank Marketing Machine Learning Project

## Problem Statement
## Dataset
## Project Architecture
## EDA Summary
## Hypothesis Testing Summary
## Modeling Approach
## AutoML Results
## Final Model
## Streamlit App
## How to Run Locally
## How to Deploy
## Project Screenshots
## Limitations
## References
```

---

## 11. Main Scripts to Implement

### `scripts/train.py`
Responsibilities:
1. Load dataset
2. Create feature sets with and without duration
3. Split data
4. Train baseline models
5. Run evaluation
6. Select best model
7. Save model
8. Save metrics

### `scripts/generate_reports.py`
Responsibilities:
1. Generate EDA report
2. Generate hypothesis testing report
3. Generate model report
4. Save figures

### `scripts/predict_batch.py`
Responsibilities:
1. Load model
2. Load input CSV
3. Validate schema
4. Generate probabilities
5. Save predictions

---

## 12. Spec Kit Implementation Milestones

### Milestone 1 — Foundation
- Repository structure
- Environment setup
- Data loading
- Basic README
- Tests for loading

### Milestone 2 — EDA
- Data understanding notebook
- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- EDA report

### Milestone 3 — Statistical Testing
- Hypothesis testing module
- Hypothesis testing notebook
- Statistical report

### Milestone 4 — Modeling
- Preprocessing pipeline
- Feature engineering
- Baseline models
- Evaluation module

### Milestone 5 — AutoML and Selection
- PyCaret/FLAML experiment
- Model comparison
- Threshold tuning
- Final model selection

### Milestone 6 — Explainability
- Feature importance
- SHAP
- Business insight extraction

### Milestone 7 — Streamlit App
- Multi-page UI
- Prediction form
- Batch upload
- Model performance page
- Business insights page

### Milestone 8 — Deployment
- Dockerfile
- Makefile
- Deployment guide
- Final README
- Final report

---

## 13. Recommended Development Order for GitHub Copilot

Use this order to reduce project errors:

1. Create project structure.
2. Implement data loading.
3. Implement schema validation.
4. Implement EDA functions.
5. Generate EDA notebooks.
6. Implement hypothesis testing.
7. Implement preprocessing and features.
8. Implement baseline modeling.
9. Implement evaluation metrics.
10. Implement AutoML notebook.
11. Implement model selection and threshold tuning.
12. Implement explainability.
13. Implement Streamlit app.
14. Add tests.
15. Add Docker and deployment files.
16. Finalize README and reports.

---

## 14. Final Model Card Template

```text
# Model Card — Bank Marketing Subscription Predictor

## Model Name
Realistic Bank Marketing Lead Scoring Model

## Model Type
Classical machine learning binary classifier

## Target
Predict whether a client will subscribe to a bank term deposit.

## Dataset
Bank Marketing dataset with social and economic indicators.

## Features Used
All selected pre-call features excluding duration.

## Features Excluded
duration, because it is not available before the call ends.

## Intended Use
Prioritize customers for marketing calls.

## Not Intended For
Making final financial decisions, credit decisions, or customer eligibility decisions.

## Main Metrics
- ROC-AUC:
- PR-AUC:
- Recall:
- Precision:
- F1:

## Limitations
- Historical campaign data from 2008–2010.
- Dataset is from a Portuguese banking institution.
- Target is imbalanced.
- Some categorical values are unknown.
- Real-world performance may differ.

## Ethical Considerations
- Do not use this model to unfairly exclude customers.
- Monitor performance across age and demographic segments.
- Use as decision support, not as the only decision-maker.
```

---

## 15. Graduation Presentation Structure

### Slide 1 — Title
Bank Marketing Subscription Prediction Using Machine Learning

### Slide 2 — Problem
Marketing campaigns are costly and not all customers are likely to subscribe.

### Slide 3 — Dataset
Explain rows, columns, target, and source.

### Slide 4 — Methodology
CRISP-DM / project workflow.

### Slide 5 — EDA Highlights
Show target imbalance and important feature distributions.

### Slide 6 — Hypothesis Testing
Show 3–5 important validated hypotheses.

### Slide 7 — Modeling
Show models tested.

### Slide 8 — AutoML
Show AutoML comparison.

### Slide 9 — Final Model
Show selected model and metrics.

### Slide 10 — Explainability
Show feature importance or SHAP.

### Slide 11 — Streamlit App
Show screenshots.

### Slide 12 — Business Recommendations
Show how bank can use the model.

### Slide 13 — Limitations
Mention duration leakage, old data, imbalance, and generalization.

### Slide 14 — Future Work
Campaign optimization, monitoring, A/B testing.

---

## 16. References

Use these references in the report:

1. Moro, S., Laureano, R., & Cortez, P. (2011). *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*. Proceedings of the European Simulation and Modelling Conference - ESM'2011.

2. Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems.

3. UCI Machine Learning Repository — Bank Marketing Dataset.

---

## 17. Important Notes for Students

1. Do not rely only on accuracy because the dataset is imbalanced.
2. Always compare a model with `duration` and without `duration`.
3. The realistic model should not use `duration`.
4. Use explainability to turn the project from a normal ML project into a business decision-support project.
5. Keep the Streamlit app simple, clean, and business-friendly.
6. Use GitHub commits for every milestone.
7. Document every major decision.
8. Use tests to make the project look professional.
9. Avoid deep learning because it is unnecessary for this structured tabular dataset.
10. Focus on insight, modeling discipline, and deployment readiness.
