# Predicting Bank Term Deposit Subscription Using Machine Learning

> A professional end-to-end graduation ML project — from raw data to a deployed Streamlit app.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-red.svg)](https://streamlit.io)
[![FLAML](https://img.shields.io/badge/AutoML-FLAML_2.6-green.svg)](https://github.com/microsoft/FLAML)
[![Tests](https://img.shields.io/badge/tests-50%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 [Live Demo on Streamlit Cloud](https://share.streamlit.io)
> **Deploy your own**: See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for step-by-step instructions

---

## Problem Statement

A Portuguese bank runs telephone marketing campaigns to sell term deposit products. The goal is to
predict whether a contacted client will subscribe to a term deposit (`yes`/`no`). Accurate predictions
allow the bank to prioritise the most likely customers, reducing campaign cost and improving conversion rates.

**Challenge**: The target variable is highly imbalanced (~11.3% positive class). The feature `duration`
(last call duration) is an excellent predictor but is **unknown before the call is made**, so it must be
excluded from any business-facing model.

---

## Dataset

| Dataset | Rows | Features | Notes |
|---------|------|----------|-------|
| `bank-additional-full.csv` | 41,188 | 21 | **Main dataset** — all analysis and final models |
| `bank-additional.csv` | 4,119 | 21 | 10% sample |
| `bank-full.csv` | 45,211 | 17 | Older version (fewer economic features) |
| `bank.csv` | 4,521 | 17 | 10% sample of bank-full |

**Source**: [UCI ML Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

---

## Architecture

```
data/raw/  →  src/data_loader.py  →  src/features.py  →  src/preprocessing.py
                                                              ↓
                                                     src/modeling.py
                                                              ↓
                                              src/model_selection.py + src/evaluation.py
                                                              ↓
                                              src/explainability.py + src/inference.py
                                                              ↓
                                                    app/streamlit_app.py (7 pages)
```

Two model tracks:
- **Benchmark model** (Feature Set A): includes `duration` — upper bound reference only
- **Realistic Business Model** (Feature Set B): excludes `duration` — used in all app predictions

---

## EDA Summary

Key findings from `notebooks/02_eda_univariate.ipynb` and `notebooks/03_eda_bivariate_multivariate.ipynb`:

- **Class imbalance**: 88.7% No / 11.3% Yes — a 1:7.9 ratio. Default threshold (0.5) suppresses minority-class recall.
- **Best categorical predictor**: `poutcome=success` — clients with a prior successful campaign subscribe at **64.7%** (vs 11.3% baseline).
- **Economic cycle dominates**: When `nr.employed` drops (economic slowdown) subscription rates rise significantly. The 5 macro-economic features are highly correlated (r > 0.8).
- **Optimal contact months**: March, September, October, December show the highest conversion rates.
- **Diminishing returns**: After 3 campaign contacts, conversion probability drops sharply. Clients contacted 4+ times are ~40% less likely to convert.
- **Job & age matter**: Students (31%) and retired clients (25%) subscribe at 3–4× the overall rate. Bimodal age pattern — under-30 and over-60 are most receptive.
- **`unknown` values** in `default` (20.9%), `education` (4.2%), `housing`/`loan` (2.4%) are retained as a valid category — not imputed.

---

## Hypothesis Testing

7 statistical tests conducted in `notebooks/04_hypothesis_testing.ipynb` (α = 0.05, N = 41,188):

| ID | Hypothesis | Test | p-value | Decision | Effect Size |
|----|-----------|------|---------|----------|-------------|
| H1 | Job type is associated with subscription | Chi-Square | 4.19 × 10⁻¹⁹⁹ | ✅ Reject H₀ | Cramér's V = 0.153 |
| H2 | Education level is associated with subscription | Chi-Square | 3.31 × 10⁻³⁸ | ✅ Reject H₀ | Cramér's V = 0.069 |
| H3 | Housing loan status is associated with subscription | Chi-Square | 0.0583 | ❌ Fail to reject | Cramér's V = 0.012 |
| H4 | Previous campaign outcome is associated with subscription | Chi-Square | < 10⁻³⁰⁰ | ✅ Reject H₀ | Cramér's V = 0.321 |
| H5 | Age differs between subscribers and non-subscribers | Mann-Whitney U | 0.0161 | ✅ Reject H₀ | r = 0.022 |
| H6 | Campaign contacts differ between groups | Mann-Whitney U | 3.42 × 10⁻³⁸ | ✅ Reject H₀ | r = 0.110 |
| H7 | Economic indicators differ between groups | Mann-Whitney U | < 10⁻³⁰⁰ | ✅ Reject H₀ | r = 0.498 (large) |

Key takeaways: **6 of 7 confirmed**. H3 (housing loan) is not a meaningful targeting signal. Economic indicators (H7) show the largest practical effect size. Previous campaign success (H4) is the strongest categorical predictor.

---

## Modelling

8 baseline classifiers trained in `notebooks/06_modeling_baselines.ipynb` across two feature sets.

**Feature Set B results (no duration — realistic business model):**

| Model | ROC-AUC | PR-AUC | Accuracy | Subscribe Recall |
|-------|---------|--------|----------|-----------------|
| DummyClassifier | 0.505 | 0.112 | 80.4% | 0.12 |
| LogisticRegression | 0.797 | 0.435 | 82.1% | 0.65 |
| DecisionTreeClassifier | 0.622 | 0.182 | 84.5% | 0.33 |
| RandomForestClassifier | 0.785 | 0.405 | 89.4% | 0.29 |
| ExtraTreesClassifier | 0.752 | 0.324 | 88.5% | 0.31 |
| GradientBoostingClassifier | 0.796 | 0.461 | 90.1% | 0.23 |
| **HistGradientBoostingClassifier** | **0.816** | **0.498** | 85.6% | **0.65** |
| KNeighborsClassifier | 0.749 | 0.311 | 89.5% | 0.33 |

Best Realistic Business Model (baseline):
- **Model**: `HistGradientBoostingClassifier` (class_weight='balanced')
- **ROC-AUC**: 0.816
- **PR-AUC**: 0.498
- **Balanced Accuracy**: 0.766

---

## AutoML

FLAML AutoML run in `notebooks/09_advanced_pipeline.ipynb` (300s budget, 5-fold CV, metric=accuracy):

| Estimator | CV Accuracy | Test PR-AUC | Test ROC-AUC |
|-----------|-------------|-------------|--------------|
| **lgbm (FLAML champion)** | **90.12%** | **0.491** | **0.813** |
| LGBMClassifier (default) | — | 0.491 | 0.810 |
| LGBMClassifier (GridSearchCV) | 85.28% | 0.433 | 0.773 |
| CatBoostClassifier (default) | — | 0.487 | 0.809 |
| XGBClassifier (default) | — | 0.450 | 0.788 |

**FLAML champion config** (lgbm): `n_estimators=89, num_leaves=9, learning_rate=0.062, colsample_bytree=0.738`

FLAML's compact configuration outperformed all GridSearch alternatives — demonstrating AutoML's effectiveness at finding well-regularised models.

---

## Final Model

**Champion**: FLAML-lgbm with tuned threshold = **0.27** (`notebooks/09_advanced_pipeline.ipynb`)

### Root Cause: Why Default Threshold Fails

The FLAML model at default threshold (0.5) achieves 90.2% accuracy but only **24% Subscribe recall** — it predicts "No" for 76% of actual subscribers. Root causes:
1. Class imbalance (1:7.9) — majority class dominates accuracy gradient
2. Median predicted probability for actual subscribers = **0.31** (below the 0.5 threshold)
3. Accuracy metric doesn't penalise minority-class misses

### Improvement Strategies Comparison

| Strategy | Accuracy | Subscribe Recall | Subscribe F1 | Macro F1 |
|----------|----------|-----------------|--------------|----------|
| FLAML-lgbm (default thr=0.50) | **90.2%** | 0.24 | 0.36 | 0.65 |
| **FLAML-lgbm (thr=0.27) ⭐** | **88.7%** | **0.56** | **0.53** | 0.71 |
| FLAML-F1 metric (thr=0.22) | 87.3% | 0.53 | 0.49 | 0.71 |
| Balanced LGBM (thr=0.63) | 88.0% | **0.60** | **0.53** | **0.73** |

**Threshold tuning alone** (no retraining) raises Subscribe recall from 24% → 56% and F1 from 0.36 → 0.53 (+47%).

### Champion Metrics (production)

| Metric | Value |
|--------|-------|
| Model | FLAML LightGBM |
| Optimal threshold | **0.27** |
| Test Accuracy | 88.7% |
| Subscribe Precision | 0.50 |
| Subscribe Recall | **0.56** |
| Subscribe F1 | **0.53** |
| ROC-AUC | **0.813** |
| PR-AUC | 0.491 |

SHAP top features: `euribor3m`, `nr.employed`, `economic_stress_index`, `poutcome_success`, `cons.price.idx`

---

## Streamlit App (7 pages)

| Page | Description |
|------|-------------|
| Overview | Business problem, dataset summary, key metrics |
| EDA Dashboard | Full exploratory data analysis with interactive charts |
| Hypothesis Testing | Results table for all 7 statistical tests (H1–H7) |
| Model Performance | Comparison table, ROC/PR curves, lift chart, threshold tuner |
| Predict New Client | Single-customer prediction form (no duration field) |
| Batch Prediction | CSV upload → ranked predictions download |
| Business Recommendations | Segment insights, campaign strategy, model limitations |

> ⚠️ `duration` is never used in any prediction — it is only known after the call.

---

## Run Locally

```bash
# 1. Clone the repository
git clone <repo-url>
cd bank-marketing-ml-project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
make install

# Or for development (includes training tools):
# make install-dev

# 4. Verify data is present
ls data/raw/

# 5. Run tests
make test

# 6. Train all models
make train

# 7. Launch the Streamlit app
make app
```

---

## Deploy with Docker

```bash
make docker-build
make docker-run
```

Then open `http://localhost:8501` in your browser.

---

## Project Structure

```
├── app/                   # Streamlit app (entry + 6 pages)
├── data/
│   ├── raw/               # Original CSV files
│   └── processed/         # Engineered train/test splits
├── models/                # Saved .joblib model artifacts
├── notebooks/             # 9 Jupyter notebooks (01–09)
├── reports/
│   └── figures/           # All saved matplotlib/seaborn plots
├── scripts/               # train.py, generate_reports.py, predict_batch.py
├── src/                   # All Python modules
├── tests/                 # pytest test suite
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## Limitations

- Classical ML only — no deep learning or neural networks
- `duration` is excluded from all business predictions (data leakage prevention)
- Dataset spans 2008–2013 (Portuguese bank); may not generalise to other institutions or current conditions
- Models not production-hardened — no REST API serving, no drift monitoring
- Class imbalance remains challenging; threshold tuning improves recall at a small accuracy cost
- No temporal cross-validation — standard stratified split used (80/20, seed=42)

---

## References

- Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems, 62, 22–31.
- [UCI ML Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Wang, C., Wu, Q., Weimer, M., & Zhu, E. (2021). *FLAML: A Fast and Lightweight AutoML Library.* MLSys 2021.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* NeurIPS 2017. (SHAP)
- scikit-learn documentation: https://scikit-learn.org
- Streamlit documentation: https://docs.streamlit.io
