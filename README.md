# Predicting Bank Term Deposit Subscription Using Machine Learning

> A professional end-to-end graduation ML project — from raw data to a deployed Streamlit app.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-red.svg)](https://streamlit.io)

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

<!-- Fill after running notebooks/02 and notebooks/03 -->

Key findings:
- ...
- ...
- ...

---

## Hypothesis Testing

<!-- Fill after running notebooks/04 -->

Key results (H1–H7):
- ...
- ...
- ...

---

## Modelling

<!-- Fill after running scripts/train.py and notebooks/06 -->

Baseline models trained: DummyClassifier, LogisticRegression, DecisionTree, RandomForest, ExtraTrees,
GradientBoosting, HistGradientBoosting, KNN (+ optional XGBoost, LightGBM, CatBoost).

Best Realistic Business Model:
- **Model**: <!-- fill -->
- **ROC-AUC**: <!-- fill -->
- **PR-AUC (average_precision)**: <!-- fill -->
- **Optimal threshold**: <!-- fill -->

---

## AutoML

<!-- Fill after running notebooks/07 -->

PyCaret `compare_models()` results for Feature Set B:
- ...

---

## Final Model

<!-- Fill after running notebooks/08 -->

Selected model, tuned threshold, SHAP feature importance:
- ...

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
- `duration` is excluded from business predictions per project design
- All analysis uses `bank-additional-full.csv` (41,188 rows)
- Models are not production-hardened (no API serving, no monitoring)

---

## References

- Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems, 62, 22–31.
- [UCI ML Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- scikit-learn documentation: https://scikit-learn.org
- SHAP documentation: https://shap.readthedocs.io
- PyCaret documentation: https://pycaret.readthedocs.io
- Streamlit documentation: https://docs.streamlit.io
