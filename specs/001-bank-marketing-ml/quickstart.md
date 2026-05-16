# Quickstart: Bank Marketing ML Project

**Date**: 2026-05-16
**Feature**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

---

## Prerequisites

- Python 3.11+
- Git
- (Optional) Docker

---

## 1 — Clone and Set Up Environment

```bash
# Clone the repository
git clone <repo-url>
cd bank-marketing-ml-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note**: If PyCaret causes dependency conflicts, install it separately:
> ```bash
> pip install pycaret
> ```

---

## 2 — Verify Data

The four raw datasets should be present in `data/raw/`:

```bash
ls data/raw/
# bank.csv
# bank-full.csv
# bank-additional.csv
# bank-additional-full.csv
```

All files use `;` as separator. Verify with:

```python
import pandas as pd
df = pd.read_csv("data/raw/bank-additional-full.csv", sep=";")
print(df.shape)   # Expected: (41188, 21)
```

---

## 3 — Run Tests

```bash
pytest tests/ -v
```

All tests must pass before proceeding to model training.

---

## 4 — Train Models

```bash
python scripts/train.py
```

This script:
1. Loads `data/raw/bank-additional-full.csv`
2. Engineers features and builds preprocessing pipeline
3. Trains all baseline models on both feature sets
4. Saves models to `models/`
5. Saves metrics to `reports/model_metrics.csv`

Expected runtime: 5–15 minutes depending on hardware.

---

## 5 — Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

The app loads the **Realistic Business Model** (without `duration`) from `models/`.
Run `scripts/train.py` first to generate the model files.

---

## 6 — Run Notebooks

Notebooks must be run in order:

```bash
jupyter notebook
```

Or open individually in VS Code. Run in this sequence:
1. `01_data_understanding.ipynb`
2. `02_eda_univariate.ipynb`
3. `03_eda_bivariate_multivariate.ipynb`
4. `04_hypothesis_testing.ipynb`
5. `05_feature_engineering.ipynb`
6. `06_modeling_baselines.ipynb`
7. `07_automl_experiments.ipynb` (requires PyCaret)
8. `08_model_selection_explainability.ipynb`
9. `09_business_recommendations.ipynb`

---

## 7 — Generate Reports

```bash
python scripts/generate_reports.py
```

Saves EDA figures to `reports/figures/` and writes report markdown files.

---

## 8 — Batch Prediction (CLI)

```bash
python scripts/predict_batch.py \
    --input data/my_customers.csv \
    --output data/predictions.csv
```

The input CSV must contain the 19 required pre-call features (no `duration`).

---

## 9 — Docker Deployment

```bash
# Build image
docker build -t bank-marketing-ml .

# Run app
docker run -p 8501:8501 bank-marketing-ml
```

---

## 10 — Makefile Shortcuts

```bash
make install       # pip install -r requirements.txt
make test          # pytest tests/
make train         # python scripts/train.py
make app           # streamlit run app/streamlit_app.py
make docker-build  # docker build
make docker-run    # docker run on port 8501
```

---

## Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| DATA_DIR | data/raw | Override raw data directory |
| MODELS_DIR | models | Override models directory |
| REPORTS_DIR | reports | Override reports directory |
| RANDOM_SEED | 42 | Override random seed |

---

## Troubleshooting

**PyCaret import error**: Install separately: `pip install pycaret`

**SHAP import error**: Install separately: `pip install shap`

**Streamlit app shows "Model not found"**: Run `python scripts/train.py` first.

**Wrong CSV separator**: All raw data files use `;`. Use `pd.read_csv(path, sep=';')`.

**Memory error on AutoML notebook**: Use `bank-additional.csv` (4,119 rows) instead of
`bank-additional-full.csv` for the AutoML experiment only.
