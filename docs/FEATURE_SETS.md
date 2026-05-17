# Feature Sets Explained

## Overview

This project uses **two feature sets** for model training:

---

## Feature Set A (Benchmark Model)
**29 features total** — includes `duration`

### Purpose
- **Reference only** — shows the upper performance limit
- NOT used for real predictions
- Demonstrates what accuracy is possible if we knew call duration beforehand

### Why not use it?
⚠️ **Data Leakage**: The `duration` feature (call length in seconds) is only known **after** the call ends. You can't use it to decide who to call because you don't know it yet!

### Features Included
- **Client Info** (7): age, job, marital, education, default, housing, loan
- **Contact Info** (5): contact, month, day_of_week, **duration** ⚠️, campaign
- **Previous Campaign** (3): pdays, previous, poutcome
- **Economic Indicators** (5): emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
- **Engineered Features** (9): was_previously_contacted, campaign_intensity_group, age_group, economic_stress_index, has_any_loan, month_order, previous_contact_success_flag, contact_is_cellular, client_financial_pressure_flag

**Model File**: `models/benchmark_model_with_duration.joblib`

---

## Feature Set B (Realistic Business Model)
**28 features total** — excludes `duration`

### Purpose
- **Used for all predictions** in the Streamlit app
- Realistic — only uses information available **before** making the call
- What the bank actually uses to decide who to contact

### Features Included
Same as Feature Set A, but **without duration**:
- **Client Info** (7): age, job, marital, education, default, housing, loan
- **Contact Info** (4): contact, month, day_of_week, campaign
- **Previous Campaign** (3): pdays, previous, poutcome
- **Economic Indicators** (5): emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
- **Engineered Features** (9): was_previously_contacted, campaign_intensity_group, age_group, economic_stress_index, has_any_loan, month_order, previous_contact_success_flag, contact_is_cellular, client_financial_pressure_flag

**Model Files**:
- `models/realistic_model_without_duration.joblib`
- `models/champion_model_advanced_lgbm.joblib` (FLAML-tuned)
- `models/preprocessing_pipeline_set_b.joblib`

---

## Performance Comparison

| Metric | Feature Set A (with duration) | Feature Set B (no duration) |
|--------|------------------------------|----------------------------|
| ROC-AUC | ~0.93 | ~0.81 |
| PR-AUC | ~0.70 | ~0.49 |
| Use Case | Benchmark only | Production predictions |

---

## Quick Reference

```python
from src.config import FEATURE_SET_A_COLS, FEATURE_SET_B_COLS

print(f"Set A: {len(FEATURE_SET_A_COLS)} features (includes duration)")
print(f"Set B: {len(FEATURE_SET_B_COLS)} features (excludes duration)")

# Always use Set B for predictions
X_features = df[FEATURE_SET_B_COLS]
```

---

## Common Error

```
ValueError: X has 75 features, but ColumnTransformer is expecting 28 features
```

**Cause**: You're passing the entire DataFrame (with target, engineered features, and metadata) instead of just the 28 Feature Set B columns.

**Fix**: Select only Feature Set B columns:
```python
X_test = X_test[FEATURE_SET_B_COLS]  # Now has 28 features
X_test_transformed = pipeline.transform(X_test)
```
