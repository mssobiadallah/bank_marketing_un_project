# Contract: Inference Module Schema

**Interface**: `src/inference.py` — internal Python API
**Date**: 2026-05-16

## Functions

### predict_single(model, pipeline, input_dict) -> CustomerPrediction

Validates input, applies preprocessing pipeline, runs model, computes SHAP values.

**Parameters**:
- `model`: fitted sklearn-compatible classifier
- `pipeline`: fitted ColumnTransformer preprocessing pipeline
- `input_dict`: dict mapping field names to values (see Single Prediction input schema)

**Returns**: `CustomerPrediction` dict (see data-model.md Entity 7)

**Raises**:
- `ValueError` if required fields missing
- `ValueError` if schema mismatch (e.g., duration present)

---

### predict_batch(model, pipeline, df) -> pd.DataFrame

Validates schema, applies preprocessing, runs model, returns predictions sorted by probability.

**Parameters**:
- `model`: fitted sklearn-compatible classifier
- `pipeline`: fitted ColumnTransformer preprocessing pipeline
- `df`: pd.DataFrame with same schema as batch input CSV

**Returns**: `pd.DataFrame` with all input columns + `predicted_class`, `subscription_probability`, `rank`

**Raises**:
- `ValueError` if required columns missing (lists them)
- `ValueError` if DataFrame is empty

---

### validate_input_schema(input_df, feature_set) -> list[str]

Returns a list of validation error messages. Empty list = valid.

**Parameters**:
- `input_df`: pd.DataFrame or dict
- `feature_set`: "set_a" (with duration) or "set_b" (without duration)

**Returns**: `list[str]` — empty if valid, otherwise error messages

---

### load_model(path) -> tuple[classifier, pipeline]

Loads model and pipeline from .joblib files. Raises FileNotFoundError with path if missing.

---

## Feature Set B — Required Columns (Realistic Business Model)

The 19 pre-call features that must be present for any inference:

```
age, job, marital, education, default, housing, loan, contact, month, day_of_week,
campaign, pdays, previous, poutcome, emp.var.rate, cons.price.idx, cons.conf.idx,
euribor3m, nr.employed
```

`duration` is never in this list and must be rejected or stripped.

## Feature Set A — Required Columns (Benchmark Model)

Same as Feature Set B plus:
```
duration
```
