# Contract: Batch Customer Prediction

**Interface**: Streamlit page "Batch Prediction" + `scripts/predict_batch.py` + `src/inference.predict_batch()`
**Date**: 2026-05-16

## Purpose

Given a CSV file of N customers (same schema as Single Prediction, minus `duration`),
return a CSV with predictions appended, ranked by subscription probability.

## Input CSV Schema

Same as Single Prediction input schema with these additional rules:
- Separator: `,` (standard CSV for upload; internal data uses `;`)
- Header row required
- `duration` column silently stripped with a displayed warning
- Target column `y` silently ignored if present

Minimum required columns (19 total, same as single prediction):
`age, job, marital, education, default, housing, loan, contact, month, day_of_week,
campaign, pdays, previous, poutcome, emp.var.rate, cons.price.idx, cons.conf.idx,
euribor3m, nr.employed`

## Output CSV Schema

All input columns are preserved. The following columns are appended:

| Column | Type | Description |
|--------|------|-------------|
| predicted_class | int | 0 or 1 |
| subscription_probability | float | P(y=1), range [0.0, 1.0] |
| rank | int | 1 = highest probability, N = lowest |

The output is sorted by `subscription_probability` descending (rank ascending).

## Error Responses

| Condition | Behaviour |
|-----------|-----------|
| Missing required columns | Return error listing missing columns; no predictions |
| Empty CSV (0 data rows) | Return error: "CSV must contain at least 1 data row" |
| Non-CSV file uploaded | Return error: "Please upload a CSV file" |
| > 50,000 rows | Display warning and process anyway (no hard limit in code) |
| Model file not found | Display st.error(); st.stop() |
