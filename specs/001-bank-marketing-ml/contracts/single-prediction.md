# Contract: Single Customer Prediction

**Interface**: Streamlit page "Predict New Client" + `src/inference.predict_single()`
**Date**: 2026-05-16

## Purpose

Given one customer's pre-call attributes (all features except `duration`), return:
- A predicted subscription class (yes/no)
- A subscription probability
- A risk/opportunity level
- Top 5 SHAP feature contributions

## Input Schema

All fields must be provided. `duration` is NEVER accepted by this contract.

| Field | Type | Example | Constraints |
|-------|------|---------|-------------|
| age | int | 35 | 17–98 |
| job | str | "technician" | One of 12 valid job categories + "unknown" |
| marital | str | "married" | married / single / divorced / unknown |
| education | str | "university.degree" | 8 education levels + unknown |
| default | str | "no" | yes / no / unknown |
| housing | str | "yes" | yes / no / unknown |
| loan | str | "no" | yes / no / unknown |
| contact | str | "cellular" | cellular / telephone |
| month | str | "may" | jan / feb / mar / apr / may / jun / jul / aug / sep / oct / nov / dec |
| day_of_week | str | "mon" | mon / tue / wed / thu / fri |
| campaign | int | 2 | >= 1 |
| pdays | int | 999 | 0–999; 999 = not previously contacted |
| previous | int | 0 | >= 0 |
| poutcome | str | "nonexistent" | success / failure / nonexistent |
| emp.var.rate | float | -1.8 | -3.4 to 1.4 |
| cons.price.idx | float | 93.2 | 92.2–94.8 |
| cons.conf.idx | float | -36.4 | -50.8 to -26.9 |
| euribor3m | float | 1.266 | 0.634–5.045 |
| nr.employed | float | 5099.1 | 4963.6–5228.1 |

## Output Schema

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| predicted_class | int | 1 | 0 = will not subscribe, 1 = will subscribe |
| subscription_probability | float | 0.73 | Probability of class 1, range [0.0, 1.0] |
| risk_level | str | "High" | "High" (>= 0.6), "Medium" (0.3–0.59), "Low" (< 0.3) |
| top_features | list | [...] | Top 5 [{feature: str, shap_value: float}] sorted by abs(shap_value) desc |
| model_version | str | "realistic_model_without_duration" | Model identifier |
| duration_excluded | bool | true | Always true in this contract |

## Error Responses

| Condition | Behaviour |
|-----------|-----------|
| Missing required field | Raise ValueError listing missing fields |
| `duration` present in input | Strip silently; log warning |
| Invalid categorical value | Pass through; OHE handles_unknown='ignore' |
| Model file not found | Raise FileNotFoundError with file path |
