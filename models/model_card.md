# Model Card: Realistic Business Model — Bank Term Deposit Subscription

**Date**: 2026-05-16
**Version**: 1.0
**Feature**: [spec.md](../specs/001-bank-marketing-ml/spec.md)

---

## Model Name

HistGradientBoostingClassifier (Realistic Business Model — Feature Set B)

## Model Type

Histogram-based Gradient Boosting Classifier (scikit-learn `HistGradientBoostingClassifier`)
with `class_weight='balanced'` for handling class imbalance.

---

## Intended Use

This model is designed for **pre-campaign customer ranking**. Given a list of clients to contact,
the model assigns each a probability of subscribing to a term deposit. Clients are ranked from
highest to lowest probability so campaign agents can prioritise their outreach.

**Not intended for**:
- Real-time trading or financial decisions
- Automated approval or rejection without human review
- Any use case outside bank marketing campaigns

---

## Target Variable

Binary classification: `y` — "Will the client subscribe to a term deposit?"
- `1` → Yes (subscribed)
- `0` → No (did not subscribe)

---

## Dataset

| Property | Value |
|----------|-------|
| Name | `bank-additional-full.csv` |
| Source | UCI ML Repository — Bank Marketing Dataset |
| Rows | 41,188 |
| Features used | 19 (Feature Set B — no `duration`) |
| Train/test split | 80% / 20% (stratified) |
| Positive class rate | ~11.3% |

---

## Features Used (Feature Set B — No Duration)

### Raw Features (18)
| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Client age in years |
| job | Categorical | Type of job |
| marital | Categorical | Marital status |
| education | Categorical | Education level |
| default | Categorical | Has credit in default? |
| housing | Categorical | Has housing loan? |
| loan | Categorical | Has personal loan? |
| contact | Categorical | Contact communication type |
| month | Categorical | Last contact month of year |
| day_of_week | Categorical | Last contact day of the week |
| campaign | Numeric | Contacts during this campaign |
| pdays | Numeric | Days since last contact (999 = not contacted) |
| previous | Numeric | Previous campaign contacts |
| poutcome | Categorical | Previous campaign outcome |
| emp.var.rate | Numeric | Employment variation rate (quarterly) |
| cons.price.idx | Numeric | Consumer price index (monthly) |
| cons.conf.idx | Numeric | Consumer confidence index (monthly) |
| euribor3m | Numeric | Euribor 3-month rate (daily) |
| nr.employed | Numeric | Number of employees (quarterly) |

### Engineered Features (9)
| Feature | Description |
|---------|-------------|
| was_previously_contacted | 1 if pdays ≠ 999 |
| campaign_intensity_group | low / medium / high |
| age_group | young / middle / senior |
| economic_stress_index | euribor3m + emp.var.rate |
| has_any_loan | housing==yes OR loan==yes |
| month_order | integer 1–12 |
| previous_contact_success_flag | poutcome==success |
| contact_is_cellular | contact==cellular |
| client_financial_pressure_flag | default==yes OR has_any_loan |

---

## Features Excluded

| Feature | Reason |
|---------|--------|
| `duration` | **Data leakage** — only known *after* the call is completed. Including it would make the model appear accurate but be useless in practice. |

---

## Performance Metrics (Test Set)

*Filled after running `python scripts/train.py` or `notebooks/08_model_selection_explainability.ipynb`*

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8164 |
| PR-AUC (average_precision) | 0.4975 |
| Accuracy | — |
| Balanced Accuracy | — |
| Precision (@ opt. threshold) | — |
| Recall (@ opt. threshold) | — |
| F1 (@ opt. threshold) | — |
| Top-decile lift | — |

**Optimal threshold**: *fill after running notebook 08*

---

## Limitations

1. **No `duration` feature** — the model cannot use call length since it's unknown before contacting.
2. **Class imbalance** (~11.3% positive) — even with `class_weight='balanced'`, some false negatives are inevitable.
3. **Temporal validity** — the dataset covers 2008–2013. Economic conditions have changed; model may need retraining.
4. **Geography** — trained on a Portuguese bank dataset; results may not generalise to other countries.
5. **Contact bias** — only customers who were contacted are in the dataset; customers never contacted are absent.
6. **Not production-hardened** — no API serving, monitoring, drift detection, or A/B testing infrastructure.

---

## Ethical Considerations

- **Fairness**: The model uses `age`, `job`, `marital`, and `education` which may correlate with protected characteristics. Fairness analysis across demographic groups is recommended before deployment.
- **Transparency**: SHAP explanations are provided per prediction to support agent understanding.
- **Human oversight**: This model is a ranking tool. Human agents make the final call decision.
- **Bias amplification**: Focusing calls on predicted high-probability customers may reinforce existing biases in who is contacted.
- **Data privacy**: Customer data must be handled in compliance with GDPR and applicable banking regulations.

---

## Training Infrastructure

- **Language**: Python 3.12
- **Framework**: scikit-learn 1.8.0
- **Hardware**: Local macOS machine (CPU only)
- **Training time**: ~60 seconds for all baseline models

---

## How to Reproduce

```bash
git checkout 001-bank-marketing-ml
make install
make train
# or
python scripts/train.py
```

---

## References

- Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems, 62, 22–31.
- scikit-learn HistGradientBoostingClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
