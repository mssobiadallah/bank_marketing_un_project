# Hypothesis Testing Report
**Dataset**: bank-additional-full.csv (N = 41,188)  
**Significance level**: α = 0.05  
**Date**: auto-generated  

## Results Summary
| ID | Hypothesis | Feature | Test | Statistic | p-value | Reject H₀ | Effect Size |
|----|-----------:|---------|------|----------:|--------:|:---------:|:-----------:|
| H1 | H1: Job type is associated with subscription | `job` | Chi-Square | 961.24 | 4.19e-199 | ✅ Yes | 0.1528 (Cramér's V) |
| H2 | H2: Education level is associated with subscription | `education` | Chi-Square | 193.11 | 3.31e-38 | ✅ Yes | 0.0685 (Cramér's V) |
| H3 | H3: Housing loan status is associated with subscription | `housing` | Chi-Square | 5.68 | 0.0583 | ❌ No | 0.0117 (Cramér's V) |
| H4 | H4: Previous campaign outcome is associated with subscription | `poutcome` | Chi-Square | 4230.52 | 0.00e+00 | ✅ Yes | 0.3205 (Cramér's V) |
| H5 | H5: Age differs between subscribers and non-subscribers | `age` | Mann-Whitney U | 82955833.50 | 0.0161 | ✅ Yes | 0.0216 (Rank-biserial r) |
| H6 | H6: Campaign contacts differ between subscribers and non-subscribers | `campaign` | Mann-Whitney U | 75428808.50 | 3.42e-38 | ✅ Yes | 0.1104 (Rank-biserial r) |
| H7 | H7: Economic indicators differ between subscribers and non-subscribers | `nr.employed` | Mann-Whitney U | 42565804.00 | 0.00e+00 | ✅ Yes | 0.4980 (Rank-biserial r) |

## Business Interpretations
### H1: job
Students and retired clients subscribe at significantly higher rates than blue-collar workers, making occupation a strong targeting signal.

### H2: education
University-educated clients are more likely to subscribe. Education level should inform segmentation strategy.

### H3: housing
The association between housing loan status and subscription is weak, suggesting it has limited targeting value alone.

### H4: poutcome
Clients with a previous successful campaign outcome subscribe at ~65% rate — over 5× the baseline. Prior success is the strongest categorical predictor.

### H5: age
Subscribers tend to be younger or older (bimodal), while non-subscribers cluster in the middle-aged working range. Age segmentation is useful.

### H6: campaign
Non-subscribers receive significantly more campaign contacts. Excessive contact is associated with lower conversion — diminishing returns are evident.

### H7: nr.employed
Low employment levels (economic downturn) are associated with higher subscription rates, confirming that macroeconomic context drives decisions.

