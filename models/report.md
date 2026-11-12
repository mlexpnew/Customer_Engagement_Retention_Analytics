### RetainLens model report (8000 train / 2000 held-out customers, 0 rejected by validation)

| Candidate (5-fold CV on train) | ROC-AUC |
|---|---|
| logistic_regression | 0.780 ± 0.020 |
| random_forest | 0.857 ± 0.011 |
| gradient_boosting (selected) | 0.857 ± 0.006 |

| Held-out metric | Value |
|---|---|
| ROC-AUC | 0.864 |
| PR-AUC (base rate 20.3%) | 0.708 |
| Brier score (calibrated) | 0.101 |
| Top 10% riskiest: share who actually left | 82.0% (4.03x the base rate) |
| Top 10% riskiest: share of all leavers caught | 40.3% |

| Calibration bin | Customers | Predicted | Observed |
|---|---|---|---|
| 0.0-0.1 | 953 | 0.05 | 0.04 |
| 0.1-0.2 | 385 | 0.14 | 0.12 |
| 0.2-0.3 | 257 | 0.25 | 0.24 |
| 0.3-0.4 | 114 | 0.35 | 0.43 |
| 0.4-0.5 | 52 | 0.45 | 0.40 |
| 0.5-0.6 | 40 | 0.56 | 0.62 |
| 0.6-0.7 | 41 | 0.65 | 0.54 |
| 0.7-0.8 | 34 | 0.75 | 0.77 |
| 0.8-0.9 | 43 | 0.85 | 0.86 |
| 0.9-1.0 | 81 | 0.97 | 0.96 |

**Retention budget test** (EUR 20,000 on held-out customers, realised with actual outcomes)

| Strategy | Contacted | Real leavers reached | Net value (EUR) |
|---|---|---|---|
| retainlens_expected_value | 500 | 239 | 406,848 |
| highest_churn_probability | 500 | 287 | 360,877 |
| highest_balance | 500 | 112 | 223,596 |
| random | 500 | 95 | 111,726 |

**Fairness**: gender is not a model input. Using it would add +0.0021 CV ROC-AUC.

| Gender | Customers | Churn rate | ROC-AUC | Flagged | Leavers caught |
|---|---|---|---|---|---|
| Female | 861 | 24.7% | 0.86 | 34.0% | 75.6% |
| Male | 1139 | 17.0% | 0.865 | 28.4% | 77.8% |

| Geography | Customers | Churn rate | ROC-AUC | Flagged | Leavers caught |
|---|---|---|---|---|---|
| France | 1021 | 16.7% | 0.848 | 22.9% | 68.8% |
| Germany | 503 | 31.4% | 0.856 | 53.3% | 88.0% |
| Spain | 476 | 16.6% | 0.861 | 23.9% | 70.9% |

| Top churn drivers (mean risk reduction if fixed) | |
|---|---|
| Product holding is a churn signal | 0.091 |
| Life-stage segment with high churn | 0.061 |
| Inactive in the last period | 0.047 |
| Market with elevated churn | 0.026 |
| Balance pattern linked to leaving | 0.020 |
| Credit score band with higher churn | 0.008 |
| Tenure stage with higher churn | -0.001 |

**Drift check**: held-out month `ok`, simulated shifted month `act` (worst PSI 2.18, retrain recommended).
