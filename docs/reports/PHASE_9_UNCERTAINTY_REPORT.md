# Phase 9.2 — Prediction Uncertainty Quantification Report
## Academic Deliverable — Conformal Calibration & Empirical Coverage Diagnostics

**Project:** India Air Quality Analysis and Next-Day AQI Forecasting  
**Course:** `24UADPC302 — Predictive Modeling Using SPSS (IBM)`  
**Institution:** Sanjivani University, Kopargaon, Maharashtra  
**Department:** Third Year B.Tech, Artificial Intelligence & Data Science (AIDS), Semester 5  
**Faculty Guide:** Mr. Rohit Pawar  
**Student Team:**  
- Akshad Makhana (`2124UDSM1007`)  
- Shubham Pawar (`2124UDSM1074`)  
- Yash Kolhe (`2124UDSM1107`)  
- Pranav Hadole (`2124UDSM1108`)  

**Repository:** `C:\Users\KANHA\Desktop\SPSS PROJECT\`  
**Target Milestone:** Phase 9.2 (Prediction Uncertainty Quantification)  
**Status:** IMPLEMENTED, VERIFIED & AUDITED  
**Date:** October 2026  

---

## 1. Executive Summary & Objective

In environmental health forecasting and municipal decision-making, deterministic point forecasts ($\hat{y}_{t+1}$) are insufficient for risk-sensitive planning. Public health advisories, school closures, and odd-even vehicular restrictions require calibrated uncertainty bounds that quantify the likelihood of severe pollution excursions.

Phase 9.2 introduces an uncertainty-quantification layer for the frozen Random Forest next-day AQI forecasting champion (`models/random_forest_val_model.joblib`).

### Key Governance & Methodological Guarantees
1. **Model Immutability:** The frozen model artifact was evaluated in strictly read-only mode. No retraining, refitting, parameter tuning, or model replacement occurred.
2. **Chronological Calibration Barrier:** Calibration quantiles were calculated strictly from historical pre-2019 validation observations ($2018-07-01$ to $2018-12-31$, $N=2,982$). Observations from the $2019$ primary test ($N=6,201$) and $2020$ stress test ($N=3,124$) partitions were strictly quarantined and used exclusively for held-out empirical evaluation.
3. **No Target Tuning:** Test set outcomes were never accessed to adjust interval widths or choose quantiles. Empirical coverage is reported as an empirical measurement of model generalization under distribution shift, not a pre-tuned target.
4. **Preservation of Completed Milestones:** Phase 8 evaluation metrics ($2019$ test MAE $26.2993$, RMSE $55.6234$), the academic submission package (`submission/phase8_academic_submission/`), and Phase 9.1 SHAP explainability outputs were verified via SHA-256 hashes and remain identical to baseline.

---

## 2. Dataset Partitioning & Calibration Allocation

The repository maintains four strictly non-overlapping chronological partitions partitioned by `Target_Date` ($t+1$):

| Partition Name | Target Date Range | Shared-Cohort Rows ($N$) | Role in Phase 9.2 |
| :--- | :--- | :---: | :--- |
| **Historical Training** | 2015-01-02 to 2018-06-30 | $10,042$ | Pre-training baseline (estimator fitting partition) |
| **Designated Validation** | 2018-07-01 to 2018-12-31 | $2,982$ | **Calibration Partition ($D_{cal}$)** |
| **Primary Test** | 2019-01-01 to 2019-12-31 | $6,201$ | **Held-Out Generalization Evaluation ($D_{test}$)** |
| **Exogenous Stress Test** | 2020-01-01 to 2020-07-01 | $3,124$ | **Lockdown Structural Shock Evaluation ($D_{stress}$)** |

### Operational Calibration Protocol
- **Calibration Cohort ($D_{cal}$):** Sliced from `city_day_features.parquet` with filter `split == 'val'`, `is_target_valid == True`, `is_shared_cohort == True`.
- **Temporal Bounds:** Earliest `2018-07-01`, Latest `2018-12-31`.
- **Validation Residual Summary:**
  - Mean Absolute Error ($\text{MAE}_{cal}$): **$27.3559$ AQI points**
  - Mean Bias Error ($\text{MBE}_{cal} = \text{mean}(y - \hat{y})$): **$+2.7845$ AQI points** (slight historical underprediction)
  - Residual Standard Deviation: **$54.6710$ AQI points**

---

## 3. Mathematical Formulation & Quantile Computation

### A. Residual and Nonconformity Definition
Following project conventions, the prediction residual is defined as:
$$e_i = y_i - \hat{y}_i$$
where $y_i$ is the observed next-day AQI on date $t+1$, and $\hat{y}_i$ is the frozen Random Forest prediction.
- Positive residual ($e_i > 0$): Model underpredicted air pollution ($y_i > \hat{y}_i$).
- Negative residual ($e_i < 0$): Model overpredicted air pollution ($y_i < \hat{y}_i$).

The primary nonconformity score is the absolute residual:
$$s_i = |y_i - \hat{y}_i|$$

### B. Finite-Sample Conformal Quantile (Symmetric Baseline)
For a nominal coverage level $1 - \alpha$, standard inductive split conformal prediction specifies the finite-sample order-statistic rank:
$$k = \left\lceil (n + 1)(1 - \alpha) \right\rceil$$

For $n = 2,982$ sorted calibration scores $s_{(1)} \le s_{(2)} \le \dots \le s_{(n)}$:
- **Nominal 90% Interval ($\alpha = 0.10$):**
  $$k_{90} = \lceil (2982 + 1)(0.90) \rceil = \lceil 2684.7 \rceil = 2685 \implies \text{Index: } 2684$$
  $$\mathbf{q_{0.90} = 57.0621 \text{ AQI points}}$$
- **Nominal 95% Interval ($\alpha = 0.05$):**
  $$k_{95} = \lceil (2982 + 1)(0.95) \rceil = \lceil 2833.85 \rceil = 2834 \implies \text{Index: } 2833$$
  $$\mathbf{q_{0.95} = 90.3847 \text{ AQI points}}$$

### C. Interval Construction & Operational Bounding
1. **Physical Non-Negativity:** Ambient AQI cannot physically be negative. Any lower bound evaluated below zero is truncated at zero:
   $$\hat{y}_{i, lower}^{adj} = \max(0.0, \hat{y}_i - q_{1-\alpha})$$
2. **Unclipped Continuous Upper Bound:**
   $$\hat{y}_{i, upper} = \hat{y}_i + q_{1-\alpha}$$
   Continuous predictions are **strictly NOT clipped at 500**. The dataset contains valid physical and sensor readings extending to $1,672$ AQI points. Truncating intervals at 500 would mask model error and distort continuous uncertainty evaluation.

### D. Secondary Asymmetric Residual Quantile Method
To account for residual skewness (severe spikes produce larger underpredictions than overpredictions), asymmetric empirical quantiles were evaluated:
- **Nominal 90%:** $\Delta_{lower} = -58.8980$, $\Delta_{upper} = +55.4823$ AQI points.
- **Nominal 95%:** $\Delta_{lower} = -94.0852$, $\Delta_{upper} = +86.8513$ AQI points.
- Interval: $[\max(0.0, \hat{y}_i + \Delta_{lower}), \hat{y}_i + \Delta_{upper}]$.

---

## 4. Scientific Assumptions & Time-Series Non-Exchangeability

> [!WARNING]
> **Violation of Exchangeability in Atmospheric Time Series:**  
> Standard conformal prediction theory proves that marginal coverage $P(Y_{n+1} \in \hat{C}(X_{n+1})) \ge 1 - \alpha$ holds **if and only if** the calibration and test samples are exchangeable (e.g. i.i.d. draws from a stationary distribution).  
> **This assumption is violated in chronological air quality forecasting:**
> 1. **Temporal Serial Correlation:** Atmospheric persistence ensures that errors are autocorrelated ($e_{t+1}$ depends on $e_t$).
> 2. **Seasonal Distributional Shift:** Indo-Gangetic winter stagnation (November–January) exhibits fundamentally higher residual variance than monsoon washout (July–September).
> 3. **Inter-Annual Weather Variability:** Western disturbances and planetary boundary layer heights vary from year to year.
> 
> Consequently, the intervals presented here must be characterized as **conformal-style empirical calibration intervals**. Empirical coverage on held-out test data is an empirical diagnostic measurement, not a mathematically guaranteed future bound.

---

## 5. Calibration Reuse Limitation

The Designated Validation partition ($2018-07-01$ to $2018-12-31$) was previously utilized during:
1. **Phase 7 Model Selection:** Comparing candidate validation MAE across Ridge, Random Forest, and HistGradientBoosting to designate the champion.
2. **Phase 9.1 Explainability:** Evaluating global SHAP importance and local case attributions.

**Epistemic Disclosure:** Because the validation partition was used for model selection, the model's validation errors are slightly smaller than they would be on a completely virgin calibration set (selection bias). However, in this frozen project architecture:
- Retraining the model on an earlier sub-split would violate the frozen Phase 7 and Phase 8 baselines.
- Using the 2019 test set for calibration would contaminate the held-out evaluation barrier.
- Therefore, calibrating on the pre-2019 validation split is the **only scientifically defensible protocol** that preserves all frozen milestones while strictly preventing future data leakage.

---

## 6. Empirical Evaluation Results (2019 Test & 2020 Stress)

Both symmetric conformal and asymmetric empirical intervals were evaluated across the two quarantined future partitions:

### Overall Summary Table

| Evaluation Partition | Uncertainty Method | Nominal Coverage ($1-\alpha$) | Sample Count ($N$) | Empirical Coverage (%) | Mean Width (AQI Pts) | Median Width (AQI Pts) | Min Width (AQI Pts) | Max Width (AQI Pts) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2019 Primary Test** | **Symmetric Conformal** | **90%** | $6,201$ | **90.21%** | $113.90$ | $114.12$ | $100.30$ | $114.12$ |
| 2019 Primary Test | Symmetric Conformal | 95% | $6,201$ | **94.97%** | $175.08$ | $180.77$ | $133.62$ | $180.77$ |
| 2019 Primary Test | Asymmetric Empirical | 90% | $6,201$ | **90.36%** | $114.07$ | $114.38$ | $98.72$ | $114.38$ |
| 2019 Primary Test | Asymmetric Empirical | 95% | $6,201$ | **95.05%** | $174.12$ | $180.94$ | $130.09$ | $180.94$ |
| **2020 Stress Test** | **Symmetric Conformal** | **90%** | $3,124$ | **94.40%** | $113.69$ | $114.12$ | $99.47$ | $114.12$ |
| 2020 Stress Test | Symmetric Conformal | 95% | $3,124$ | **98.08%** | $172.10$ | $180.77$ | $132.79$ | $180.77$ |
| 2020 Stress Test | Asymmetric Empirical | 90% | $3,124$ | **94.43%** | $113.81$ | $114.38$ | $97.89$ | $114.38$ |
| 2020 Stress Test | Asymmetric Empirical | 95% | $3,124$ | **98.11%** | $170.68$ | $180.94$ | $129.26$ | $180.94$ |

### Scientific Analysis of Coverage Findings
1. **2019 Held-Out Generalization:**
   - On the primary 2019 test set ($N=6,201$), symmetric conformal intervals achieve **$90.21\%$ empirical coverage** at the nominal $90\%$ level, and **$94.97\%$ empirical coverage** at the nominal $95\%$ level.
   - This alignment indicates that despite non-exchangeability, the pre-2019 validation partition captured a representative spread of annual residual variance for 2019.
2. **2020 Stress Partition Over-Coverage:**
   - In the 2020 stress period ($N=3,124$), empirical coverage rises to **$94.40\%$** (nominal 90%) and **$98.08\%$** (nominal 95%).
   - *Observed Metric Difference:* Across the identical shared cohort (Cohort A), the model's observed MAE in the 2020 stress period was **$19.3288$ AQI points** (approximately **$19.33$**), compared with **$27.3559$ AQI points** (approximately **$27.36$**) in the 2018 validation calibration cohort (as documented in `reports/phase8_stress_metrics.csv`). Because the 2020 forecast errors were substantially smaller on average than those observed during the 2018 calibration period, fixed calibration intervals ($\pm 57.06$ AQI points for 90% and $\pm 90.38$ AQI points for 95%) empirically enclosed a higher proportion of observations ($94.40\%$ and $98.08\%$). The present predictive evaluation documents this metric divergence, but does not independently establish its physical or macroeconomic causes (such as lockdown emission restrictions).

---

## 7. Subgroup Diagnostics: City-Level & CPCB Severity Bands

While marginal coverage in 2019 is $90.21\%$, subgroup diagnostics reveal significant spatial and conditional heterogeneity:

### A. City-Level Coverage Breakdown (2019 Primary Test, Symmetric 90%)

| City | Region / Climate | Sample Count ($N$) | Empirical Coverage (%) | Mean Width (AQI Pts) | Methodological Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Hyderabad** | Southern Inland | $365$ | **100.00%** | $112.82$ | Stable meteorological regime; zero coverage failures |
| **Bengaluru** | Southern Plateau | $365$ | **99.73%** | $113.98$ | Consistent moderate air quality; high conservatism |
| **Thiruvananthapuram** | Southern Coastal | $347$ | **99.14%** | $113.18$ | Marine breeze ventilation; tight baseline errors |
| **Mumbai** | Western Coastal | $365$ | **99.18%** | $114.07$ | Coastal dispersion dampens extreme residual swings |
| **Amaravati** | Southeastern Inland | $303$ | **99.01%** | $113.91$ | Well-calibrated moderate error regime |
| **Visakhapatnam** | Eastern Coastal | $325$ | **96.31%** | $114.11$ | Well-bounded coastal regime |
| **Jaipur** | Semi-Arid Northwestern | $365$ | **95.62%** | $114.12$ | Good generalization across seasonal dust shifts |
| **Chennai** | Southeastern Coastal | $365$ | **95.07%** | $114.07$ | Monsoon coastal ventilation regime |
| **Kolkata** | Eastern Gangetic Delta | $365$ | **94.52%** | $113.13$ | Consistent winter/summer representation |
| **Brajrajnagar** | Eastern Industrial | $301$ | **93.69%** | $114.12$ | Industrial mining baseline captured |
| **Patna** | Middle Indo-Gangetic | $355$ | **93.24%** | $114.12$ | Heavy winter smog regime captured |
| **Amritsar** | Northwestern Plains | $359$ | **92.76%** | $114.02$ | Agricultural stubble burning season captured |
| **Lucknow** | Central Indo-Gangetic | $365$ | **90.68%** | $114.12$ | Perfectly matches nominal 90% baseline |
| **Jorapokhar** | Eastern Mining Cluster | $294$ | **89.46%** | $114.12$ | Minor under-coverage from industrial volatility |
| **Talcher** | Coal Mining Corridor | $287$ | **87.80%** | $114.12$ | Fugitive coal dust creates unmodeled spikes |
| **Delhi** | National Capital / IGP | $365$ | **85.21%** | $114.12$ | Acute winter inversion spikes exceed $\pm 57$ buffer |
| **Gurugram** | NCR Industrial/Urban | $363$ | **84.85%** | $114.08$ | NCR trans-boundary pollution spikes |
| **Ahmedabad** | Western Semi-Arid | $347$ | **27.09%** | $114.12$ | **Severe under-coverage due to localized CO sensor anomaly** |

### B. Coverage Breakdown by CPCB Hazard Category (2019 Test, Symmetric 90%)

| Statutory CPCB Bucket | Continuous AQI Range | Sample Count ($N$) | Empirical Coverage (%) | Mean Width (AQI Pts) | Diagnostic Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Good** | $[0, 50]$ | $287$ | **97.91%** | $110.91$ | Truncation at 0.0 prevents negative lower bounds |
| **Satisfactory** | $(50, 100]$ | $2,187$ | **98.86%** | $113.92$ | Stable baseline; low model error |
| **Moderate** | $(100, 200]$ | $2,263$ | **93.73%** | $114.12$ | Core distribution; well-calibrated |
| **Poor** | $(200, 300]$ | $667$ | **80.21%** | $114.12$ | Onset of winter inversion; increasing error |
| **Very Poor** | $(300, 400]$ | $460$ | **79.35%** | $114.12$ | Severe stagnation; model underestimates peaks |
| **Severe** | $> 400$ | $337$ | **38.58%** | $114.12$ | **Conditional under-coverage during acute hazardous spikes** |

---

## 8. Extreme Sensor Anomaly Diagnostic ($\text{AQI} > 500$)

The official CPCB National Air Quality Index scale terminates at 500 (Severe). However, continuous target measurements in the dataset exceed 500 during acute pollution and sensor anomaly episodes:

- **2019 Primary Test Cohort ($\text{AQI} > 500$):** $N = 150$ observations (primarily Ahmedabad CO anomaly and extreme Delhi winter events).
  - Empirical 90% Coverage: **$29.33\%$**
  - Mean Prediction Interval Width: **$114.12$ AQI points**
- **2020 Stress Test Cohort ($\text{AQI} > 500$):** $N = 17$ observations.
  - Empirical 90% Coverage: **$23.53\%$**
  - Mean Prediction Interval Width: **$114.12$ AQI points**

### Methodological Insight
This diagnostic confirms the theoretical prediction: **marginal conformal intervals provide population-level guarantees, not conditional guarantees for rare, extreme spikes**. When observed AQI reaches $800–1,600$ AQI points, a symmetric buffer of $\pm 57.06$ points around a tree-bounded forecast ($\approx 500–1,000$) cannot bridge the gap. In operational deployments, extreme episode forecasting requires specialized extreme-value theory (EVT) conformalization.

---

## 9. Visual Evidence

The generated visualization is saved at `reports/figures/prediction_intervals.png` ($503,640$ bytes):
- **Cohort Visualized:** Delhi (representative northern industrial city exhibiting high seasonal variance).
- **Time Window:** 90 consecutive calendar days from **2019-01-01 to 2019-03-31** (Winter peak to Spring transition).
- **Plot Features:**
  - Blue shaded ribbon: Symmetric 90% calibrated prediction interval ($\pm 57.06$ AQI points around $\hat{y}$).
  - Solid blue line: Random Forest next-day forecast $\hat{y}$.
  - Green circular points: Observed next-day AQI successfully captured within the prediction band.
  - Red triangular markers: Observed AQI spikes that breached the prediction interval.
  - Subgroup Coverage in Window: **$75.6\%$** ($N=90$), clearly demonstrating how winter episodic stagnation challenges marginal intervals.

---

## 10. Integrity Audit & Asset Preservation

To confirm that Milestone 9.2 operated in strict read-only isolation without mutating frozen assets, cryptographic SHA-256 hashes were calculated before and after execution:

| Asset Category | File Path | Baseline SHA-256 Hash | Post-Phase 9.2 SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Model Artifact** | `models/random_forest_val_model.joblib` | `ab7246042b6cb4...` | `ab7246042b6cb4...` | **MATCH** |
| **Phase 8 Metrics** | `reports/phase8_test_metrics.csv` | `16235c23925206...` | `16235c23925206...` | **MATCH** |
| **Phase 8 Metrics** | `reports/phase8_stress_metrics.csv` | `7e799719f3630e...` | `7e799719f3630e...` | **MATCH** |
| **Phase 8 Predictions** | `reports/phase8_test_predictions.csv` | `6d2e27fb81e033...` | `6d2e27fb81e033...` | **MATCH** |
| **Phase 8 Predictions** | `reports/phase8_stress_predictions.csv` | `1622f1090db5fc...` | `1622f1090db5fc...` | **MATCH** |
| **Phase 8 Report** | `docs/reports/PHASE_8_EVALUATION_REPORT.md` | `759b00044baaf6...` | `759b00044baaf6...` | **MATCH** |
| **Phase 9.1 Engine** | `src/models/interpretability.py` | `0dcfa8993c5b45...` | `0dcfa8993c5b45...` | **MATCH** |
| **Phase 9.1 Tests** | `tests/test_interpretability.py` | `50803d53c450df...` | `50803d53c450df...` | **MATCH** |
| **Phase 9.1 CSV** | `reports/shap_feature_importance.csv` | `6c7e3077a08d4a...` | `6c7e3077a08d4a...` | **MATCH** |
| **Phase 9.1 Report** | `docs/reports/PHASE_9_INTERPRETABILITY_REPORT.md` | `51ffae4835213b...` | `51ffae4835213b...` | **MATCH** |
| **Academic Package** | `submission/phase8_academic_submission/` | 37 files verified | 37 files verified | **MATCH** |

---

## 11. Deliverables & Test Verification Matrix

### Created File Deliverables

| Deliverable Path | Nature | File Size | Description |
| :--- | :---: | :---: | :--- |
| `src/models/uncertainty.py` | Created | $16,423$ bytes | Uncertainty calibration & interval evaluation module |
| `tests/test_uncertainty.py` | Created | $8,842$ bytes | 13 automated unit and integration tests |
| `reports/phase9_uncertainty_metrics.csv` | Created | $826$ bytes | Test and stress empirical coverage and width metrics |
| `reports/phase9_city_coverage.csv` | Created | $2,410$ bytes | 18-city empirical coverage table for 2019 and 2020 |
| `reports/figures/prediction_intervals.png` | Created | $503,640$ bytes | Publication-grade interval time-series ribbon plot |
| `docs/reports/PHASE_9_UNCERTAINTY_REPORT.md` | Created | Document | Comprehensive academic deliverable |

### Test Suite Execution Summary
- **Focused Uncertainty Test Suite:** `pytest tests/test_uncertainty.py -v`  
  - Result: **13 passed in 7.57s** (Exit Code: `0`).
- **Full Repository Test Suite:** `pytest tests/ -v`  
  - Result: **74 passed in 20.02s** (Exit Code: `0`).  
  - Breakdown: 51 legacy tests + 10 SHAP interpretability tests + 13 uncertainty quantification tests. Zero failures, zero skipped.
- **Git Formatting Check:** `git diff --check`  
  - Result: **Exit Code 0** (No whitespace defects or line-ending errors).

---
*End of Phase 9.2 Prediction Uncertainty Quantification Report. Ready for Faculty Guide review.*
