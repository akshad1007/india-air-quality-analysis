# Phase 8 Acceptance Audit: Final Model Evaluation & Held-Out Validation

## India Air Quality Analysis and Next-Day AQI Forecasting

- **Document Identifier:** `docs/review/PHASE_8_ACCEPTANCE_AUDIT.md`
- **Audit Date:** 2026-10-25
- **Academic Course:** `24UADPC302: Predictive Modeling Using SPSS (IBM)`
- **Department & Institution:** TY AIDS, Sanjivani University, Kopargaon, Maharashtra
- **Faculty Guide:** Mr. Rohit Pawar
- **Project Team:**
  1. Akshad Makhana (`2124UDSM1007`)
  2. Shubham Pawar (`2124UDSM1074`)
  3. Yash Kolhe (`2124UDSM1107`)
  4. Pranav Hadole (`2124UDSM1108`)
- **Repository Remote:** `https://github.com/akshad1007/india-air-quality-analysis.git`
- **Current Git Branch:** `main` tracking `origin/main` at commit `4c98443`
- **Monetary Reporting Convention:** Indian Rupee (`₹`, `INR`), adhering to Indian numbering conventions (lakhs, crores). *Note: No financial costs, cloud computing budgets, or salaries were measured for this modeling evaluation; no artificial financial amounts are invented.*
- **Statistical Units Convention:** MAE, RMSE, and MBE are measured strictly in **AQI points**. $R^2$, Accuracy, Macro F1, and Balanced Accuracy are dimensionless statistical quantities.
- **Audit Type:** Independent Evidence-Based Acceptance Audit
- **Milestone Under Audit:** Phase 8 Final Model Evaluation & Held-Out Validation
- **Audit Final Verdict:** **ACCEPTED**

---

## 1. Audit Scope & Objective

This audit constitutes an independent, empirical verification of the deliverables, artifacts, execution logs, and codebase produced under **Phase 8 (Final Model Evaluation & Held-Out Validation)**. 

In strict accordance with academic rigor, quality-first principles, and faculty guidelines:
- No models were retuned or modified.
- No model selection decisions were retroactively altered based on test-set metrics.
- All metrics and cohort sample counts were independently recalculated from raw prediction records.
- Claims in the Phase 8 report were treated as hypotheses to verify, not established ground truth.

---

## 2. Actual Files and Artifacts Inspected

| Category | File Path | Status | Verification Summary |
| :--- | :--- | :---: | :--- |
| **Evaluation Engine** | `src/models/evaluation.py` | Present | Implements CPCB category mapping, classification diagnostics, residual aggregations, and sensitivity runner. |
| **Model Pipelines** | `src/models/pipelines.py` | Present | Defines preprocessing column transformers, encoders, and model architectures. |
| **Baseline Architecture** | `src/models/baseline.py` | Present | Implements zero-leakage Naive Persistence ($\widehat{\text{AQI}}_{t+1} = \text{AQI}_t$). |
| **Evaluation Metrics** | `src/models/metrics.py` | Present | Implements MAE, RMSE, $R^2$, and MBE calculations. |
| **Test Suite** | `tests/test_evaluation.py` | Present | 13 automated tests covering cohort sample sizes, boundaries, categories, and reproducibility. |
| **Model Artifacts** | `models/random_forest_val_model.joblib`<br>`models/ridge_regression_val_model.joblib`<br>`models/histgradientboosting_val_model.joblib` | Present | Frozen Phase 7 model pipelines (fitted strictly on training fold through `2018-06-30`). |
| **Test Predictions** | `reports/phase8_test_predictions.csv` | Present | 6,286 rows with city, target date, ground truth, and predictions for 2019 test set. |
| **Stress Predictions**| `reports/phase8_stress_predictions.csv` | Present | 3,163 rows with city, target date, ground truth, and predictions for 2020 stress set. |
| **Summary Metrics** | `reports/phase8_test_metrics.csv`<br>`reports/phase8_stress_metrics.csv`<br>`reports/phase8_classification_metrics.csv`<br>`reports/phase8_ahmedabad_sensitivity.csv` | Present | Machine-readable CSV summaries of regression, classification, and sensitivity metrics. |
| **Report Deliverable** | `docs/reports/PHASE_8_EVALUATION_REPORT.md` | Present | Comprehensive academic final evaluation report. |

---

## 3. Independent Recalculation of Regression Metrics & Cohort Counts

All metrics were recalculated directly from `reports/phase8_test_predictions.csv` and `reports/phase8_stress_predictions.csv` using `scikit-learn` and `numpy`.

### 3.1 Cohort Integrity and Key Uniqueness Checks
1. **Composite Key Uniqueness:** Evaluated duplicate check on `(City, Target_Date)`.
   - 2019 Test: `0` duplicates found across all 6,286 rows (**PASS**).
   - 2020 Stress: `0` duplicates found across all 3,163 rows (**PASS**).
2. **Temporal Alignment:** For $100\%$ of evaluated rows, `Target_Date` is strictly $1$ calendar day after observation `Date` ($\Delta D = 1\text{ day}$) (**PASS**).
3. **Cohort A Identical Row Set:** Evaluated persistence and all three models on exactly the same $6,201$ rows in 2019 and $3,124$ rows in 2020 with zero missing values (**PASS**).

### 3.2 Primary 2019 Test Set Recalculation ($N=6,201$)

| Model | Reported MAE | Recalculated MAE | Reported RMSE | Recalculated RMSE | Reported $R^2$ | Recalculated $R^2$ | Reported MBE | Recalculated MBE ($\widehat{y} - y$) | Recalculated MBE ($y - \widehat{y}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | $31.0419$ | **$31.0419$** | $70.2360$ | **$70.2360$** | $0.7445$ | **$0.7445$** | $+0.3614$ | **$+0.3614$** | **$-0.3614$** |
| **Ridge Regression** | $29.0261$ | **$29.0261$** | $55.8918$ | **$55.8918$** | $0.8382$ | **$0.8382$** | $+2.8414$ | **$+2.8414$** | **$-2.8414$** |
| **Random Forest** | $26.2993$ | **$26.2993$** | $55.6234$ | **$55.6234$** | $0.8398$ | **$0.8398$** | $+4.8666$ | **$+4.8666$** | **$-4.8666$** |
| **HistGradientBoosting** | $27.1402$ | **$27.1402$** | $54.9397$ | **$54.9397$** | $0.8437$ | **$0.8437$** | $+5.6109$ | **$+5.6109$** | **$-5.6109$** |

*Recalculated Error Reductions vs Persistence on Cohort A:*
- **Random Forest:** MAE improvement = $\frac{31.0419 - 26.2993}{31.0419} \times 100\% = \mathbf{+15.28\%}$; RMSE improvement = $\frac{70.2360 - 55.6234}{70.2360} \times 100\% = \mathbf{+20.81\%}$.
- **HistGradientBoosting:** MAE improvement = $\mathbf{+12.57\%}$; RMSE improvement = $\mathbf{+21.78\%}$.
- **Ridge Regression:** MAE improvement = $\mathbf{+6.49\%}$; RMSE improvement = $\mathbf{+20.42\%}$.

### 3.3 2020 Stress Period Recalculation ($N=3,124$)

| Model | Reported MAE | Recalculated MAE | Reported RMSE | Recalculated RMSE | Reported $R^2$ | Recalculated $R^2$ | Reported MBE | Recalculated MBE ($\widehat{y} - y$) | Recalculated MBE ($y - \widehat{y}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | $21.6306$ | **$21.6306$** | $45.9400$ | **$45.9400$** | $0.7334$ | **$0.7334$** | $+0.6421$ | **$+0.6421$** | **$-0.6421$** |
| **Ridge Regression** | $22.5458$ | **$22.5458$** | $40.3840$ | **$40.3840$** | $0.7940$ | **$0.7940$** | $+7.6277$ | **$+7.6277$** | **$-7.6277$** |
| **Random Forest** | $19.3288$ | **$19.3288$** | $39.1893$ | **$39.1893$** | $0.8060$ | **$0.8060$** | $+4.5243$ | **$+4.5243$** | **$-4.5243$** |
| **HistGradientBoosting** | $21.0474$ | **$21.0474$** | $39.1379$ | **$39.1379$** | $0.8065$ | **$0.8065$** | $+7.7221$ | **$+7.7221$** | **$-7.7221$** |

*Recalculated Error Reductions vs Persistence on Cohort A:*
- **Random Forest:** MAE improvement = $\mathbf{+10.64\%}$; RMSE improvement = $\mathbf{+14.69\%}$.
- **HistGradientBoosting:** MAE improvement = $\mathbf{+2.70\%}$; RMSE improvement = $\mathbf{+14.81\%}$.
- **Ridge Regression:** MAE change = $\mathbf{-4.23\%}$ (loss); RMSE improvement = $\mathbf{+12.09\%}$.

### 3.4 Audit Finding on Mean Bias Error (MBE) Sign Convention
> [!NOTE]
> **Audit Distinction — MBE Sign Convention:**
> In `src/models/metrics.py` line 52, Mean Bias Error is implemented as:
> $$\text{MBE}_{\text{code}} = \frac{1}{N} \sum_{i=1}^N (\widehat{y}_i - y_i)$$
> Under this implementation, positive values signify that the model predicts higher than ground truth on average (overprediction).
> In the Phase 8 markdown report narrative, the text defined $e_i = y_i - \widehat{y}_i$ where positive indicates underprediction.
> The reported values ($+4.8666$ for RF in 2019, $+4.5243$ in 2020) correspond to $\text{mean}(\widehat{y} - y)$, demonstrating that the model on average slightly overpredicts next-day AQI. When expressed as $y - \widehat{y}$, the values are $-4.8666$ and $-4.5243$. Both representations agree on the physical finding: models predict slightly higher pollutant levels than observed.

---

## 4. Audit of Evaluation Boundaries & Chronological Protocol

1. **Temporal Boundaries:**
   - 2019 Primary Test: target dates strictly in `[2019-01-01, 2019-12-31]`. Minimum target date = `2019-01-01`, Maximum target date = `2019-12-31` (**PASS**).
   - 2020 Stress Period: target dates strictly in `[2020-01-01, 2020-07-01]`. Minimum target date = `2020-01-01`, Maximum target date = `2020-07-01` (**PASS**).
2. **Reference Sample Counts:**
   - 2019 Test Cohort A: Exactly $6,201$ shared rows (**PASS**).
   - 2019 Test Cohort B: Exactly $6,286$ target-valid rows (**PASS**).
   - 2020 Stress Cohort A: Exactly $3,124$ shared rows (**PASS**).
   - 2020 Stress Cohort B: Exactly $3,163$ target-valid rows (**PASS**).
3. **No Temporal Lookahead Leakage:** Features at origin date $t$ are derived strictly prior to or on calendar date $t$. Target AQI is indexed at $t+1$. Training parameters in the loaded models were fitted strictly on dates $\le 2018-06-30$ (**PASS**).

---

## 5. Audit of CPCB Category Mapping & Classification Diagnostics

### 5.1 Category Boundary Implementation Check
The function `map_aqi_to_cpcb_bucket` in `src/models/evaluation.py` enforces:
- $\le 50.0$: `Good`
- $(50.0, 100.0]$: `Satisfactory`
- $(100.0, 200.0]$: `Moderate`
- $(200.0, 300.0]$: `Poor`
- $(300.0, 400.0]$: `Very Poor`
- $> 400.0$: `Severe`

### 5.2 Edge-Case & Boundary Stress Testing
Tested via Python execution:
- Boundary points: $50.0 \to \text{Good}$, $50.1 \to \text{Satisfactory}$, $50.5 \to \text{Satisfactory}$ (**PASS**).
- Boundary points: $100.0 \to \text{Satisfactory}$, $100.1 \to \text{Moderate}$ (**PASS**).
- Out-of-scale negative input ($-5.0 \to \text{Good}$) (**PASS**).
- Out-of-scale excessive input ($550.0 \to \text{Severe}$, $\infty \to \text{Severe}$) (**PASS**).
- Missing values ($\text{NaN} \to \text{None}$) (**PASS**).

### 5.3 Classification Metrics Recalculation (Cohort A, Random Forest)

| Metric | 2019 Test (Reported) | 2019 Test (Audited) | 2020 Stress (Reported) | 2020 Stress (Audited) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Macro F1** | $0.6678$ | **$0.6678$** | $0.6865$ | **$0.6865$** | **PASS** |
| **Balanced Accuracy** | $0.6545$ | **$0.6545$** | $0.6791$ | **$0.6791$** | **PASS** |
| **Overall Accuracy** | $74.36\%$ | **$74.36\%$** | $76.76\%$ | **$76.76\%$** | **PASS** |
| **Confusion Matrix Size** | $6 \times 6$ | **$6 \times 6$** | $6 \times 6$ | **$6 \times 6$** | **PASS** |

*Per-Class Support in 2019 Test ($N=6,201$):*
- Good: $287$
- Satisfactory: $2,187$
- Moderate: $2,263$
- Poor: $667$
- Very Poor: $460$
- Severe: $337$
*Total Support:* $287 + 2187 + 2263 + 667 + 460 + 337 = 6,201$ (Exact match, **PASS**).

---

## 6. Audit of Residuals, Subgroups & Ahmedabad Sensitivity

### 6.1 Seasonal Coverage Check
- Winter ($N=1,605$)
- Pre-Monsoon / Summer ($N=1,573$)
- Monsoon ($N=1,994$)
- Post-Monsoon / Autumn ($N=1,029$)
*Total:* $1,605 + 1,573 + 1,994 + 1,029 = 6,201$.  
The seasons are mutually exclusive, exhaustive, and account for $100\%$ of test rows (**PASS**).

### 6.2 Ahmedabad Sensitivity Check & Exact Error Contribution
- **Excluded City Count:** Ahmedabad has $347$ rows in 2019 Cohort A. The remaining 17 cities comprise exactly $5,854$ rows ($6,201 - 347 = 5,854$) (**PASS**).
- **Recalculated Metrics (Without Ahmedabad, $N=5,854$):**
  - Persistence: MAE = $23.3264$, RMSE = $37.4593$, $R^2 = 0.8330$.
  - Random Forest: MAE = $19.2984$ ($+17.27\%$ vs persistence), RMSE = $29.9717$ ($+19.99\%$ vs persistence), $R^2 = \mathbf{0.8931}$.
- **Exact Squared Error Contribution:**
  $$\sum_{\text{Ahmedabad}} (y_i - \widehat{y}_i)^2 = 13,926,980.55$$
  $$\sum_{\text{All 18 Cities}} (y_i - \widehat{y}_i)^2 = 19,185,641.08$$
  $$\text{Contribution} = \frac{13,926,980.55}{19,185,641.08} \times 100\% = \mathbf{72.59\%}$$
  *(The Phase 8 report noted Ahmedabad contributed "over 46% of total squared error"; the exact audited figure is **72.59%**, highlighting that nearly three-quarters of all prediction error variance in the test set originated from this single municipality).*

---

## 7. Audit of Model-Selection Rationale

The audit verified whether calling Random Forest the "champion" was methodologically justified:
1. **Pre-declared Validation Criterion:** In Phase 7 (`docs/reports/PHASE_7_MODELING_REPORT.md`), Random Forest was selected because it achieved the lowest validation MAE ($27.3559$), lowest validation RMSE ($59.7368$), and highest validation $R^2$ ($0.8774$).
2. **Test-Set Metric Trade-Off:** On the 2019 test set:
   - Random Forest achieved the lowest MAE ($26.2993$ vs HGB $27.1402$).
   - HistGradientBoosting achieved a slightly lower RMSE ($54.9397$ vs RF $55.6234$).
   - Random Forest achieved distinctly higher classification performance (Macro F1 $0.6678$ vs HGB $0.6057$, Balanced Accuracy $0.6545$ vs HGB $0.6124$).
3. **Verdict on Model Selection:** Random Forest's status as champion reflects strict fidelity to the predeclared Phase 7 selection criterion, not post-hoc re-selection. The report appropriately noted that HGB had marginally lower RMSE. No retrospective alterations were made (**PASS**).

---

## 8. Artifact Reproducibility & Pipeline Inference Audit

To verify prediction reproducibility, all three serialized models in `models/` were reloaded and evaluated on the test set features:

```python
diff_rf = np.max(np.abs(saved_rf_preds - newly_inferred_rf_preds))
diff_ridge = np.max(np.abs(saved_ridge_preds - newly_inferred_ridge_preds))
diff_hist = np.max(np.abs(saved_hist_preds - newly_inferred_hist_preds))
```

- **Random Forest Difference:** `0.0000000000` (**PASS**)
- **Ridge Regression Difference:** `0.0000000000` (**PASS**)
- **HistGradientBoosting Difference:** `0.0000000000` (**PASS**)

All predictions in `reports/phase8_test_predictions.csv` and `reports/phase8_stress_predictions.csv` are $100\%$ bit-for-bit reproducible from the saved model artifacts.

---

## 9. Automated Test Suite Audit

The full pytest suite was executed:
```powershell
pytest tests/ -v
```

- **Exit Code:** `0`
- **Total Tests Collected:** `51`
- **Passed:** `51 (100%)`
- **Failed / Skipped:** `0 / 0`
- **Execution Time:** `16.34 seconds`

The test suite validates data schemas, preprocessing immutability, feature alignment, lack of lookahead leakage, cross-validation chronological boundaries, CPCB boundary handling, and numerical reproducibility.

---

## 10. Git Status & Safety Restrictions Audit

- **Current Branch:** `main` tracking `origin/main` at commit `4c98443`.
- **Git Commits / Pushes:** `0` commits made; `0` pushes made.
- **Git Index:** No files staged (`git status` shows nothing added to commit).
- **Working Tree Integrity:** All source files, processed data, and Phase 6–7 artifacts are intact and untouched.

---

## 11. Classification of Audit Findings

### PASS Findings
1. **[PASS] Strict Model Configuration Freeze:** Phase 7 models were loaded and evaluated without retraining or retuning.
2. **[PASS] Independent Metric Recalculation:** All regression metrics (MAE, RMSE, $R^2$, MBE) and percentage improvements match saved outputs to 4 decimal places.
3. **[PASS] Sample Count Reconciliations:** Exact sample sizes ($6,201$ shared test, $3,124$ shared stress, $6,286$ valid test, $3,163$ valid stress) verified.
4. **[PASS] Bit-for-Bit Inference Reproducibility:** Re-running inference on model artifacts produces identical outputs ($\Delta = 0.000000$).
5. **[PASS] CPCB Category Handling:** Boundaries, out-of-scale values, and NaNs are handled explicitly and correctly.
6. **[PASS] Test Suite Execution:** 51 of 51 tests pass cleanly in $16.34$ seconds.
7. **[PASS] Git Safety Compliance:** No commits or pushes were executed.

### WARNING Findings (Non-Blocking Documentation Clarifications)
1. **[WARNING] MBE Sign Convention Definition in Report:** `src/models/metrics.py` implements $\text{MBE} = \frac{1}{N}\sum(\widehat{y}_i - y_i)$, yielding positive values ($+4.87$) for overprediction. The markdown report text stated the formula as $y_i - \widehat{y}_i$ while presenting positive numbers. This report clarifies the distinction: the numbers reflect overprediction ($\widehat{y} > y$).
2. **[WARNING] Ahmedabad Squared Error Contribution:** The report stated Ahmedabad contributes "over 46% of total squared error"; the exact audited figure is **72.59%**.
3. **[WARNING] Non-Causal Contextual Phrasing:** The report notes lockdown impacts and inversion effects. These should be treated as descriptive associations and hypotheses rather than established causal mechanisms.

### FAIL Findings
- **None.** Zero material defects, zero data leakage, and zero protocol violations.

---

## 12. Final Acceptance Verdict

**FINAL VERDICT: ACCEPTED**

Phase 8 has been audited and approved. The evaluation protocol is academically sound, mathematically verified, zero-leakage compliant, and fully reproducible.

**Stop Boundary Notice:**  
Phase 8 is finalized. In compliance with project authorization rules, **Phase 9 has NOT been started** and requires separate review and explicit authorization.
