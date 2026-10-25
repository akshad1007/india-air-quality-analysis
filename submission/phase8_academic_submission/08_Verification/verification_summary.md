# Academic Submission Package: Verification Summary & Integrity Log

- **Project:** India Air Quality Analysis and Next-Day AQI Forecasting
- **Course:** `24UADPC302: Predictive Modeling Using SPSS (IBM)`
- **Institution:** Sanjivani University, Kopargaon, Maharashtra
- **Package Location:** `C:\Users\KANHA\Desktop\SPSS PROJECT\submission\phase8_academic_submission\`
- **Date Generated:** 2026-10-25
- **Status:** **VERIFIED COMPLETE (37 FILES, 6.87 MB)**

---

## 1. Automated Test Suite Execution Evidence

The complete automated test suite was executed against the project implementation:
```powershell
pytest tests/ -v
```

### Verified Execution Log Summary
- **Exit Code:** `0` (Success)
- **Total Tests Collected:** `51`
- **Total Tests Passed:** `51 (100%)`
- **Failures / Errors / Skips:** `0 / 0 / 0`
- **Execution Runtime:** `15.56 seconds`
- **Test Modules Verified:**
  - `tests/test_evaluation.py` (11 tests passed): Verifies cohort counts, CPCB interval boundaries, out-of-scale mapping, confusion matrix 6x6 geometry, MBE sign convention, and metric recalculations.
  - `tests/test_models.py` (12 tests passed): Verifies persistence identity, chronological fold boundaries, training-only preprocessing, predictor isolation, and finite outputs.
  - `tests/test_features.py` (10 tests passed): Verifies city boundary isolation, calendar target alignment, future perturbation invariance, and lack of lookahead.
  - `tests/test_preprocessing.py` (6 tests passed): Verifies raw immutability, target non-imputation, cross-city leakage prevention, and bounded ffill.
  - `tests/test_data_validation.py` (8 tests passed): Verifies file integrity, SHA-256 checksums, and composite key uniqueness.
  - `tests/test_eda_artifacts.py` (4 tests passed): Verifies figures, tables, and notebook clean execution.

---

## 2. Independent Metric Recalculation Evidence

All headline regression metrics were independently recalculated directly from the raw prediction records (`phase8_test_predictions.csv` and `phase8_stress_predictions.csv`):

### 2019 Primary Held-Out Test Evaluation (Shared Cohort A, $N=6,201$)
- **Naive Persistence Baseline:** MAE = $31.0419\text{ AQI points}$, RMSE = $70.2360\text{ AQI points}$, $R^2 = 0.7445$, MBE = $+0.3614\text{ AQI points}$.
- **Random Forest (Champion):** MAE = **$26.2993\text{ AQI points}$** (**$+15.28\%$ error reduction**), RMSE = **$55.6234\text{ AQI points}$** (**$+20.81\%$ error reduction**), $R^2 = \mathbf{0.8398}$, MBE = **$+4.8666\text{ AQI points}$**.
- **HistGradientBoosting:** MAE = $27.1402\text{ AQI points}$ ($+12.57\%$), RMSE = **$54.9397\text{ AQI points}$** (**$+21.78\%$ error reduction**), $R^2 = \mathbf{0.8437}$, MBE = $+5.6109\text{ AQI points}$.
- **Ridge Regression:** MAE = $29.0261\text{ AQI points}$ ($+6.49\%$), RMSE = $55.8918\text{ AQI points}$ ($+20.42\%$), $R^2 = 0.8382$, MBE = $+2.8414\text{ AQI points}$.

### 2020 COVID-19 Lockdown Stress Period (Shared Cohort A, $N=3,124$)
- **Random Forest (Champion):** MAE = **$19.3288\text{ AQI points}$** (**$+10.64\%$ error reduction** vs Persistence $21.6306$), RMSE = **$39.1893\text{ AQI points}$** (**$+14.69\%$ error reduction** vs Persistence $45.9400$), $R^2 = \mathbf{0.8060}$, MBE = $+4.5243\text{ AQI points}$.

### Ahmedabad Squared-Error Contribution Recalculation
- **Cohort:** 2019 Shared Cohort A ($N=6,201$)
- **Denominator (Total Network Squared Error, RF):** $\sum_{i=1}^{6201} (y_i - \widehat{y}_{i, \text{RF}})^2 = \mathbf{19,185,641.08}$
- **Numerator (Ahmedabad Squared Error, RF, $N=347$ rows):** $\sum_{i \in \text{Ahmedabad}} (y_i - \widehat{y}_{i, \text{RF}})^2 = \mathbf{13,926,980.55}$
- **Exact Contribution Ratio:** $\frac{13,926,980.55}{19,185,641.08} \times 100\% = \mathbf{72.59\%}$ (**VERIFIED**).

---

## 3. Inference Reproducibility Evidence

- **Method:** Serialized model pipelines (`random_forest_val_model.joblib`, `ridge_regression_val_model.joblib`, `histgradientboosting_val_model.joblib`) were reloaded from disk and evaluated on test features (`data/processed/city_day_features.parquet`).
- **Result:** Maximum absolute difference between newly generated predictions and saved prediction CSVs was $\mathbf{0.0000000000}$ across all three model architectures (**100% Bit-for-Bit Deterministic Reproducibility**).

---

## 4. File-Copy Hash Verification (SHA-256 Integrity Log)

Every copied file was compared against its source counterpart in `C:\Users\KANHA\Desktop\SPSS PROJECT\`:

| Destination Path in Package | File Size | SHA-256 Checksum (Prefix) | Match Status |
| :--- | :---: | :---: | :---: |
| `02_Modeling_and_Evaluation/PHASE_7_MODELING_REPORT.md` | $14,072\text{ bytes}$ | `c9ef6239e96ba513...` | **MATCH** |
| `02_Modeling_and_Evaluation/PHASE_8_EVALUATION_REPORT.md` | $34,317\text{ bytes}$ | `759b00044baaf690...` | **MATCH** |
| `03_Acceptance_Audits/PHASE_6_READINESS_REVIEW.md` | $32,282\text{ bytes}$ | `3a28e8da4a2b8d1e...` | **MATCH** |
| `03_Acceptance_Audits/PHASE_7_ACCEPTANCE_AUDIT.md` | $15,967\text{ bytes}$ | `c551c5f9c1df278a...` | **MATCH** |
| `03_Acceptance_Audits/PHASE_8_ACCEPTANCE_AUDIT.md` | $17,570\text{ bytes}$ | `0ce2495ad25c08db...` | **MATCH** |
| `04_Evaluation_Results/model_selection_summary.csv` | $569\text{ bytes}$ | `47f02593d08870df...` | **MATCH** |
| `04_Evaluation_Results/phase8_test_metrics.csv` | $802\text{ bytes}$ | `16235c23925206e3...` | **MATCH** |
| `04_Evaluation_Results/phase8_stress_metrics.csv` | $790\text{ bytes}$ | `7e799719f3630ecb...` | **MATCH** |
| `04_Evaluation_Results/phase8_classification_metrics.csv` | $1,177\text{ bytes}$ | `254e49c76e4ebdd7...` | **MATCH** |
| `04_Evaluation_Results/phase8_ahmedabad_sensitivity.csv` | $964\text{ bytes}$ | `b098e3e14f676df7...` | **MATCH** |
| `04_Evaluation_Results/phase8_residual_by_city.csv` | $826\text{ bytes}$ | `1597f9459223b9dd...` | **MATCH** |
| `04_Evaluation_Results/phase8_residual_by_season.csv` | $240\text{ bytes}$ | `f16e9316d03fd286...` | **MATCH** |
| `04_Evaluation_Results/phase8_residual_by_bucket.csv` | $305\text{ bytes}$ | `3149498399bf31df...` | **MATCH** |
| `05_Prediction_Evidence/phase8_test_predictions.csv` | $719,604\text{ bytes}$ | `6d2e27fb81e033aa...` | **MATCH** |
| `05_Prediction_Evidence/phase8_stress_predictions.csv` | $360,922\text{ bytes}$ | `1622f1090db5fc27...` | **MATCH** |
| `05_Prediction_Evidence/validation_predictions.csv` | $299,341\text{ bytes}$ | `65af924373d4ffdc...` | **MATCH** |
| `06_Source_Code/features/__init__.py` | $715\text{ bytes}$ | `1f70f7d891828aff...` | **MATCH** |
| `06_Source_Code/features/feature_builder.py` | $13,560\text{ bytes}$ | `befed9db275b310f...` | **MATCH** |
| `06_Source_Code/models/__init__.py` | $924\text{ bytes}$ | `9c4500971fe6eb1f...` | **MATCH** |
| `06_Source_Code/models/baseline.py` | $2,020\text{ bytes}$ | `e3e603e05b20f14a...` | **MATCH** |
| `06_Source_Code/models/evaluation.py` | $17,154\text{ bytes}$ | `181ff69300781737...` | **MATCH** |
| `06_Source_Code/models/metrics.py` | $2,826\text{ bytes}$ | `0bd15164f1bbe65e...` | **MATCH** |
| `06_Source_Code/models/pipelines.py` | $4,904\text{ bytes}$ | `eb399bda6d259ef2...` | **MATCH** |
| `06_Source_Code/models/train.py` | $11,695\text{ bytes}$ | `136c8ec12eed1899...` | **MATCH** |
| `06_Source_Code/models/validation.py` | $6,819\text{ bytes}$ | `e5d4cc7a1872adbb...` | **MATCH** |
| `06_Source_Code/tests/test_data_validation.py` | $5,248\text{ bytes}$ | `63ce8e483836e0bb...` | **MATCH** |
| `06_Source_Code/tests/test_eda_artifacts.py` | $4,567\text{ bytes}$ | `18ef5f80f8bdf7fc...` | **MATCH** |
| `06_Source_Code/tests/test_evaluation.py` | $9,326\text{ bytes}$ | `cc91359d36e07e53...` | **MATCH** |
| `06_Source_Code/tests/test_features.py` | $13,350\text{ bytes}$ | `366f3cd9bf07a0b8...` | **MATCH** |
| `06_Source_Code/tests/test_models.py` | $12,617\text{ bytes}$ | `5dd08d7351cfabb9...` | **MATCH** |
| `06_Source_Code/tests/test_preprocessing.py` | $3,837\text{ bytes}$ | `6e73abc275fafd99...` | **MATCH** |
| `07_Model_Artifacts/histgradientboosting_val_model.joblib` | $406,337\text{ bytes}$ | `b088a1370efa1f16...` | **MATCH** |
| `07_Model_Artifacts/random_forest_val_model.joblib` | $5,157,962\text{ bytes}$ | `ab7246042b6cb447...` | **MATCH** |
| `07_Model_Artifacts/ridge_regression_val_model.joblib` | $4,978\text{ bytes}$ | `b2d9629a0d74f059...` | **MATCH** |

*Integrity Conclusion:* 100% of copied files match their source counterpart bit-for-bit with zero file corruptions or discrepancies.

---

## 5. Security & Privacy Audit

- **Credentials & API Keys:** Scanned all files; zero API keys, database credentials, passwords, tokens, or private secrets exist in the package.
- **Environment Files:** No `.env` or configuration secrets were included.
- **Compilation Artifacts:** All `__pycache__` directories and `.pyc` files were excluded.

---

## 6. Distinction: Verified Facts vs. Hypotheses

1. **Directly Verified Facts:**
   - 51/51 tests passing in 15.56 seconds.
   - Exact sample counts: 2019 Cohort A ($6,201$) / Cohort B ($6,286$); 2020 Cohort A ($3,124$) / Cohort B ($3,163$).
   - Re-running inference from saved models gives identical predictions.
   - Ahmedabad accounts for $72.59\%$ of 2019 test squared error.
   - Pre-declared validation MAE criterion selected Random Forest ($27.3559$ vs $28.1849$).
2. **Untested / Descriptive Hypotheses:**
   - Physical cause of Ahmedabad CO sensor saturation ($>400\text{ mg/m}^3$) is documented as an empirical anomaly; specific hardware failure modes were not bench-tested.
   - Socio-economic impact of COVID-19 lockdown coinciding with error compression is an associative correlation; atmospheric causal mechanisms require external numerical counterfactual modeling.

---

## 7. Submission Readiness Sign-Off

The academic submission package is fully verified, self-contained, and structurally sound.  
**Phase 9 has NOT been started.** The project remains strictly frozen at Phase 8 pending faculty review.
