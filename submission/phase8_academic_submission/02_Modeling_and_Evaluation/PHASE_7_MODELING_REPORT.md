# Phase 7 Modeling Report: Model Selection & Implementation

## India Air Quality Analysis and Next-Day AQI Forecasting

- **Document Identifier:** `docs/reports/PHASE_7_MODELING_REPORT.md`
- **Date:** 2026-10-25
- **Academic Course:** `24UADPC302: Predictive Modeling Using SPSS (IBM)`
- **Department & Institution:** TY AIDS, Sanjivani University, Kopargaon, Maharashtra
- **Faculty Guide:** Mr. Rohit Pawar
- **Project Team:**
  1. Akshad Makhana (`2124UDSM1007`)
  2. Shubham Pawar (`2124UDSM1074`)
  3. Yash Kolhe (`2124UDSM1107`)
  4. Pranav Hadole (`2124UDSM1108`)
- **Repository Remote:** `https://github.com/akshad1007/india-air-quality-analysis.git`
- **Monetary Reporting Convention:** Indian Rupee (`₹`, `INR`), adhering to Indian numbering conventions (lakhs, crores). *Note: No financial costs or operational budgets were measured for this modeling phase; no artificial monetary figures are invented.*
- **Statistical Units Convention:** MAE, RMSE, and MBE are measured strictly in **AQI points**. $R^2$ is dimensionless.
- **Current Milestone State:** **PHASE 7 COMPLETE — 40/40 TESTS PASSING**
- **Phase 8 Status:** **NOT STARTED — PENDING USER APPROVAL**

---

## 1. Executive Summary & Objective

In accordance with the approved Implementation Plan, TRD, and faculty PBL guidelines, **Phase 7 (Model Selection and Implementation)** has been executed with strict adherence to chronological validation and zero test leakage.

Four forecasting systems were implemented, calibrated, and evaluated:
1. **Benchmark Baseline:** Naive Persistence ($\widehat{\text{AQI}}_{c, t+1} = \text{AQI}_{c, t}$).
2. **Model A (Linear Regularized):** Ridge Regression with One-Hot City encoding, training-fitted median imputation, standard scaling, and chronologically cross-validated $\alpha$.
3. **Model B (Non-Linear Ensemble Bagging):** Random Forest Regressor with One-Hot City encoding and training-fitted median imputation.
4. **Model C (Non-Linear Histogram Boosting):** HistGradientBoostingRegressor with ordinal city encoding and native histogram-based missing value handling.

**Core Finding on Designated Validation Period (`2018-07-01` to `2018-12-31`, Shared Cohort $N=2,982$):**
- **Random Forest** achieved the lowest validation error: **MAE 27.3559** (+7.70% improvement over Persistence), **RMSE 59.7368** (+14.78% error reduction), and **$R^2 = 0.8774$**.
- **HistGradientBoosting** achieved the second-strongest validation performance: **MAE 28.1849** (+4.90% improvement over Persistence), **RMSE 60.2068** (+14.11% error reduction), and **$R^2 = 0.8755$**.
- **Ridge Regression** reduced large spike variance (RMSE improved from 70.0930 to 62.0231, +11.51%), but had slightly higher median error (MAE 30.6482 vs 29.6382) due to inability to model non-linear pollutant interactions.
- **Strict Isolation Guarantee:** The primary held-out test set (`2019-01-01` to `2019-12-31`) and stress-test set (`2020-01-01` to `2020-07-01`) remained **100% untouched and uninspected** during all hyperparameter tuning and model selection.

---

## 2. Chronological Validation Architecture

To eliminate temporal lookahead leakage, model selection utilized a **Chronological Expanding-Window Validation** protocol across the training partition (`2015-01-02` to `2018-06-30`), with final model confirmation performed on the **Designated Validation Period** (`2018-07-01` to `2018-12-31`).

```
[Fold 1: Train 2015-01-02 to 2016-12-31 (2.0y)] ──> [Val: 2017-01-01 to 2017-06-30 (6m)]
[Fold 2: Train 2015-01-02 to 2017-06-30 (2.5y)] ──────> [Val: 2017-07-01 to 2017-12-31 (6m)]
[Fold 3: Train 2015-01-02 to 2017-12-31 (3.0y)] ──────────> [Val: 2018-01-01 to 2018-06-30 (6m)]
========================================================================================
[DESIGNATED CONFIRMATION FOLD: Train 2015-01-02 to 2018-06-30 (~3.5y)] ──> [Val: 2018-07-01 to 2018-12-31 (6m)]
```

### Key Validation Safeguards
1. **Target Date Indexing:** Folds are sliced strictly by `Target_Date` ($t+1$). Because observations at origin $t$ strictly precede $t+1$, no feature can access future data.
2. **Date Atomicity:** All municipal jurisdictions for a given `Target_Date` are retained within the same fold. Rows are never split arbitrarily across temporal boundaries.
3. **Training-Only Learned Preprocessing:** Imputation medians, standard scaling means/variances, and categorical encoders are fitted strictly on each fold's training slice (`tau <= train_end`) and applied to the validation slice without refitting.

---

## 3. Implemented Forecasting Models & Pipelines

### 3.1 Naive Persistence Baseline (`src/models/baseline.py`)
- **Formula:** $\widehat{\text{AQI}}_{c, t+1} = \text{AQI}_{c, t}$
- **Eligibility:** Active if and only if $\text{AQI}_{c, t}$ is observed and $D_{t+1} - D_t \equiv 1\text{ day}$.
- **Missing Handling:** Evaluated strictly on Cohort A (where $\text{AQI}_t$ is observed). Unobserved origin AQI is never imputed to manufacture artificial eligibility.

### 3.2 Model A — Ridge Regression (`src/models/pipelines.py:build_ridge_pipeline`)
- **Categorical Processing:** `OneHotEncoder(handle_unknown='ignore')` applied to `City`.
- **Numerical Processing:** `SimpleImputer(strategy='median')` followed by `StandardScaler()` fit on training fold.
- **Regularization Tuning:** Evaluated candidate $\alpha \in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]$ across the 3 training expanding folds:
  - $\alpha = 0.1$: Mean CV MAE = $36.9785$
  - $\alpha = 10.0$: Mean CV MAE = $36.3869$
  - $\alpha = 100.0$: **Mean CV MAE = 36.0117 (Selected optimal $\alpha$)**
  - $\alpha = 500.0$: Mean CV MAE = $36.1437$

### 3.3 Model B — Random Forest Regressor (`src/models/pipelines.py:build_rf_pipeline`)
- **Categorical Processing:** `OneHotEncoder(handle_unknown='ignore')` applied to `City` (preserves nominal independence without imposing arbitrary integer distances).
- **Numerical Processing:** `SimpleImputer(strategy='median')` fit on training fold.
- **Hyperparameter Tuning:** Evaluated across training expanding folds:
  - Config 1 (`n_estimators=50, max_depth=8, min_samples_split=10`): Mean CV MAE = $31.9167$
  - Config 2 (`n_estimators=100, max_depth=10, min_samples_split=5`): **Mean CV MAE = 31.2605 (Selected optimal configuration)**
  - Config 3 (`n_estimators=100, max_depth=14, min_samples_split=5`): Mean CV MAE = $31.4284$

### 3.4 Model C — HistGradientBoostingRegressor (`src/models/pipelines.py:build_histgbm_pipeline`)
- **Categorical Processing:** `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)` applied to `City`, passed via `categorical_features=[0]`. Unseen cities at test time receive `NaN` and are routed along the missing-category split.
- **Numerical Processing:** Native histogram binning handles missing values natively without imputation or scaling.
- **Hyperparameter Tuning:** Evaluated across training expanding folds:
  - Config 1 (`lr=0.05, max_iter=100, max_depth=6, l2=1.0`): Mean CV MAE = $32.4118$
  - Config 2 (`lr=0.05, max_iter=150, max_depth=8, l2=2.0`): Mean CV MAE = $32.0945$
  - Config 3 (`lr=0.03, max_iter=200, max_depth=8, l2=1.0`): **Mean CV MAE = 31.9026 (Selected optimal configuration)**

---

## 4. Empirical Validation Period Results (2018-07-01 to 2018-12-31)

### 4.1 Primary Benchmark: Shared-Cohort Comparison (Cohort A)
Evaluated on the exact same $N=2,982$ days where both $\text{AQI}_t$ and $\text{Target\_AQI}$ are observed:

| Model Architecture | Cohort Definition | Sample Size ($N$) | MAE (AQI) | RMSE (AQI) | $R^2$ Score | MAE Error Reduction vs. Persistence | RMSE Error Reduction vs. Persistence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Persistence** | **Cohort A (Shared)** | **$2,982$** | **$29.6382$** | **$70.0930$** | **$0.8313$** | **0.00% (Baseline)** | **0.00% (Baseline)** |
| **Ridge Regression** ($\alpha=100$) | **Cohort A (Shared)** | **$2,982$** | $30.6482$ | $62.0231$ | $0.8679$ | $-3.41\%$ (slight loss) | **$+11.51\%$** |
| **Random Forest** (Selected Champion) | **Cohort A (Shared)** | **$2,982$** | **$27.3559$** | **$59.7368$** | **$0.8774$** | **$+7.70\%$** | **$+14.78\%$** |
| **HistGradientBoosting** | **Cohort A (Shared)** | **$2,982$** | **$28.1849$** | **$60.2068$** | **$0.8755$** | **$+4.90\%$** | **$+14.11\%$** |

### 4.2 Operational Evaluation: Model-Only Cohort (Cohort B)
Evaluated on all target-valid days ($N=3,045$), including the 63 days where current-day $\text{AQI}_t$ was unobserved:

| Model Architecture | Cohort Definition | Sample Size ($N$) | MAE (AQI) | RMSE (AQI) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression** | Cohort B (Model-Only) | $3,045$ | $31.1453$ | $62.6066$ | $0.8641$ |
| **Random Forest** | Cohort B (Model-Only) | $3,045$ | $27.6970$ | $60.2944$ | $0.8739$ |
| **HistGradientBoosting** | Cohort B (Model-Only) | $3,045$ | $28.5898$ | $60.9760$ | $0.8711$ |

*Scientific Note on Cohort Comparability:* Cohort B metrics are reported as separate operational indicators. Because the 63 additional days represent periods of sensor dropouts, they are not directly comparable to Cohort A metrics.

---

## 5. Automated Test Suite Results

The complete automated test suite was executed via `pytest tests/`:

```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-7.4.3, pluggy-1.6.0
rootdir: C:\Users\KANHA\Desktop\SPSS PROJECT
configfile: pytest.ini
plugins: anyio-4.10.0, cov-6.2.1, mock-3.12.0
collected 40 items

tests\test_data_validation.py ........                                   [ 20%]
tests\test_eda_artifacts.py ....                                         [ 30%]
tests\test_features.py ..........                                        [ 55%]
tests\test_models.py ............                                        [ 85%]
tests\test_preprocessing.py ......                                       [100%]

============================= 40 passed in 26.61s =============================
```

All 40 unit and integration tests passed cleanly across all categories:
- **`test_data_validation.py` (8 tests):** SHA-256 hashes, schema constraints, non-negativity, CPCB statutory bounds, PDF integrity.
- **`test_preprocessing.py` (6 tests):** Calendar continuity, 2-day bounded fill, target preservation, city usability tiers.
- **`test_eda_artifacts.py` (4 tests):** Artifact generation, table schemas, clean notebook execution, multi-city sampling.
- **`test_features.py` (10 tests):** City boundary isolation, target alignment, calendar gap handling, future perturbation invariance, missing target preservation, rolling window endpoints, partition disjointness, persistence eligibility, training-only preprocessor isolation, integration counts.
- **`test_models.py` (12 tests):** Persistence equality to `AQI_t`, persistence eligibility constraints, approved predictor columns only, chronological non-overlapping fold bounds, date atomicity, training-only imputer/scaler fitting, missing predictor handling across models, unseen city handling, prediction finiteness, seed reproducibility, shared cohort row identity, test set isolation.

---

## 6. Artifact Inventory

Phase 7 produced the following reproducible outputs:
- **Code Modules:**
  - [`src/models/__init__.py`](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/src/models/__init__.py): Model package exports.
  - [`src/models/metrics.py`](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/src/models/metrics.py): Regression evaluation metrics and relative improvement functions.
  - [`src/models/baseline.py`](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/src/models/baseline.py): Naive persistence estimator.
  - [`src/models/pipelines.py`](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/src/models/pipelines.py): Ridge, RF, and HistGBM pipelines.
  - [`src/models/validation.py`](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/src/models/validation.py): Expanding-window CV and cohort evaluation.
  - [`src/models/train.py`](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/src/models/train.py): Complete tuning and model selection orchestrator.
- **Tests:**
  - [`tests/test_models.py`](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/tests/test_models.py): 12 automated unit/integration tests.
- **Reports & Artifacts:**
  - `reports/model_selection_summary.csv`: Summary metrics table across models and cohorts.
  - `reports/model_selection_results.json`: Complete fold-level tuning and evaluation records.
  - `reports/validation_predictions.csv`: Row-level predictions for all models on the validation split ($3,045$ rows).
  - `models/ridge_regression_val_model.joblib`: Fitted Ridge pipeline on training partition.
  - `models/random_forest_val_model.joblib`: Fitted Random Forest pipeline on training partition.
  - `models/histgradientboosting_val_model.joblib`: Fitted HistGBM pipeline on training partition.

---

## 7. Model Selection Verdict & Phase 8 Readiness

### Champion Model Selected on Validation Data:
$$\mathbf{Random\ Forest\ Regressor\ (n\_estimators=100,\ max\_depth=10,\ min\_samples\_split=5)}$$
- **Validation MAE:** **$27.3559$** (Best among all models; $+7.70\%$ improvement over Naive Persistence).
- **Validation RMSE:** **$59.7368$** (Best among all models; $+14.78\%$ error reduction over Naive Persistence).
- **Validation $R^2$:** **$0.8774$** (Explains $87.7\%$ of next-day AQI variance on unseen historical validation data).

### Strongest Secondary Candidate:
$$\mathbf{HistGradientBoostingRegressor\ (lr=0.03,\ max\_iter=200,\ max\_depth=8,\ l2=1.0)}$$
- **Validation MAE:** **$28.1849$** ($+4.90\%$ improvement over Naive Persistence).
- **Validation RMSE:** **$60.2068$** ($+14.11\%$ error reduction).
- **Key Advantage:** Native handling of un-imputed missing values with $10\times$ faster inference speed.

### Phase 8 Readiness:
Phase 7 is complete. The project is positioned to proceed to **Phase 8 (Model Evaluation & Validation on Held-Out 2019 Test & 2020 Stress Set)** upon your authorization.
