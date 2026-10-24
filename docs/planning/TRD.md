# Technical Requirements Document (TRD)
## India Air Quality Analysis and Next-Day AQI Forecasting

- **System Architecture Version:** 1.2.0 (Academic Alignment & Phase 6 Readiness Baseline)
- **Academic Subject:** Predictive Modeling Using SPSS (IBM) (`24UADPC302`)
- **Academic Term:** Semester 5 (Third Year B.Tech)
- **Department & Institution:** TY AIDS Department, Sanjivani University, Kopargaon, Maharashtra
- **Faculty Guide:** Mr. Rohit Pawar
- **Project Team:**
  1. Akshad Makhana (`2124UDSM1007`)
  2. Shubham Pawar (`2124UDSM1074`)
  3. Yash Kolhe (`2124UDSM1107`)
  4. Pranav Hadole (`2124UDSM1108`)
- **Target OS / Environment:** Windows 10/11 x64, Python 3.10.11
- **Project Root Directory:** `c:\Users\KANHA\Desktop\SPSS PROJECT`
- **Execution Paradigm:** Modular Clean Architecture, Pipeline Orchestration, Reproducible Testing

---

### 1. System Overview & Architectural Principles

The India Air Quality Intelligence platform is an end-to-end analytical and machine learning system engineered for empirical environmental research and out-of-time forecasting. The architecture enforces three non-negotiable principles:
1. **Strict Decoupling of Concerns:** Raw data ingestion, validation, transformation, feature generation, model training, evaluation, and presentation layers reside in separate modules with single responsibilities.
2. **Deterministic Reproducibility:** Data processing and machine learning workflows are completely deterministic under explicit random seeds, time-aware splits, and automated pipeline scripts.
3. **Leakage-Immune Temporal Design:** All transformations, scalers, imputations, and feature calculations operate strictly on historical windows relative to the prediction origin $T$, preventing any contamination from future observations.

---

### 2. High-Level Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                              RAW ENVIRONMENTAL DATA                               |
| (city_day.csv, city_hour.csv, station_day.csv, station_hour.csv, stations.csv)    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        INGESTION & DATA VALIDATION LAYER                          |
|  - Checksum & schema audit                                                        |
|  - Data type enforcement (datetime64[ns], float32)                                |
|  - Range boundary validation (0 <= AQI <= 1000, 0 <= Pollutant)                   |
|  - Missingness profiling & temporal continuity checks                             |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      DATA PREPROCESSING & CLEANING PIPELINE                       |
|  - City-level continuous calendar reindexing                                      |
|  - Empirical missingness handling: Bounded forward-fill vs. native tree support   |
|  - Active-period coverage filtering (calibrated in Phase 3)                       |
|  - Outlier flagging (IQR / physical limits; preserve severe physical spikes)      |
|  - Storage to data/interim/ and data/processed/                                   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      FEATURE ENGINEERING & SELECTION LAYER                        |
|  - Calendar features (DayOfWeek, Month, DayOfYear, IsWeekend)                     |
|  - Lag features (AQI_{t-1}, AQI_{t-2}, AQI_{t-7}, PM2.5_{t-1})                   |
|  - Rolling statistics (Rolling 7d Mean, Std, Max with closed='left')              |
|  - Target creation: Next_Day_AQI = AQI_{t+1}                                      |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      CHRONOLOGICAL SPLIT & MODEL BENCHMARK                        |
|  - Train: 2015-01-01 to 2018-06-30 | Val: 2018-07-01 to 2018-12-31                 |
|  - Primary Out-of-Time Held-out Test: 2019-01-01 to 2019-12-31 (Full 365 Days)    |
|  - Exogenous Shock Stress-Test: 2020-01-01 to 2020-07-01 (COVID-19 Lockdown)      |
|  - Models: Naive Persistence, RidgeCV, RandomForest, HistGradientBoosting        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        EVALUATION & RESIDUAL DIAGNOSTICS                          |
|  - Primary Metrics: MAE, RMSE, R2, MAPE (in native AQI units)                     |
|  - Secondary Classification Diagnostics: Continuous predictions mapped to         |
|    CPCB Buckets -> Confusion Matrix, Balanced Accuracy, Macro F1                  |
|  - Stress-Testing: Quantify performance collapse under 2020 lockdown emissions    |
|  - Residual diagnostics across cities and pollution severity tiers               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                          PRESENTATION & DASHBOARD LAYER                           |
|  - Streamlit multi-page analytical dashboard (`app/`)                             |
|  - Static academic report figures (`reports/figures/`)                            |
|  - Comprehensive tables and metrics JSON (`reports/metrics/`)                     |
+-----------------------------------------------------------------------------------+
```

---

### 3. Chronological Validation & Splitting Rationale

Following the senior data scientist audit (`ISS-01`), the splitting protocol is formally structured to prevent temporal leakage and evaluate models on an authentic, complete annual atmospheric cycle:

| Data Split Partition | Calendar Range | Purpose & Scientific Rationale |
| :--- | :--- | :--- |
| **Historical Training Set** | `2015-01-01` to `2018-06-30` (~3.5 years) | Model parameter estimation on multi-year baseline data. |
| **Validation Set** | `2018-07-01` to `2018-12-31` (6 months) | Hyperparameter tuning and model selection using TimeSeriesSplit. |
| **Primary Held-Out Test Set** | **`2019-01-01` to `2019-12-31` (365 days)** | **Primary Academic Evaluation:** Untouched out-of-time test set containing a complete, uncorrupted annual meteorological cycle (Winter, Pre-Monsoon, Monsoon, Stubble Burning, Early Winter). |
| **Exogenous Shock Stress-Test Set** | **`2020-01-01` to `2020-07-01` (~6 months)** | **Dedicated Shock Scenario:** Evaluates model resilience during the unprecedented March–May 2020 nationwide COVID-19 lockdown emission collapse. |

---

### 4. Missing Data Strategy: Imputation vs. Native Tree Handling

The audit (`ISS-03`) highlighted the risk of dogmatically enforcing a 2-day forward-fill. In the revised architecture:
1. **Linear Interpolation Forbidden for Forecasting:** Linear interpolation uses future values ($T+1$) and is strictly banned for forecasting features.
2. **Candidate Imputation Approaches Evaluated in Phase 4:**
   - *Baseline Imputation:* Per-city bounded forward-fill ($\le 2$ days).
   - *Native Handling:* For tree-based models (`HistGradientBoostingRegressor`, `LightGBM`), missing values will also be evaluated natively without synthetic imputation to measure empirical impact.
3. **Active Period Completeness:** Station coverage will be calculated relative to each station's verified commissioning date rather than penalizing stations inactive in 2015.

---

### 5. Verified Tooling & Environment

- **Python Version:** 3.10.11 (Verified)
- **Active Core Packages:**
  - `pandas`: 2.3.2 | `numpy`: 2.2.6
  - `scikit-learn`: 1.7.2 | `scipy`: 1.15.3
  - `matplotlib`: 3.10.3 | `seaborn`: 0.13.2 | `plotly`: 6.3.0
  - `statsmodels`: 0.14.5 | `lightgbm`: 4.6.0 | `xgboost`: 3.0.4
  - `streamlit`: 1.28.1 | `pytest`: 7.4.3

---

### 6. Component Specifications & Testing Strategy

- `src/validation/data_validator.py`: Asserts schema conformity, missingness rates, and zero negative concentrations.
- `src/preprocessing/cleaner.py`: Standardizes calendars, manages bounded forward-fills, and logs row-level changes.
- `src/features/feature_builder.py`: Generates strictly backward-looking features (`shift(1)` applied to rolling windows and lags).
- `src/models/trainer.py`: Encapsulates Naive Persistence, RidgeCV, RandomForest, and HistGradientBoosting.
- `src/evaluation/evaluator.py`: Computes MAE, RMSE, $R^2$, MAPE, and maps continuous outputs to CPCB confusion matrices.
- `tests/test_features.py`: Automated unit test verifying zero temporal leakage between feature timestamps and prediction targets.
