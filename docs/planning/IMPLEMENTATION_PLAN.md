# Phased Implementation Plan (IMPLEMENTATION_PLAN.md)
## India Air Quality Intelligence: Exploratory Analysis, Pollution Pattern Discovery and AQI Forecasting

- **Document Version:** 1.1.0 (Audited & Methodologically Revised)
- **Document History:**
  - `v1.0.0` (2026-10-24): Initial phased roadmap.
  - `v1.1.0` (2026-10-24): Revised following planning audit. Updated chronological partitions to 2019 primary out-of-time test + 2020 shock evaluation; framed missingness imputation as empirical calibration in Phase 3; decoupled regression modeling and secondary classification diagnostics.
- **Project Lead & Technical Director:** Senior Data Scientist & ML Architect
- **Target Timeline & Methodology:** 14 Sequential Phases (Phase 0 to Phase 13), Strict Verification Gates
- **Academic Standard:** AIDS Department PBL 100-Mark Rubric Compliance

---

### Phase Roadmap & Rubric Coverage Overview

```mermaid
graph LR
    P0[P0: Faculty Audit] --> P1[P1: Scope & PRD]
    P1 --> P2[P2: Data Retrieval]
    P2 --> P3[P3: Data Quality & Profiling]
    P3 --> P4[P4: Preprocessing & Calibration]
    P4 --> P5[P5: Statistical EDA]
    P5 --> P6[P6: Leak-Free Features]
    P6 --> P7[P7: Baselines]
    P7 --> P8[P8: Candidate ML Models]
    P8 --> P9[P9: 2019 Test & 2020 Shock Eval]
    P9 --> P10[P10: Policy Insights]
    P10 --> P11[P11: Streamlit App]
    P11 --> P12[P12: Academic Reports & PPT]
    P12 --> P13[P13: Verification & Defense]
```

---

### Detailed Phase Specifications

#### Phase 0: Faculty Document Inspection & Requirements Traceability
- **Status:** **COMPLETED**
- **Deliverable:** [docs/requirements_traceability.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/requirements_traceability.md).

#### Phase 1: Problem Definition, Objectives, Success Criteria & Scope
- **Status:** **COMPLETED & AUDITED (v1.1.0)**
- **Deliverables:** [docs/planning/PRD.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/planning/PRD.md), [docs/review/PLANNING_AUDIT.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/review/PLANNING_AUDIT.md).

#### Phase 2: Dataset Source Inspection, Retrieval & Inventory
- **Objective:** Acquire the 5 CPCB Kaggle dataset CSV files, verify file integrity, and establish raw storage tier.
- **Prerequisites:** User provides or confirms raw CSVs in `data/raw/`.
- **Tasks:**
  1. Verify presence of `city_day.csv`, `city_hour.csv`, `station_day.csv`, `station_hour.csv`, `stations.csv`.
  2. Compute file sizes and SHA-256 checksums (`src/data/inventory.py`).
  3. Validate schema headers and log confirmed row counts.
- **Rubric Addressed:** **Category 2: Dataset & Data Understanding (10 Marks).**

#### Phase 3: Data Quality Assessment & Empirical Missingness Audit
- **Objective:** Profile dataset columns, missingness mechanisms, sensor dropout distributions, and verify absence of `StationType`.
- **Prerequisites:** Phase 2 complete.
- **Tasks:**
  1. Implement automated data profiling (`src/validation/data_validator.py` and `notebooks/01_data_audit.ipynb`).
  2. Generate missingness percentage tables and city-wise temporal coverage heatmaps.
  3. Analyze length of continuous missing gaps per city to empirically evaluate bounded fill options (1-day vs. 2-day vs. native tree handling).
  4. Finalize data dictionary and author `docs/reports/data_quality_report.md`.
- **Rubric Addressed:** **Category 2: Dataset & Data Understanding (10 Marks).**

#### Phase 4: Reproducible Cleaning & Preprocessing Pipeline
- **Objective:** Implement modular data cleaning, continuous calendar reindexing, calibrated imputation, and spatial filtering.
- **Prerequisites:** Phase 3 audit findings.
- **Tasks:**
  1. Standardize column names and cast dates to `datetime64[ns]`.
  2. Implement per-city continuous calendar generator to expose unmonitored days.
  3. Execute calibrated forward-fill based on Phase 3 gap analysis.
  4. Filter out cities with insufficient active operational coverage.
  5. Preserve authentic physical pollution spikes in training and test sets.
- **Rubric Addressed:** **Category 3: Data Preprocessing & Cleaning (10 Marks).**

#### Phase 5: Exploratory Data Analysis & Statistical Investigation
- **Objective:** Answer research questions through rigorous statistical distributions, seasonal decompositions, and correlation analysis.
- **Prerequisites:** Phase 4 clean dataset.
- **Tasks:**
  1. Compute univariate distributions (mean, median, IQR, 95th percentile) for all chemical pollutants.
  2. Plot multi-year city AQI trends with CPCB hazard bands.
  3. Perform seasonal time-series decomposition (Winter, Pre-Monsoon, Monsoon, Post-Monsoon).
  4. Calculate chemical precursor correlation matrices ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$).
  5. Quantify COVID-19 lockdown shock (March–May 2020 vs. historical baseline).
- **Rubric Addressed:** **Category 4: Exploratory Data Analysis (EDA) (10 Marks).**

#### Phase 6: Prediction Target Definition & Leakage-Free Feature Engineering
- **Objective:** Construct leak-free feature matrix for next-day AQI prediction ($T+1$) using strictly historical observations ($\le T$).
- **Prerequisites:** Cleaned time-series data from Phase 4.
- **Tasks:**
  1. Extract calendar features (`day_of_week`, `month`, `day_of_year`, `is_weekend`).
  2. Generate autoregressive lags ($AQI_{t-1}, AQI_{t-2}, AQI_{t-3}, AQI_{t-7}$) grouped by city.
  3. Generate backward-looking rolling statistics (7-day and 14-day rolling mean, std, max) shifted by 1 day.
  4. Engineer pollutant lag proxies ($PM2.5_{t-1}, NO2_{t-1}, SO2_{t-1}, CO_{t-1}$).
  5. Construct primary target ($y_{reg} = AQI_{t+1}$).
  6. Partition into chronological splits:
     - Train: 2015-01-01 to 2018-06-30
     - Validation: 2018-07-01 to 2018-12-31
     - Primary Held-Out Test: 2019-01-01 to 2019-12-31 (Full 365-day annual cycle)
     - Stress-Test Set: 2020-01-01 to 2020-07-01 (COVID-19 lockdown shock)
- **Tests & Verification:** `pytest tests/test_features.py` (Asserts zero temporal leakage).
- **Rubric Addressed:** **Category 5: Feature Engineering / Selection (10 Marks).**

#### Phase 7: Baseline Forecasting & Classification Formulation
- **Objective:** Establish the naive persistence benchmark and formalize secondary classification diagnostics.
- **Prerequisites:** Chronological feature matrices.
- **Tasks:**
  1. Implement Naive Persistence Baseline ($\hat{y}_{t+1} = AQI_t$).
  2. Compute baseline metrics on 2018 validation fold, 2019 primary test fold, and 2020 shock fold.
- **Rubric Addressed:** **Category 6: Model Selection & Implementation (Partial).**

#### Phase 8: Candidate Machine Learning Model Training
- **Objective:** Implement, tune, and train supervised regression models using scikit-learn pipelines.
- **Prerequisites:** Phase 6 and Phase 7 complete.
- **Tasks:**
  1. Implement Ridge Regression with cross-validated regularization (`RidgeCV`).
  2. Implement Random Forest Regressor (`RandomForestRegressor`).
  3. Implement Histogram-based Gradient Boosting (`HistGradientBoostingRegressor`).
  4. Perform hyperparameter tuning on 2018 validation set using TimeSeriesSplit cross-validation.
  5. Serialize trained model artifacts to `models/*.pkl`.
- **Rubric Addressed:** **Category 6: Model Selection & Implementation (15 Marks).**

#### Phase 9: Model Evaluation, Validation & Residual Diagnostics
- **Objective:** Evaluate candidate models on untouched 2019 full-year test set and 2020 shock set; conduct deep residual diagnostics.
- **Prerequisites:** Trained model artifacts from Phase 8.
- **Tasks:**
  1. Compute primary out-of-time test metrics on 2019: MAE, RMSE, $R^2$, MAPE.
  2. Compare ML candidates against Naive Persistence Baseline.
  3. Secondary Classification Diagnostics: Threshold continuous predictions into CPCB Buckets; compute Confusion Matrix, Macro F1, Balanced Accuracy.
  4. Stress-Testing on 2020: Quantify error degradation during the March–May 2020 COVID-19 lockdown.
  5. Residual diagnostics: Plot $e_t = y_t - \hat{y}_t$ over time, inspect error across AQI severity tiers and cities.
- **Rubric Addressed:** **Category 7: Model Evaluation & Validation (15 Marks).**

#### Phase 10: Environmental Insights & Evidence-Backed Recommendations
- **Objective:** Translate analytical and predictive findings into grounded, actionable recommendations for stakeholders.
- **Rubric Addressed:** **Category 8: Business Insights & Recommendations (10 Marks).**

#### Phase 11: Interactive Streamlit Analytical Dashboard
- **Objective:** Construct a clean, multi-page Streamlit dashboard for interactive exploration and offline scenario forecasting.
- **Rubric Addressed:** Presentation & Demonstration Value (Category 9 Enhancement).

#### Phase 12: Academic Documentation, Presentation & Viva Preparation
- **Objective:** Prepare the complete academic submission package: Technical Report, Executive Summary, Presentation Deck, and Viva Q&A Guide.
- **Rubric Addressed:** **Category 9: Documentation & Presentation (10 Marks).**

#### Phase 13: Full Verification Checklist & Repository Finalization
- **Objective:** Execute end-to-end reproducibility audit, finalize `README.md`, verify traceability matrix, and clean repository.
- **Rubric Addressed:** Complete 100-Mark Rubric Verification.
