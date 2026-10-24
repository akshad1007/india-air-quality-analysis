# Product Requirements Document (PRD)

## India Air Quality Analysis and Next-Day AQI Forecasting

- **Project Version:** 1.2.0 (Academic Alignment & Phase 6 Readiness Baseline)
- **Academic Subject:** Predictive Modeling Using SPSS (IBM) (`24UADPC302`)
- **Academic Term:** Semester 5 (Third Year B.Tech)
- **Department & Institution:** TY AIDS Department, Sanjivani University, Kopargaon, Maharashtra
- **Faculty Guide:** Mr. Rohit Pawar
- **Project Team:**
  1. Akshad Makhana (`2124UDSM1007`)
  2. Shubham Pawar (`2124UDSM1074`)
  3. Yash Kolhe (`2124UDSM1107`)
  4. Pranav Hadole (`2124UDSM1108`)
- **Domain:** Environmental Data Science, Public Health Informatics, Time-Series Forecasting
- **Academic Context:** Problem-Based Learning (PBL) Submission, AIDS Department (100 Marks Rubric)
- **Status:** Approved Post-Audit Baseline

---

### 1. Executive Summary

Air pollution is among the most severe environmental and public health crises in India. Urban populations across Indo-Gangetic plains, coastal zones, and Deccan plateaus face chronic exposure to particulate matter ($PM_{2.5}$, $PM_{10}$) and toxic gases ($NO_2$, $SO_2$, $CO$, $O_3$). However, pollution dynamics vary dramatically across geographies, seasons, and meteorological regimes.

The **India Air Quality Intelligence** project is an end-to-end data science, empirical analysis, and machine learning initiative designed to extract actionable environmental intelligence from the multi-year national monitoring dataset (2015–2020) published by CPCB (Central Pollution Control Board). The project rigorously evaluates historical pollution distributions, quantifies seasonal and diurnal patterns across major Indian metropolitan clusters, establishes a leak-free temporal feature engineering pipeline, and benchmarks predictive machine learning models against strong naive baselines for **next-day city-level AQI forecasting**.

The deliverable includes reproducible analytical notebooks, modular Python engineering code, automated tests, a comprehensive technical report, an executive summary, and an analytical Streamlit dashboard for research and policy exploration.

---

### 2. Problem Statement & Real-World Context

#### 2.1 The Environmental Reality

According to global health studies (e.g., State of Global Air, WHO), air pollution contributes to over 1.6 million premature deaths annually in India. While the Central Pollution Control Board (CPCB) operates the National Air Quality Monitoring Programme (NAQMP) and Continuous Ambient Air Quality Monitoring Stations (CAAQMS), raw environmental data exhibits high levels of sensor missingness, intermittent calibration drift, uneven spatial coverage, and extreme non-linear episodic spikes (such as post-monsoon crop residue burning and winter temperature inversions in North India).

#### 2.2 Analytical Deficit in Existing Work

Many academic projects treat air quality as a generic tabular regression or classification problem. Common methodological failures include:

1. **Random Train-Test Splitting:** Leaking future temporal structure into historical training folds.
2. **Feature Leakage:** Incorporating same-day contemporaneous pollutant measurements or future rolling windows to predict the target AQI.
3. **Ignoring Sensor Missingness:** Arbitrarily imputing missing values with zeros or dataset-wide means, corrupting actual physical baselines.
4. **Disregarding Naive Baselines:** Failing to verify whether complex machine learning architectures (e.g., Random Forests, Gradient Boosting) actually outperform a simple persistence model (Tomorrow's AQI = Today's AQI).
5. **Evaluating on Unrepresentative Periods:** Evaluating models exclusively on anomalous periods (e.g., the 2020 COVID-19 lockdown) without assessing generalizability across complete annual seasonal cycles.

#### 2.3 Refined Empirical Research Questions

The project is framed around six empirically verifiable research questions strictly aligned with verified dataset fields:

- **RQ1 (Historical Trends & Macro Shifts):** What macro-level temporal trajectories and structural changes occurred in ambient AQI across monitored Indian cities between 2015 and 2019/2020?
- **RQ2 (Spatial & Station Heterogeneity):** How do pollutant concentrations and AQI profiles differ across geographical regions (derived from State/City metadata) and across intra-city monitoring stations within major urban centers? *(Revised post-audit: removed unsupported "industrial vs. residential" station-type claims).*
- **RQ3 (Seasonal & Meteorological Signatures):** What are the dominant seasonal (Winter, Pre-Monsoon, Monsoon, Post-Monsoon) drivers of critical pollutants ($PM_{2.5}$ vs. $PM_{10}$ vs. gaseous precursors)?
- **RQ4 (Predictive Feature Utility & Leakage Prevention):** Which historical temporal features, autoregressive lags ($T-1, T-2, T-7$), and backward-looking rolling statistical moments hold statistically significant signal for predicting next-day AQI without future data leakage?
- **RQ5 (Model Benchmark & Forecast Feasibility):** Can supervised machine learning models (Ridge Regression, Random Forest, Histogram-based Gradient Boosting) outperform a naive persistence baseline under strict chronological out-of-time evaluation across a complete annual cycle (2019)?
- **RQ6 (Subgroup Diagnostics & Exogenous Shock Stress-Testing):** Under what specific environmental regimes (e.g., severe winter smog vs. clean monsoon days) do forecasting models degrade, and how resilient are historical models when subjected to an unprecedented exogenous structural shock (the March–May 2020 COVID-19 lockdown)?

---

### 3. Project Objectives & Boundaries

#### 3.1 Primary Objective

Develop, validate, and document a rigorous, reproducible, and leak-free machine learning pipeline to forecast next-day city-level Air Quality Index (AQI) across representative Indian cities using historical monitoring observations, benchmarking candidates against persistence baselines.

#### 3.2 Secondary Objectives

1. **Comprehensive Data Audit:** Profile all 5 files of the CPCB India Air Quality dataset (2015–2020), quantifying missingness mechanisms, duplicate keys, temporal continuity, and sensor coverage.
2. **Empirical Exploratory Analysis:** Uncover seasonal cycles, city-to-city correlations, pollutant-to-AQI dominance, and the structural impact of macro events (e.g., 2020 lockdown).
3. **Leakage-Safe Feature Engineering:** Construct date-time, autoregressive lag, and backward-looking rolling statistics strictly constrained to information available at forecast origin $T$.
4. **Hierarchical Modeling Paradigm:**
   - *Primary (Regression):* Direct next-day numerical AQI forecasting evaluated on MAE, RMSE, and $R^2$.
   - *Secondary Evaluation (Classification / Diagnostic):* Map continuous AQI predictions into statutory CPCB AQI Buckets (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) to generate Confusion Matrices, Balanced Accuracy, and Macro F1-scores, fulfilling Rubric Criterion 7 without introducing model divergence. Optional independent classification model explored as a methodological ablation.
5. **Defensible Chronological Validation:**
   - *Primary Out-of-Time Test Set:* Full calendar year **2019 (Jan 1 – Dec 31, 2019)**, ensuring a complete 365-day annual cycle across all seasons.
   - *Exogenous Shock Evaluation Set:* **2020 (Jan 1 – Jul 1, 2020)**, evaluating model resilience during the historic COVID-19 lockdown emission collapse.
6. **Actionable Recommendations:** Translate empirical findings into grounded policy, public advisory, and monitoring network recommendations.
7. **Academic Excellence & Software Engineering:** Maintain clean modular Python packages (`src/`), full unit test coverage (`pytest`), automated documentation, and presentation materials for viva defense.

#### 3.3 Scope & Boundaries (What This Project Is and Is NOT)

- **IN SCOPE:**
  - Historical offline analysis of CPCB ambient air quality data (2015–2020).
  - Daily city-level aggregation analysis and station-level profiling.
  - Multi-step feature extraction using historical lags ($T-1, T-2, T-7$) and rolling statistics.
  - Linear, Bagging, and Gradient Boosting regression models.
  - Residual diagnostics, error profiling across cities and pollution tiers.
  - Interactive analytical dashboard for scenario review and visual exploration.
- **OUT OF SCOPE (Explicit Non-Goals):**
  - Real-time operational forecasting engine connected to live IoT hardware.
  - Dispersion modeling or chemical transport physics simulations (e.g., WRF-Chem).
  - Prescribing clinical medical diagnoses or issuing statutory public health orders.
  - Fabricating station-level land-use or zoning classifications not present in source metadata.

---

### 4. Stakeholder Analysis & Intended Users

| Stakeholder Persona                                      | Core Needs & Information Demands                                                                                   | How This Solution Serves Them                                                                                   | Operational Caveat / Boundary                                                               |
| :------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ |
| **Environmental Planners & CPCB/SPCB Authorities** | Long-term trend analysis, seasonal peak anticipation, identification of chronic non-attainment zones.              | Quantifies seasonal recurrence of hazardous particulate episodes and ranks city pollutant severity.             | Prototype historical analytical tool; does not replace regulatory CPCB statutory reporting. |
| **Public Health Communication Teams**              | Advance notice of severe air quality days to schedule public advisories and school/outdoor activity warnings.      | Next-day AQI category classification provides calibrated probability alerts for 'Very Poor' and 'Severe' tiers. | Does not replace official medical alerts or localized hyper-local exposure monitoring.      |
| **Academic Researchers & Data Scientists**         | Reproducible baseline, benchmark datasets, transparent handling of environmental missingness and temporal leakage. | Open, fully documented codebase with strict chronological validation and honest residual error analysis.        | Demonstrates methodological rigor rather than commercial proprietary claims.                |
| **Informed Citizens & Civil Society**              | Intuitive understanding of historical pollution patterns, seasonal cycles, and city comparisons.                   | Interactive dashboard visualizing multi-year trends, dominant pollutants, and model forecast transparency.      | Educational and informative prototype based on historical 2015–2020 data.                  |

---

### 5. Functional & Analytical Requirements

#### 5.1 Ingestion & Data Quality Requirements

- **FR-01:** Ingest and profile all five canonical Kaggle CSV files: `city_day.csv`, `city_hour.csv`, `station_day.csv`, `station_hour.csv`, and `stations.csv`.
- **FR-02:** Detect duplicate timestamps, missing values, non-standard sensor encodings, and negative concentration anomalies.
- **FR-03:** Empirically audit sensor continuity per city and define candidate retention thresholds (e.g. evaluating 50%, 60%, 70% active coverage during operational periods) based on verified data distributions.

#### 5.2 Exploratory Data Analysis Requirements

- **FR-04:** Compute distribution parameters (mean, median, IQR, 95th percentile) for primary pollutants: $PM_{2.5}, PM_{10}, NO, NO_2, NO_x, NH_3, CO, SO_2, O_3$, Benzene, Toluene, Xylene.
- **FR-05:** Calculate and visualize seasonal decomposition (trend, seasonality, residuals) and Pearson/Spearman correlation matrices across chemical species.
- **FR-06:** Benchmark city-level AQI distributions against National Ambient Air Quality Standards (NAAQS) and CPCB color-coded hazard bands.

#### 5.3 Feature Engineering & Leakage Protection

- **FR-07:** Construct calendar features (Day of Week, Month, Quarter, Day of Year, Is_Weekend).
- **FR-08:** Generate temporal lag features ($AQI_{t-1}, AQI_{t-2}, AQI_{t-3}, AQI_{t-7}$) per city.
- **FR-09:** Generate backward-looking rolling statistics (7-day and 14-day rolling mean, min, max, standard deviation) using strict `closed='left'` or shifting to eliminate contemporaneous leakage.
- **FR-10:** Compute pollutant sub-index proxies available strictly at day $T-1$.

#### 5.4 Modeling & Validation Requirements

- **FR-11:** Implement Naive Baseline (Persistence Model: $\hat{y}_{t+1} = y_t$).
- **FR-12:** Implement Linear Baseline (Ridge Regression with cross-validated regularization).
- **FR-13:** Implement Non-Linear Ensembles (Random Forest Regressor, Histogram-based Gradient Boosting / LightGBM).
- **FR-14:** Enforce strictly chronological partitions:
  - Training: 2015-01-01 to 2018-06-30
  - Validation: 2018-07-01 to 2018-12-31 (Hyperparameter tuning)
  - Primary Held-Out Test: 2019-01-01 to 2019-12-31 (Representative 365-day full cycle)
  - Shock Evaluation Set: 2020-01-01 to 2020-07-01 (COVID-19 lockdown structural break)
- **FR-15:** Implement Secondary Classification Evaluation by mapping predicted continuous AQI into standard CPCB Buckets, generating Confusion Matrix, Balanced Accuracy, and Macro F1-score.

#### 5.5 Reporting & Dashboard Delivery

- **FR-16:** Interactive Streamlit dashboard presenting: Executive Overview, Data Quality Health Check, Spatial & Temporal EDA Explorer, Model Performance & Residual Diagnostics, and Interactive What-If Scenario Forecaster.
- **FR-17:** Complete academic project documentation suite.

---

### 6. Success Metrics & Acceptance Criteria

#### 6.1 Analytical & Quality Success Criteria (Deliverable Quality)

- **SC-01:** Complete schema inventory and quality audit covering 100% of available CSV files.
- **SC-02:** Zero data leakage in feature engineering (verified by unit tests confirming no future timestamps in lag/rolling computations).
- **SC-03:** 100% reproducible training and evaluation pipeline runnable via a single automated script or CLI command.
- **SC-04:** Minimum of 85% code test coverage across core data preprocessing, feature engineering, and evaluation modules.

#### 6.2 Measured Model Performance Targets (Empirical Benchmarking)

- **SC-05 (Baseline Superiority on 2019 Test Set):** Primary candidate ML model (Gradient Boosting / Random Forest) must demonstrate lower Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) than the Naive Persistence baseline on the representative 2019 chronological held-out test split.
- **SC-06 (Shock Resilience Diagnostics):** Quantify performance degradation between 2019 normal operations and 2020 lockdown operations, documenting exact failure modes.

---

### 7. Faculty Rubric Mapping (100 Marks)

|  Criterion No.  | Rubric Category                             |     Marks     | Project Implementation / Document Alignment                                                                       |
| :-------------: | :------------------------------------------ | :-----------: | :---------------------------------------------------------------------------------------------------------------- |
|   **1**   | Problem Definition & Business Understanding | **10** | PRD Section 2, Section 3, Section 4; Research Questions, Stakeholder definitions, and Environmental KPIs.         |
|   **2**   | Dataset & Data Understanding                | **10** | DATA_SCHEMA.md, Data Quality Audit, CPCB monitoring station schema, completeness statistics, dictionary.          |
|   **3**   | Data Preprocessing & Cleaning               | **10** | TRD Section 5, `src/preprocessing/`, missingness strategies, anomaly handling, spatial grouping.                |
|   **4**   | Exploratory Data Analysis (EDA)             | **10** | Statistical distributions, seasonal decomposition, correlation matrices, city comparisons, NAAQS benchmarking.    |
|   **5**   | Feature Engineering / Selection             | **10** | Date-time extraction, autoregressive lags, rolling statistics, leakage prevention unit tests, feature importance. |
|   **6**   | Model Selection & Implementation            | **15** | Naive baseline, Ridge, Random Forest, HistGradientBoosting, scikit-learn pipelines, chronological splitting.      |
|   **7**   | Model Evaluation & Validation               | **15** | Out-of-time test evaluation (2019 full year + 2020 shock), MAE, RMSE, $R^2$, Confusion Matrix for AQI classes.  |
|   **8**   | Business Insights & Recommendations         | **10** | Evidence-backed environmental policy recommendations, public health alert protocols, network limitations.         |
|   **9**   | Documentation & Presentation                | **10** | Six planning documents, README, Technical Report, PPT Deck outline, Viva defense guide, clean modular Git repo.   |
| **TOTAL** | **Nine Rubric Categories**            | **100** | **Complete coverage across all lifecycle phases without extraneous categories.**                            |

---

### 8. Definition of Done (DoD)

A phase or deliverable is defined as **Done** only when: 

1. Implementation code is modular, type-annotated, and documented in `src/`.
2. Execution is 100% reproducible from raw files to final figures without manual data manipulation.
3. Automated unit tests pass via `pytest`.
4. Analytical interpretations are backed by computed figures and tables rather than conjecture.
5. All relevant sections in the Requirements Traceability Matrix (`docs/requirements_traceability.md`) are verified and updated.
