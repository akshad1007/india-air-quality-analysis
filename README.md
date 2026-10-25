# India Air Quality Analysis and Next-Day AQI Forecasting

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Tests: 88 Passed](https://img.shields.io/badge/pytest-88%20passed-brightgreen.svg)](tests/)
[![Architecture: Modular](https://img.shields.io/badge/architecture-leak--free%20pipeline-orange.svg)](src/)
[![Documentation: 24--Page Report](https://img.shields.io/badge/academic%20report-24%20pages%20PDF-purple.svg)](docs/report/india_air_quality_project_report.pdf)

An end-to-end, reproducible Data Science and Machine Learning engineering project analyzing multi-year ambient air quality records across India (2015–2020) and benchmarking predictive models for next-day city-level Air Quality Index (AQI) forecasting with TreeSHAP explainability, conformal prediction uncertainty intervals, and a verified command-line inference engine.

---

## 1. Academic Project Identity

- **Project Title:** India Air Quality Analysis and Next-Day AQI Forecasting
- **Subject / Course Code:** `24UADPC302 — Predictive Modeling Using SPSS (IBM)`
- **Academic Term:** Semester 5 (Third Year B.Tech), Academic Year 2026–2027
- **Department:** Department of Artificial Intelligence and Data Science (TY AIDS)
- **Institution:** Sanjivani University, Kopargaon, Maharashtra, India
- **Faculty Guide:** Mr. Rohit Pawar, Assistant Professor, Department of AI & Data Science

### Project Team Members

| No. | Student Name | University Roll / PRN Number |
| :---: | :--- | :--- |
| 1 | **Akshad Makhana** | `2124UDSM1007` |
| 2 | **Shubham Pawar** | `2124UDSM1074` |
| 3 | **Yash Kolhe** | `2124UDSM1107` |
| 4 | **Pranav Hadole** | `2124UDSM1108` |

*Note on Course Code Identification:* The course designation `24UADPC302 — Predictive Modeling Using SPSS (IBM)` represents the institutional curriculum subject title. To guarantee modular reproducibility, continuous calendar data engineering, and automated testing, the production technical implementation is engineered in Python (`scikit-learn`, `pandas`, `numpy`, `joblib`, `pytest`, `shap`).

---

## 2. Problem Statement & Motivation

Ambient air pollution is a catastrophic public health challenge across urban India, contributing to an estimated 1.6+ million premature deaths annually. Although the Central Pollution Control Board (CPCB) operates an extensive Continuous Ambient Air Quality Monitoring Station (CAAQMS) network across urban centers, operational next-day forecasting faces several severe technical hurdles:

1. **Pervasive Temporal Data Leakage in Academic Literature:** Many published studies randomly partition time-series records or use contemporaneous day-$t$ predictors to predict day-$t$ AQI. Such formulations conceal future information, creating non-deployable models that collapse during prospective forecasting.
2. **Disregard for Naive Baselines:** Complex non-linear regression models are frequently promoted without proving that they outperform a simple, cost-free Naive Persistence baseline ($\text{AQI}_{t+1} = \text{AQI}_t$).
3. **Severe Telemetry Missingness & Intermittent Dropouts:** Sensor outages and reporting gaps reach 15% to 60% across primary criteria pollutants ($\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, $\text{O}_3$).
4. **Point Forecasts Lack Uncertainty Bounds:** Standard regression outputs omit probabilistic uncertainty, leaving municipal decision-makers without confidence bounds during volatile pollution spikes.

This project resolves these challenges by establishing a leak-free temporal feature engineering pipeline, chronological cross-validation, comprehensive out-of-time evaluation (2019 test and 2020 COVID-19 lockdown shock), Path-Dependent TreeSHAP explainability, and finite-sample conformal prediction intervals.

---

## 3. Key Capabilities

- **Strict Temporal Barrier:** Forecasts next-day continuous composite AQI ($\text{AQI}_{c, t+1}$) using strictly the information set available at or before prediction origin day $t$, enforcing municipal boundary isolation.
- **Predeclared Model Selection:** Evaluated Naive Persistence, Ridge Regression, Random Forest, and HistGradientBoosting across chronological training folds ($\le \text{June 2018}$) and selected the champion based on pre-declared validation MAE (July–December 2018).
- **Out-of-Time & Stress Testing:** Evaluated on an untouched 2019 held-out primary benchmark ($N=6,201$) and an exogenous stress-test partition during the 2020 nationwide COVID-19 lockdown ($N=3,124$).
- **Explainable AI (XAI):** Integrated Path-Dependent TreeSHAP to quantify global feature importances and local decision breakdowns for typical vs extreme pollution episodes.
- **Distribution-Free Uncertainty Quantification:** Implemented split-conformal prediction calibrated on pre-2019 validation residuals ($N=2,982$), delivering 90% and 95% prediction intervals with non-negative lower-bound truncation.
- **Audited Inference CLI:** Headless command-line tool supporting single-row JSON inference and atomic batch CSV processing with physical constraint verification and imputation disclosure.
- **Comprehensive Verification Suite:** 88 automated unit and integration tests passing with 100% success rate.

---

## 4. Dataset Source, Architecture & Scope

- **Dataset Source:** Multi-year CAAQMS surface monitoring records published by CPCB and compiled by Rohan Rao on [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india).
- **Temporal Span:** January 1, 2015 to July 1, 2020.
- **Geographic Coverage:** 18 Tier-1 qualified municipal jurisdictions with continuous multi-year telemetry: Ahmedabad, Amaravati, Amritsar, Bengaluru, Brajrajnagar, Chennai, Delhi, Gurugram, Hyderabad, Jaipur, Jorapokhar, Kolkata, Lucknow, Mumbai, Patna, Talcher, Thiruvananthapuram, and Visakhapatnam. (8 fragmented cities with $<40\%$ coverage were quarantined).
- **Primary Modeling Table:** `city_day.csv` (29,531 raw rows across 16 criteria columns).

---

## 5. Workflow and Leakage-Free Methodology

```
Raw CAAQMS Data (2015-2020)
       │
       ▼
Continuous Calendar Reindexing & Monotonic Date Sorting
       │
       ▼
Bounded Forward-Fill (max 2 consecutive days; Target Strictly NEVER Imputed)
       │
       ▼
Feature Engineering Matrix (19 Approved Predictors)
  ├── Group Isolation: Grouped by City (No cross-city leakage)
  ├── Autoregressive Lags: AQI(t), AQI(t-1), AQI(t-6)
  ├── Chemistry: PM2.5(t), PM10(t), NO2(t), SO2(t), CO(t), O3(t), PM_ratio(t)
  ├── Backward Rolling Moments: 3-day Mean, 7-day Mean, 7-day Std (closed at t)
  └── Target Calendar Harmonics: sin/cos Day-of-Year (365.25d), DOW, Weekend, Month
       │
       ▼
Chronological Partitions (Indexed strictly by Target Date t+1)
  ├── Training Partition:    Target Date <= 2018-06-30 (N = 10,958)
  ├── Validation Partition:  2018-07-01 to 2018-12-31 (N = 2,982, Shared Cohort A)
  ├── Primary Test Set:      2019-01-01 to 2019-12-31 (N = 6,201, Shared Cohort A)
  └── Stress Test Set:       2020-01-01 to 2020-07-01 (N = 3,124, COVID-19 Shock)
       │
       ▼
Model Selection (Predeclared Validation MAE) -> Frozen Random Forest Champion
       │
       ▼
Multi-Phase Evaluation (2019 Test + 2020 Stress + Ahmedabad Sensitivity)
       │
       ▼
Post-Modeling Layers: TreeSHAP Explainability + Conformal Prediction Intervals + CLI
```

---

## 6. Empirical Evaluation Results

### Model Performance Comparison on 2019 Held-Out Test Set (Shared Cohort A, $N=6,201$)

| Model Architecture | MAE (AQI pts) | RMSE (AQI pts) | $R^2$ Score | MBE (AQI pts) | MAE Impr. vs Persistence | RMSE Impr. vs Persistence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Persistence Baseline** | 31.0419 | 70.2360 | 0.7445 | +0.3614 | 0.00% | 0.00% |
| **Ridge Regression** ($\alpha=10.0$) | 29.0261 | 55.8918 | 0.8382 | +2.8414 | +6.49% | +20.42% |
| **HistGradientBoosting** | 27.1402 | **54.9397** | **0.8437** | +5.6109 | +12.57% | **+21.78%** |
| **Random Forest (Champion)** | **26.2993** | 55.6234 | 0.8398 | **+4.8666** | **+15.28%** | +20.81% |

*Key Findings:*
- **Random Forest** achieved the lowest MAE of **26.30 AQI points** (+15.28% error reduction over Persistence) and explained 83.98% of variance ($R^2 = 0.8398$).
- **CPCB Category Diagnostic Accuracy:** Random Forest attained **74.36%** multi-class accuracy (Macro F1 = 0.6678), significantly outperforming Persistence (69.21% accuracy, Macro F1 = 0.6057).
- **2020 COVID-19 Lockdown Generalization ($N=3,124$):** Random Forest maintained robust predictive skill with an MAE of **19.33 AQI points** (+10.64% improvement over Persistence 21.63), RMSE of **39.19**, and $R^2 = \mathbf{0.8060}$.

---

## 7. Ahmedabad Sensor Saturation Anomaly & Sensitivity Analysis

Residual diagnostics revealed that in 2019 Cohort A, **Ahmedabad accounts for only 347 rows (5.6% of records) but contributed 72.59% of total network squared error** (13.93M out of 19.19M points) due to extreme CO sensor saturation ($\text{CO} > 400\text{ mg/m}^3$) driving observed AQI past statutory limits ($>1,000$).

| Evaluation Subset | Model Pipeline | MAE (AQI pts) | RMSE (AQI pts) | $R^2$ Score | RMSE Impr. |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Full Network (18 Cities) | Random Forest | 26.2993 | 55.6234 | 0.8398 | +20.81% |
| **Without Ahmedabad (17 Cities)** | Naive Persistence | 26.8370 | 44.4082 | 0.7644 | 0.00% |
| **Without Ahmedabad (17 Cities)** | **Random Forest** | **21.6508** | **29.9678** | **0.8931** | **+32.52%** |

Excluding the saturated sensor cohort reveals that across the remaining 17 municipal jurisdictions, Random Forest RMSE is **29.97 AQI points** with an $R^2$ of **0.8931**.

---

## 8. Explainability via Path-Dependent TreeSHAP

Model interpretability was formalized using TreeSHAP on the frozen Random Forest pipeline:
- **Dominant Predictive Drivers:** Observed same-day $\text{AQI}_t$ (mean $|\text{SHAP}| = 35.63\text{ AQI points}$) and $\text{PM25}_t$ (mean $|\text{SHAP}| = 16.59$) govern 66.8% of global model variance.
- **Secondary Modulators:** 7-day backward rolling average ($\text{AQI\_roll\_mean\_7d}$, mean $|\text{SHAP}| = 5.70$) and seasonal harmonics ($\text{target\_cos\_doy}$, mean $|\text{SHAP}| = 4.29$) provide momentum and seasonal adjustment.
- **Mathematical Additivity Verification:** For every evaluation sample, local additivity $f(x) = \phi_0 + \sum \phi_j(x)$ matched direct pipeline predictions within $5.91 \times 10^{-12}$ AQI points.

Visual assets are archived in `reports/figures/`:
- Global Summary Beeswarm: [`reports/figures/shap_summary.png`](reports/figures/shap_summary.png)
- Feature Importance Bar Chart: [`reports/figures/shap_importance.png`](reports/figures/shap_importance.png)
- Typical Day Waterfall Breakdown: [`reports/figures/shap_waterfall_typical.png`](reports/figures/shap_waterfall_typical.png)
- Extreme Episode Waterfall Breakdown: [`reports/figures/shap_waterfall_extreme.png`](reports/figures/shap_waterfall_extreme.png)

---

## 9. Prediction Uncertainty via Conformal Prediction

Using split-conformal prediction calibrated on pre-2019 validation absolute residuals ($N=2,982$), finite-sample quantiles were established:
- **90% Conformal Quantile:** $q_{0.90} = \mathbf{57.0621\text{ AQI points}}$ ($k = 2685$)
- **95% Conformal Quantile:** $q_{0.95} = \mathbf{90.3847\text{ AQI points}}$ ($k = 2834$)

Intervals are constructed as $[ \max(0.0, \widehat{y} - q), \widehat{y} + q ]$. Lower bounds are truncated at 0.0 (non-negative continuous AQI); upper bounds are **unclipped at 500** to preserve continuous risk assessment.

| Evaluation Cohort | Nominal Level | Conformal Margin | Empirical Coverage | Mean Interval Width |
| :--- | :---: | :---: | :---: | :---: |
| **2019 Primary Test** ($N=6,201$) | 90% | $\pm 57.0621$ | **90.21%** | 113.90 AQI points |
| **2019 Primary Test** ($N=6,201$) | 95% | $\pm 90.3847$ | **94.97%** | 175.08 AQI points |
| **2020 Stress Test** ($N=3,124$)  | 90% | $\pm 57.0621$ | **94.40%** | 113.69 AQI points |
| **2020 Stress Test** ($N=3,124$)  | 95% | $\pm 90.3847$ | **98.08%** | 172.10 AQI points |

### Critical Methodological Limitations
1. **Exchangeability in Time Series:** Conformal coverage guarantees assume exchangeable observations. Autocorrelation and seasonal drift introduce distribution shift; intervals represent empirical calibration baselines rather than asymptotic guarantees.
2. **Extreme Event Under-Coverage:** For volatile observations where observed AQI exceeded 500, empirical interval coverage degraded to **29.33%**. Prediction intervals are **not guaranteed reliable during extreme episodes**.

---

## 10. Inference Command-Line Interface (CLI)

The repository includes a headless command-line inference tool in [src/models/predict_cli.py](src/models/predict_cli.py).

### Single-Row Prediction (JSON Input)
```powershell
python -m src.models.predict_cli --input-json data/sample_delhi.json
```
```text
================================================================================
INDIA AIR QUALITY INTELLIGENCE — NEXT-DAY AQI FORECASTING CLI
Model: Frozen Random Forest Pipeline (Phase 7 Champion)
================================================================================
City:                 Delhi (Tier-1 Qualified)
Target Date:          2019-01-16
Point Forecast (AQI): 278.45
CPCB Category:        Poor
--------------------------------------------------------------------------------
Uncertainty Quantification (Conformal Prediction Intervals):
  90% Prediction Interval: [221.39, 335.51] (Margin: ±57.06)
  95% Prediction Interval: [188.07, 368.83] (Margin: ±90.38)
--------------------------------------------------------------------------------
DISCLAIMER: Prediction intervals represent finite-sample conformal calibration on the
pre-2019 validation set (N=2,982). Intervals are not guaranteed reliable during
extreme episodes (AQI > 500 coverage degrades to 29.33%) or severe distribution drift.
================================================================================
```

### Batch Prediction (CSV Input with Output File)
```powershell
python -m src.models.predict_cli --input-csv data/batch_features.csv --output-file reports/forecasts.csv
```

### JSON Formatted Output
```powershell
python -m src.models.predict_cli --input-json data/sample_delhi.json --format json
```

### Operational Guardrails Enforced by CLI
- **Non-Negativity Constraint:** Rejects negative physical concentrations ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3 < 0$) with error code 2.
- **Transparent Imputation:** Missing numerical features trigger pipeline median imputation, disclosed explicitly in the output.
- **Unknown-City Extrapolation Warning:** Unseen cities encode to an all-zero vector; the CLI emits an explicit extrapolation notice.
- **Model Immutability:** Loaded in strictly read-only mode via `load_frozen_pipeline()`; never calls `.fit()` or alters parameters.

---

## 11. Academic Project Report

The complete, authoritative academic project report is compiled in LaTeX using XeLaTeX with **Times New Roman** typography and the institutional header:

- **Compiled PDF Report (24 Pages):** [`docs/report/india_air_quality_project_report.pdf`](docs/report/india_air_quality_project_report.pdf)
- **LaTeX Source File:** [`docs/report/india_air_quality_project_report.tex`](docs/report/india_air_quality_project_report.tex)
- **Bibliography File:** [`docs/report/references.bib`](docs/report/references.bib)

### How to Recompile the LaTeX Report
Using XeLaTeX with MiKTeX:
```powershell
cd docs/report
bibtex india_air_quality_project_report
xelatex -interaction=nonstopmode india_air_quality_project_report.tex
xelatex -interaction=nonstopmode india_air_quality_project_report.tex
cd ../..
```

---

## 12. Repository Directory Structure

```text
SPSS PROJECT/
├── README.md                                  # Top-level GitHub landing page & project documentation
├── pytest.ini                                 # Pytest configuration file
├── college-logo.png                           # Sanjivani University banner logo
│
├── docs/                                      # Project documentation and reports
│   ├── report/                                # Academic project report (LaTeX + PDF)
│   │   ├── india_air_quality_project_report.pdf  # Final compiled 24-page report PDF
│   │   ├── india_air_quality_project_report.tex  # Complete editable LaTeX source
│   │   ├── references.bib                     # Academic bibliography database
│   │   ├── college-logo.png                   # Institutional logo asset
│   │   └── figures/                           # High-resolution report figure assets (14 PNGs)
│   ├── reports/                               # Milestone engineering markdown reports
│   │   ├── PHASE_7_MODELING_REPORT.md         # Model selection & chronological tuning
│   │   ├── PHASE_8_EVALUATION_REPORT.md       # Final 2019 test & 2020 stress evaluation
│   │   ├── PHASE_9_INTERPRETABILITY_REPORT.md # TreeSHAP explainability engine report
│   │   └── PHASE_9_UNCERTAINTY_REPORT.md      # Conformal prediction quantification report
│   └── review/                                # Milestone readiness and acceptance audit logs
│
├── models/                                    # Serialized, immutable machine learning models
│   ├── random_forest_val_model.joblib         # Champion Random Forest pipeline (SHA-256 verified)
│   ├── ridge_regression_val_model.joblib      # Baseline Ridge pipeline
│   └── histgradientboosting_val_model.joblib  # Comparator HistGBM pipeline
│
├── reports/                                   # Empirical metrics, evaluation CSVs, and figures
│   ├── figures/                               # 14 publication-grade analytical PNG plots (300 DPI)
│   ├── model_selection_summary.csv            # Phase 7 validation period metric comparison
│   ├── phase8_test_metrics.csv                # Primary 2019 test regression metrics
│   ├── phase8_stress_metrics.csv              # 2020 COVID-19 lockdown stress metrics
│   ├── phase8_classification_metrics.csv      # CPCB category diagnostic metrics
│   ├── phase8_ahmedabad_sensitivity.csv       # Ahmedabad anomaly sensitivity analysis
│   ├── phase9_uncertainty_metrics.csv         # Conformal interval coverage & width metrics
│   ├── phase9_city_coverage.csv               # Municipal-level empirical coverage table
│   └── shap_feature_importance.csv            # Global mean absolute SHAP values (36 features)
│
├── src/                                       # Modular Python engineering source code
│   ├── features/
│   │   └── feature_builder.py                 # Leak-free temporal feature engineering
│   └── models/
│       ├── baseline.py                        # Naive Persistence baseline model
│       ├── evaluation.py                      # Multi-metric evaluation & CPCB diagnostic mapper
│       ├── interpretability.py                # Path-Dependent TreeSHAP explainability engine
│       ├── metrics.py                         # Authoritative regression & classification metrics
│       ├── pipelines.py                       # Supervised pipeline definitions
│       ├── predict_cli.py                     # Headless inference CLI interface
│       ├── train.py                           # Chronological model training & tuning
│       ├── uncertainty.py                     # Split-conformal uncertainty quantification
│       └── validation.py                      # Expanding-window temporal cross-validation
│
├── submission/                                # Frozen academic submission milestone
│   └── phase8_academic_submission/            # 37-file self-contained Phase 8 submission package
│
└── tests/                                     # Automated test suite (88 tests, 100% passing)
    ├── test_data_validation.py                # Raw data integrity & checksum tests
    ├── test_eda_artifacts.py                  # EDA visualization tests
    ├── test_evaluation.py                     # Evaluation metric & CPCB mapping tests
    ├── test_features.py                       # Temporal barrier & lag continuity tests
    ├── test_interpretability.py               # TreeSHAP additivity & background tests
    ├── test_models.py                         # Model reproducibility & predictor restriction tests
    ├── test_predict_cli.py                    # Inference CLI interface tests (14 tests)
    ├── test_preprocessing.py                  # Calendar reindexing & non-imputation tests
    └── test_uncertainty.py                    # Conformal quantile & interval bounds tests
```

---

## 13. Getting Started & Reproducibility Guide

### 1. Environment Prerequisites
- Python 3.10.x
- Git

### 2. Clone the Repository
```powershell
git clone https://github.com/akshad1007/india-air-quality-analysis.git
cd india-air-quality-analysis
```

### 3. Install Dependencies
```powershell
pip install pandas pyarrow scipy matplotlib seaborn scikit-learn joblib shap pytest
```

### 4. Run the Full Test Suite (88 Passing Tests)
```powershell
pytest tests/ -v
```
*Expected Result: 88 passed in ~20-25 seconds (zero failures, zero warnings).*

### 5. Run the Focused CLI Test Suite
```powershell
pytest tests/test_predict_cli.py -v
```
*Expected Result: 14 passed in ~7-9 seconds.*

---

## 14. Academic Limitations and Responsible Use

1. **Meteorological Limitations:** Surface CAAQMS telemetry records criteria pollutants only. Atmospheric boundary layer height, relative humidity, and wind velocity are literature-based hypotheses and were not incorporated as real-time features.
2. **Exchangeability in Environmental Time Series:** Conformal intervals serve as calibrated empirical baselines. Temporal autocorrelation and meteorological regime shifts can cause coverage drift.
3. **Extreme Pollution Events:** Prediction intervals are homoscedastic constant-width bands (subject to zero-truncation). For peak episodes where AQI exceeds 500, empirical interval coverage drops to 29.33%. **Intervals must not be interpreted as guaranteed bounds during severe smog emergencies.**
4. **Non-Causal Interpretation:** Performance differentials between 2019 and 2020 coincide with the COVID-19 mobility restrictions, but cannot be causally proven from observational telemetry alone without atmospheric chemistry transport models.

---

## 15. Citation and Attribution

- **Original Data Source:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India.
- **Dataset Curator:** Rohan Rao, published on [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india).
- **Academic Context:** Developed as a Problem-Based Learning (PBL) project in `24UADPC302: Predictive Modeling Using SPSS (IBM)` at the Department of AI & Data Science, Sanjivani University, Kopargaon.
