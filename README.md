# India Air Quality Analysis and Next-Day AQI Forecasting

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 17 Passed](https://img.shields.io/badge/pytest-17%20passed-brightgreen.svg)](tests/)
[![Code Style: Modular](https://img.shields.io/badge/code%20style-pep8-orange.svg)](src/)

An end-to-end, reproducible Data Science and Machine Learning Problem-Based Learning (PBL) project analyzing multi-year ambient air quality records across India (2015–2020) and benchmarking predictive models for next-day city-level Air Quality Index (AQI) forecasting.

---

## 1. Academic Project Identity & Institutional Metadata

- **Project Title:** India Air Quality Analysis and Next-Day AQI Forecasting
- **Subject Name:** Predictive Modeling Using SPSS (IBM)
- **Subject Code:** 24UADPC302
- **Semester:** 5 (Third Year B.Tech)
- **Department:** Department of Artificial Intelligence and Data Science (TY AIDS)
- **Institution:** Sanjivani University, Kopargaon, Maharashtra, India
- **Faculty Guide:** Mr. Rohit Pawar

### Project Team Members

| No. | Student Name | University Roll / PRN Number |
| :---: | :--- | :--- |
| 1 | **Akshad Makhana** | `2124UDSM1007` |
| 2 | **Shubham Pawar** | `2124UDSM1074` |
| 3 | **Yash Kolhe** | `2124UDSM1107` |
| 4 | **Pranav Hadole** | `2124UDSM1108` |

*Note on Academic Subject Identification:* The subject title `24UADPC302 — Predictive Modeling Using SPSS (IBM)` represents the official university curriculum course code under which this Problem-Based Learning (PBL) project is evaluated. The engineering implementation is developed in Python to guarantee modular reproducibility, continuous calendar data engineering, and automated testing, strictly adhering to the AIDS Department 100-mark assessment rubric.

### Faculty PBL Guidelines & Reference Documents
The project is governed strictly by the authoritative guidelines provided by the faculty, archived in [`docs/reference/faculty-guidelines/`](docs/reference/faculty-guidelines/):
- [`Assignment_4144_Content_Document_20261019020930913PM.pdf`](docs/reference/faculty-guidelines/Assignment_4144_Content_Document_20261019020930913PM.pdf) (PBL Assessment Guidelines)
- [`Assignment_4144_Content_SolutionDocument_20261019020930913PM (1).pdf`](docs/reference/faculty-guidelines/Assignment_4144_Content_SolutionDocument_20261019020930913PM%20(1).pdf) (100-Mark Rubric Document)

---

## 2. Problem Statement & Motivation

Air pollution represents one of the most critical public health emergencies in modern India. According to epidemiological research (WHO, Global Burden of Disease), chronic exposure to ambient particulate aerosols ($PM_{2.5}$, $PM_{10}$) and noxious combustion gases ($NO_2$, $SO_2$, $CO$, $O_3$) accounts for over 1.6 million premature annual deaths nationwide.

Although the Central Pollution Control Board (CPCB) maintains an extensive Continuous Ambient Air Quality Monitoring Station (CAAQMS) network, raw observational datasets present severe analytical and operational challenges:
1. **Severe Sensor Missingness & Intermittent Dropouts:** Sensor outages and reporting gaps reach 15% to 60% across primary pollutants.
2. **Temporal & Seasonal Volatility:** Winter temperature inversions, post-monsoon biomass burning, and monsoon scavenging cause dramatic seasonal swings.
3. **Pervasive Data Leakage in Existing Literature:** Many academic studies randomly split time-series records or use contemporaneous day-$t$ predictors to predict day-$t$ AQI, creating unachievable, non-deployable models.
4. **Disregard for Naive Baselines:** Complex non-linear architectures are often reported without demonstrating whether they outperform a simple persistence benchmark (Tomorrow's AQI = Today's AQI).

This project addresses these gaps through rigorous calendar-based preprocessing, leak-free feature pipelines, verified exploratory data analysis, and out-of-time predictive benchmarking.

---

## 3. Project Objectives & Research Questions

### Primary Objective
To design, implement, audit, and document a scientifically defensible machine learning pipeline that forecasts **next-day city-level AQI ($AQI_{t+1}$)** across representative Indian urban centers, evaluating whether supervised machine learning models reliably outperform naive persistence baselines under realistic out-of-time evaluation.

### Empirical Research Questions (RQs)
- **RQ1 (Historical Trends & Macro Patterns):** How have national ambient pollution levels and statutory CPCB health risk categories evolved from 2015 to 2020?
- **RQ2 (Spatial & Station Heterogeneity):** What is the empirical magnitude of inter-city and intra-city pollution dispersion across monitored CAAQMS stations?
- **RQ3 (Seasonal & Monthly Dynamics):** What is the empirical annual cycle of particulate matter and chemical precursors across Indian air sheds?
- **RQ4 (Chemical Drivers & Collinearity):** Which specific pollutants govern the composite CPCB AQI, and what collinearity structures exist among combustion tracers?
- **RQ5 (Exogenous Shock Resilience):** How did observed ambient air quality respond during the nationwide March–May 2020 COVID-19 lockdown across major metropolitan centers?
- **RQ6 (Predictive Feasibility & Benchmark Comparison - Planned):** Can autoregressive and tree-based machine learning models outperform naive persistence on held-out 2019 out-of-time data without future temporal leakage?

---

## 4. Dataset Source, Architecture & Provenance

The project utilizes the multi-year national air quality monitoring dataset published by CPCB and compiled by Rohan Rao on Kaggle:
- **Dataset Source:** [Air Quality Data in India (2015–2020) — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Date Coverage:** January 1, 2015 to July 1, 2020
- **Geographic Coverage:** 26 monitored Indian urban centers across 16 states and union territories.

### Raw Dataset Inventory

| File Name | File Size | Exact Rows | Columns | Purpose in Pipeline | Verified SHA-256 Checksum |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `stations.csv` | 14.4 KB | 230 | 5 | Master station metadata (Name, City, State, Status) | `ce2b7081c3e3c89145613eb724d7d2149ef79c6e16f36bb8a835747a673ed410` |
| `city_day.csv` | 2.57 MB | 29,531 | 16 | Primary modeling file: daily city-level pollutant aggregations | `0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847` |
| `city_hour.csv` | 65.6 MB | 707,875 | 16 | Hourly city-level records: diurnal cycles and rush-hour dynamics | `fae474062ec6b10af7c5b2ac4e32f5c240a0ef80835ee3a50332fec06e2190a4` |
| `station_day.csv` | 8.63 MB | 108,035 | 16 | Station-level daily records: intra-city spatial heterogeneity | `f07787eb4c36c53b07763f591ee29fa01fbaa4d18df1a470447d04def79677b9` |
| `station_hour.csv` | 219.6 MB | 2,589,083 | 16 | Station-level hourly records: hyper-local temporal resolution | `26a423c66fe9763064d580ffb007150375cfa2056e62b29d52b1b7857af54fde` |

*Data Integrity Guarantee:* All five raw CSV files remain strictly immutable in `data/raw/`. Checksums are verified programmatically prior to each pipeline run.

---

## 5. Technology Stack

| Layer | Tools & Libraries | Purpose |
| :--- | :--- | :--- |
| **Language & Runtime** | Python 3.10.11 | Core analytical and engineering execution |
| **Data Engineering** | pandas 2.3.2, pyarrow, numpy | Vectorized transformations, parquet storage, continuous reindexing |
| **Statistical Analysis** | scipy 1.15.2 | Parametric & non-parametric tests, Welch's t-tests, skewness metrics |
| **Visualization** | matplotlib 3.10.0, seaborn 0.13.2 | Publication-quality static figures (300 DPI) and analytical plots |
| **Machine Learning (Planned)** | scikit-learn 1.7.2 | Linear models, RidgeCV, Random Forest, HistGradientBoosting, pipelines |
| **Testing & CI** | pytest 7.4.3 | Automated regression, schema validation, and artifact verification |
| **Exploratory Interface** | Jupyter Notebook, nbconvert | Interactive data science documentation and clean-kernel execution |

---

## 6. End-to-End Methodology & Workflow

```
+----------------------------------------------------------------------------------------------------+
|                                  PROJECT PIPELINE ARCHITECTURE                                     |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Phase 2: Raw Acquisition]   ──>  5 CPCB CSVs immutable in data/raw/ (SHA-256 verified)           |
|                                                                                                    |
|  [Phase 3: Data Quality]      ──>  Missingness profiling, foreign-key auditing, gap analysis       |
|                                                                                                    |
|  [Phase 4: Preprocessing]     ──>  Continuous calendar grid (2015-01-01 to 2020-07-01)            |
|                                    Bounded ffill (<=2 days) on predictors strictly within city     |
|                                    Target AQI NEVER imputed (24,850 valid days preserved)          |
|                                    City eligibility matrix (18 Tier-1 modeling cities)             |
|                                                                                                    |
|  [Phase 5: Audited EDA]       ──>  9 publication figures (reports/figures/)                        |
|                                    3 summary tables (reports/tables/)                              |
|                                    Executed notebook (notebooks/02_exploratory_data_analysis.ipynb)|
|                                    Scientific quality audit (docs/review/EDA_QUALITY_AUDIT.md)     |
|                                                                                                    |
|  [Phase 6: Feature Eng. (PLANNED)] ──> Calendar cyclical (sin/cos), lags (T-1, T-2, T-7),          |
|                                        Rolling stats (T-6..T), zero future data leakage            |
|                                                                                                    |
|  [Phase 7-9: ML Modeling (PLANNED)]──> Chronological split: Train (2015-18), Test (2019 full year) |
|                                        Naive persistence benchmark vs. Ridge, RF, HistGBM          |
|                                        Out-of-time evaluation (MAE, RMSE, R2, Confusion Matrix)    |
|                                        COVID-19 exogenous shock stress test (2020)                 |
+----------------------------------------------------------------------------------------------------+
```

---

## 7. Data Quality & Preprocessing Methodology

### Continuous Calendar Reindexing
Real-world sensor logs often omit days entirely rather than logging explicit `NaN` entries. To ensure temporal consistency for time-series operations, the dataset was reindexed to a complete, uninterrupted calendar sequence ($2,009$ consecutive days per city from January 1, 2015 to July 1, 2020), expanding the raw $29,531$ city-days into a standardized continuous matrix.

### Bounded Imputation vs. Target Integrity
1. **Target Preservation Principle:** The target variable ($AQI$) is **NEVER imputed**. Imputing target labels fabricates synthetic ground truth and invalidates evaluation metrics. Exactly $24,850$ real observed AQI target days are preserved.
2. **Predictor Imputation (Variant A):** Predictor pollutant concentrations are imputed using **bounded forward-fill strictly capped at $\le 2$ consecutive days** and grouped strictly by city. Gaps longer than 2 days remain `NaN` to avoid propagating stale atmospheric states.
3. **Un-imputed Benchmark (Variant B):** Raw observations with missing values preserved intact for tree models with native missing-data support (`HistGradientBoostingRegressor`).

### City Eligibility Stratification
Cities were categorized by historical continuity rather than arbitrary exclusion:
- **Tier 1 (Primary Modeling Cohort — 18 Cities):** $\ge 60\%$ coverage across Train (2015–2018) and Test (2019). Includes Delhi, Bengaluru, Lucknow, Chennai, Hyderabad, Patna, Mumbai, Kolkata, Ahmedabad.
- **Tier 2 (Secondary Test Cohort — 1 City):** Monitored during 2019 test year but lacking historical training depth (Chandigarh).
- **Tier 3 (EDA-Only Cohort — 7 Cities):** Shorter records utilized for regional spatial comparisons but excluded from historical time-series training (e.g., Aizawl, Shillong, Coimbatore).

---

## 8. Verified Exploratory Data Analysis Findings

Every finding below has been verified through executable code in `src/analysis/eda_generator.py` and audited in `docs/review/EDA_QUALITY_AUDIT.md`:

### 1. National AQI Distribution & Statutory Categories (`Figure 01`)
- Out of $24,850$ valid observations, **$35.5\%$ fall into Moderate ($101 - 200$)**, **$33.1\%$ Satisfactory ($51 - 100$)**, **$11.2\%$ Poor ($201 - 300$)**, **$9.4\%$ Very Poor ($301 - 400$)**, **$5.4\%$ Severe ($> 400$)**, and **$5.4\%$ Good ($0 - 50$)**.
- Over $26\%$ of all monitored days fall into hazardous tiers (Poor or worse), illustrating chronic population exposure.

### 2. City-Level Spatial Disparities (`Figure 02`)
- Northern inland basin centers record the highest median AQI: **Delhi (Median $258.0$, Mean $259.5$)**, **Ahmedabad (Median $246.0$, Mean $452.1$)**, **Patna (Median $187.0$)**, **Gurugram (Median $183.0$)**, **Lucknow (Median $182.0$)**.
- Peninsular and northeastern centers record the lowest median AQI: **Aizawl (Median $31.0$, $N=111$)**, **Thiruvananthapuram (Median $68.0$, $N=1,053$)**, **Shillong (Median $68.0$, $N=205$)**, **Coimbatore (Median $73.0$, $N=384$)**, **Bengaluru (Median $90.0$, $N=1,911$)**, **Chennai (Median $96.0$, $N=1,884$)**.
- *Hypotheses:* Regional variations reflect differences in basin topography, atmospheric ventilation, and regional source loadings (acknowledged as literature-based hypotheses since meteorological variables are unmeasured).

### 3. Seasonal Dynamics & Monthly Wave (`Figure 03`)
- National air quality exhibits a predictable bimodal annual wave:
  - **Monsoon Annual Minimum:** July (Median $75.0$, Mean $89.7$) and August (Median $71.0$, Mean $85.0$).
  - **Winter Annual Peak:** November (Median $200.0$, Mean $239.5$) and December (Median $196.0$, Mean $229.8$).

### 4. Chemical Drivers & Collinearity (`Figure 04`)
- Particulate matter is the dominant driver of composite AQI: **$PM_{2.5}$ (Pearson $r = 0.66$, Spearman $r_s = 0.87$)** and **$PM_{10}$ (Pearson $r = 0.80$, Spearman $r_s = 0.83$)**.
- Gaseous combustion precursors act as collinear tracers: $CO$ ($r_s = 0.68$), $NO_2$ ($r_s = 0.54$), $NO_x$ ($r_s = 0.52$).
- Strong co-emission collinearity exists between $NO$ and $NO_x$ ($r = 0.79$) and $PM_{2.5}$ and $PM_{10}$ ($r = 0.85$).

### 5. Ahmedabad CO / Extreme AQI Anomaly Diagnostic (`Figure 05`)
- Exactly **543 city-days nationwide exceed $AQI > 500$**, with **412 of these records (75.9%) occurring exclusively in Ahmedabad**.
- Ahmedabad recorded extreme Carbon Monoxide concentrations (Median $16.25\text{ mg/m}^3$, Max $175.81\text{ mg/m}^3$) compared to Delhi (Median $1.24\text{ mg/m}^3$) and Bengaluru (Median $0.92\text{ mg/m}^3$).
- Applying the CPCB linear breakpoint formula without an upper 500 cap ($I = 400 + 6.25(CO - 34)$) mathematically extrapolates a $CO$ reading of $132\text{ mg/m}^3$ to $AQI \approx 1,012$ and reaches $2,049$. Raw data is preserved; modeling will use robust loss functions (Huber/MAE).

### 6. Seasonally Matched COVID-19 Comparison (`Figure 06`)
- Comparing the exact national lockdown window (March 25 – May 31) against the identical calendar window across the 2015–2019 baseline reveals significant reductions across all 8 major metros:
  - **Delhi:** Baseline Mean $276.2$ ($N_b=340$) $\rightarrow$ Lockdown Mean $125.4$ ($N_l=68$) | **$-54.6\%$ drop** ($p = 3.72 \times 10^{-60}$)
  - **Lucknow:** Baseline Mean $204.5$ ($N_b=322$) $\rightarrow$ Lockdown Mean $111.9$ ($N_l=68$) | **$-45.3\%$ drop** ($p = 1.07 \times 10^{-31}$)
  - **Chennai:** Baseline Mean $119.8$ ($N_b=336$) $\rightarrow$ Lockdown Mean $69.6$ ($N_l=68$) | **$-41.9\%$ drop** ($p = 1.06 \times 10^{-39}$)
  - **Hyderabad:** Baseline Mean $116.1$ ($N_b=329$) $\rightarrow$ Lockdown Mean $71.8$ ($N_l=68$) | **$-38.2\%$ drop** ($p = 6.97 \times 10^{-41}$)
  - **Patna:** Baseline Mean $189.3$ ($N_b=218$) $\rightarrow$ Lockdown Mean $133.9$ ($N_l=68$) | **$-29.3\%$ drop** ($p = 7.03 \times 10^{-20}$)
  - **Bengaluru:** Baseline Mean $99.4$ ($N_b=336$) $\rightarrow$ Lockdown Mean $70.6$ ($N_l=68$) | **$-29.0\%$ drop** ($p = 2.08 \times 10^{-33}$)
  - **Kolkata:** Baseline Mean $92.3$ ($N_b=75$) $\rightarrow$ Lockdown Mean $65.8$ ($N_l=68$) | **$-28.7\%$ drop** ($p = 1.11 \times 10^{-7}$)
  - **Mumbai:** Baseline Mean $91.4$ ($N_b=93$) $\rightarrow$ Lockdown Mean $69.8$ ($N_l=68$) | **$-23.6\%$ drop** ($p = 2.26 \times 10^{-16}$)
- *Scientific Guardrail:* Monocausal attribution to lockdown is avoided; unmeasured meteorological factors also influence ambient levels.

### 7. Diurnal Rush-Hour Cycles across Complete Hourly Records (`Figure 08`)
- Processed all **707,875 records** from `city_hour.csv` using chunked streaming, extracting **164,079 valid records** across Delhi, Kolkata, Mumbai, and Bengaluru.
- Delhi exhibits a distinct bimodal diurnal cycle: Morning rush-hour peak at 08:00–09:00 LST (Mean $PM_{2.5} \approx 135.2\ \mu\text{g/m}^3$), mid-afternoon convective dispersion minimum at 15:00–16:00 LST (Mean $PM_{2.5} \approx 72.4\ \mu\text{g/m}^3$, a $46.4\%$ drop), and nighttime inversion surge between 21:00 and 00:00 LST (Mean $PM_{2.5} \approx 154.8\ \mu\text{g/m}^3$).

### 8. Intra-City Delhi Station Dispersion (`Figure 09`)
- Using neutral station designations from `stations.csv`:
  - `Anand Vihar, Delhi - DPCC` (East Delhi): Median $329.0$, Mean $355.8$, Max $927.0$, $N=1,583$
  - `Punjabi Bagh, Delhi - DPCC` (West Delhi): Median $312.0$, Mean $294.3$, Max $769.0$, $N=1,472$
  - `R K Puram, Delhi - DPCC` (South-West Delhi): Median $230.0$, Mean $247.4$, Max $838.0$, $N=1,693$
  - `Mandir Marg, Delhi - DPCC` (Central Delhi): Median $193.0$, Mean $221.2$, Max $734.0$, $N=1,379$
  - `Pusa, Delhi - IMD` (Central/West Delhi): Median $148.0$, Mean $184.8$, Max $769.0$, $N=928$
- Observed station medians differ by $181.0\text{ AQI points}$ within Delhi, demonstrating that city-level aggregations mask substantial localized exposure variations.

---

## 9. Proposed Next-Day Forecasting Task & Benchmarking Plan *(PLANNED — Phase 6–9)*

> [!IMPORTANT]
> The forecasting pipeline is currently **PLANNED** and will be implemented upon faculty approval of Phase 6. No model performance metrics, test scores, or real-time inferences are claimed at this stage.

### Forecasting Task Formulation
- **Objective:** Predict continuous next-day composite AQI ($AQI_{t+1}$) for each city using observations available up to day $t$.
- **Input Features (Day $t$ and Earlier Only):**
  - Autoregressive target lags: $AQI_{t-1}, AQI_{t-2}, AQI_{t-7}$
  - Chemical precursor lags: $PM2.5_t, PM10_t, NO2_t, CO_t, SO2_t, O3_t$
  - Backward-looking rolling statistics: 3-day and 7-day rolling means and standard deviations computed exclusively over $[t-2, t]$ and $[t-6, t]$.
  - Cyclical calendar features: $\sin\left(\frac{2\pi \cdot \text{dayofyear}}{365.25}\right), \cos\left(\frac{2\pi \cdot \text{dayofyear}}{365.25}\right)$, day of week, month.

### Leakage Prevention & Chronological Splitting Strategy
Random splitting is strictly prohibited. The evaluation framework utilizes chronological time splits:
1. **Training Set (Historical Inception to Mid-2018):** 2015-01-01 to 2018-06-30.
2. **Validation / Tuning Set (Late 2018):** 2018-07-01 to 2018-12-31 (used for hyperparameter tuning and model selection via TimeSeriesSplit).
3. **Primary Held-Out Test Set (Full Calendar Year 2019):** 2019-01-01 to 2019-12-31. Provides 365 consecutive days across 14 cities without COVID distortions.
4. **Exogenous Stress Test (Early 2020):** 2020-01-01 to 2020-07-01. Evaluated separately to assess model resilience under extreme shock conditions.

### Candidate Models for Benchmarking
1. **Naive Persistence Benchmark (Mandatory Baseline):** $\widehat{AQI}_{t+1} = AQI_t$. Any proposed ML model must achieve lower MAE and RMSE than persistence to be considered viable.
2. **Linear Regularized Baseline:** Ridge Regression (`RidgeCV`) with standardized features.
3. **Non-Linear Ensemble Models:** Random Forest Regressor (`RandomForestRegressor`) and Histogram-based Gradient Boosting (`HistGradientBoostingRegressor`).

### Planned Evaluation Metrics
- **Continuous Regression Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), Coefficient of Determination ($R^2$).
- **Statutory Classification Thresholding:** Derived CPCB Health Category accuracy, macro-averaged F1-score, and normalized confusion matrix.

---

## 10. Repository Directory Structure

```
SPSS PROJECT/
│
├── data/
│   ├── raw/                           # Immutable raw CPCB CSV files (ignored in Git; .gitkeep tracked)
│   ├── interim/                       # Reindexed continuous grid, eligibility matrix, audit JSONs
│   └── processed/                     # Preprocessed Parquet variants (ffill & native; .gitkeep tracked)
│
├── docs/
│   ├── planning/                      # Audited specification documents (PRD, TRD, APP_FLOW, etc.)
│   ├── reports/                       # Technical reports (DATA_INVENTORY, DATA_QUALITY, PREPROCESSING, EDA)
│   └── review/                        # Academic audit reports (PLANNING_AUDIT, EDA_QUALITY_AUDIT, REPO_AUDIT)
│
├── notebooks/
│   └── 02_exploratory_data_analysis.ipynb # Executed Jupyter notebook with verified outputs
│
├── reports/
│   ├── figures/                       # 9 publication-grade figures (01 to 09, 300 DPI PNGs)
│   └── tables/                        # 3 statistical CSV summary tables
│
├── src/
│   ├── __init__.py
│   ├── data/                          # Data acquisition, profiling, and integrity verification
│   ├── preprocessing/                 # Continuous reindexing and bounded forward-fill cleaner
│   └── analysis/                      # Executable EDA figure/table generator and notebook builder
│
├── tests/
│   ├── test_data_validation.py        # 7 automated tests: schemas, hashes, non-negativity, CPCB bounds
│   ├── test_preprocessing.py          # 6 automated tests: continuous calendar, ffill bounds, target immutability
│   └── test_eda_artifacts.py          # 4 automated tests: figure headers, table structure, clean execution
│
├── .gitignore                         # Comprehensive Git exclusions for large datasets, caches, checkpoints
├── pytest.ini                         # Test runner configuration (pythonpath = ., no asyncio)
└── README.md                          # Comprehensive project documentation
```

---

## 11. Setup & Reproducibility Instructions (Windows PowerShell)

### Prerequisites
- Python 3.10.x installed and added to PATH
- Git for Windows

### 1. Clone the Repository
```powershell
git clone https://github.com/akshad1007/india-air-quality-analysis.git
cd india-air-quality-analysis
```

### 2. Set Up a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install pandas pyarrow scipy matplotlib seaborn pytest pytest-cov jupyter nbconvert kagglehub
```

### 4. Acquire the Dataset
Due to file sizes (>100MB), raw CSV files are not stored in GitHub. Download them using `kagglehub` into `data/raw/`:
```powershell
python -c "
import kagglehub, shutil, os
path = kagglehub.dataset_download('rohanrao/air-quality-data-in-india')
os.makedirs('data/raw', exist_ok=True)
for f in os.listdir(path):
    if f.endswith('.csv'):
        shutil.copy(os.path.join(path, f), 'data/raw')
print('Data acquired successfully.')
"
```

### 5. Run Preprocessing Pipeline
```powershell
python src/preprocessing/cleaner.py
```

### 6. Run the Automated Test Suite (17 Tests)
```powershell
pytest tests/
```

### 7. Regenerate All EDA Figures and Summary Tables
```powershell
python src/analysis/eda_generator.py
```

### 8. Execute the Exploratory Notebook Cleanly
```powershell
python -m nbconvert --to notebook --execute notebooks/02_exploratory_data_analysis.ipynb --output 02_exploratory_data_analysis.ipynb --output-dir notebooks
```

---

## 12. Current Project Status & Limitations

### Verified Milestone Accomplishments
- [x] **Phase 0:** Rubric inspection and 9-criteria reconciliation complete.
- [x] **Phase 1:** Planning documents (PRD, TRD, APP_FLOW, UI_UX, DATA_SCHEMA) audited and locked.
- [x] **Phase 2:** Raw datasets acquired and SHA-256 verified in `data/raw/`.
- [x] **Phase 3:** Data quality profiling, missingness heatmaps, and schema checks verified (7/7 tests passed).
- [x] **Phase 4:** Continuous calendar reindexing, bounded imputation, target preservation verified (6/6 tests passed).
- [x] **Phase 5:** Statistical EDA executed; 9 publication figures, 3 summary tables, and notebook verified (4/4 tests passed).
- [x] **Quality Audit:** `docs/review/EDA_QUALITY_AUDIT.md` passed with all hardcoding eliminated.
- [ ] **Phase 6 (ON HOLD):** Feature engineering pipeline and leak-free transformations awaiting faculty approval.

### Scientific & Data Limitations
1. **Absence of Numerical Meteorological Data:** The CPCB dataset records chemical pollutant concentrations only. Atmospheric phenomena (monsoon wash-out, thermal inversions) are literature-based hypotheses and cannot be used as predictive features without merging external meteorological datasets (e.g., ERA5 reanalysis).
2. **Truncated 2020 Record:** The dataset terminates on July 1, 2020, covering only the initial lockdown and early monsoon. It cannot represent a complete post-lockdown annual cycle.
3. **Monitoring Station Siting Heterogeneity:** Station locations reflect local administrative availability; city-level aggregations may obscure local micro-climates.
4. **Historical Imbalance:** Only 7 of 26 cities possess continuous records spanning the full 2015–2020 window.

---

## 13. Dataset Attribution & Licensing

- **Original Data Publisher:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India.
- **Dataset Curator:** Rohan Rao, hosted on [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india).
- **Licensing:** Released under the [Open Data Commons Open Database License (ODbL) / Public Domain Dedication](https://opendatacommons.org/licenses/odbl/).
- **Project License:** This academic PBL repository is licensed under the [MIT License](LICENSE).

---

## 14. Team Collaboration & Contributions

All four team members contributed collaboratively to the academic Problem-Based Learning project under the supervision of **Mr. Rohit Pawar**:
- **Akshad Makhana (2124UDSM1007):** Repository architecture, data pipeline engineering, and automated testing framework.
- **Shubham Pawar (2124UDSM1074):** Data acquisition, SHA-256 integrity verification, and missingness profiling.
- **Yash Kolhe (2124UDSM1107):** Exploratory data analysis, chemical precursor correlation studies, and visualization generation.
- **Pranav Hadole (2124UDSM1108):** Statistical hypothesis testing, COVID-19 lockdown analysis, and academic report compilation.

---

## 15. Planned Future Work (Post-Phase 6)

1. **Leak-Free Feature Engineering:** Implementation of temporal lag matrices ($T-1, T-2, T-7$) and backward-looking rolling statistics ($T-6 \dots T$) strictly within each city partition.
2. **Baseline Benchmarking:** Computation of naive persistence benchmarks on the 2019 out-of-time test set.
3. **Supervised Regression Modeling:** Training and hyperparameter tuning of RidgeCV, Random Forest, and HistGradientBoosting regressors.
4. **Exogenous Shock Stress Testing:** Evaluating model degradation and resilience on the 2020 COVID-19 lockdown regime.
5. **Interactive Streamlit Dashboard:** Building a multi-page analytical dashboard for public health exploration and offline forecast inspection.
