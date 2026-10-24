# Repository Configuration, Academic Alignment & Phase 6 Readiness Audit
## India Air Quality Analysis and Next-Day AQI Forecasting

- **Document Identifier:** `docs/review/REPO_AND_ACADEMIC_AUDIT.md`
- **Audit Date:** 2026-10-24
- **Lead Data Scientist & Academic Auditor:** Senior Data Scientist & Project Auditor
- **Repository Remote:** `https://github.com/akshad1007/india-air-quality-analysis.git`
- **Audit Scope:** Complete repository health, Git configuration, academic alignment, data hygiene, test verification, and Phase 6 (Feature Engineering) readiness.

---

### 1. Academic Project Alignment & Identity Details

In accordance with institutional guidelines for the Problem-Based Learning (PBL) submission, the project metadata is formally registered as follows:

- **Project Title:** India Air Quality Analysis and Next-Day AQI Forecasting
- **Subject Name:** Predictive Modeling Using SPSS (IBM)
- **Subject Code:** 24UADPC302
- **Academic Term:** Semester 5 (Third Year B.Tech)
- **Department:** Department of Artificial Intelligence & Data Science (TY AIDS)
- **Institution:** Sanjivani University, Kopargaon, Maharashtra, India
- **Faculty Guide:** Mr. Rohit Pawar

#### Student Project Team

| No. | Student Name | University Roll / PRN Number |
| :---: | :--- | :--- |
| 1 | **Akshad Makhana** | `2124UDSM1007` |
| 2 | **Shubham Pawar** | `2124UDSM1074` |
| 3 | **Yash Kolhe** | `2124UDSM1107` |
| 4 | **Pranav Hadole** | `2124UDSM1108` |

#### Explicit Clarification on Academic Subject Identification
The subject title `24UADPC302 — Predictive Modeling Using SPSS (IBM)` serves strictly as the official university curriculum course code under which this PBL project is registered and evaluated. As clarified by the project leadership, this project follows the established Python data science and machine learning architecture for large-scale data engineering, continuous calendar reindexing, chunked stream analysis, automated unit testing, and model benchmarking. All technical decisions remain grounded in the existing project plan (`PRD.md`, `TRD.md`, `DATA_SCHEMA.md`).

---

### 2. Repository Git State & Hygiene Audit

#### 2.1 Git Repository Initialization & State Inspection
1. **Initial Local State Discovered:**
   - Inspection of `c:\Users\KANHA\Desktop\SPSS PROJECT` revealed that no local `.git` directory was previously initialized in the project root.
   - Instead, Git was traversing up to an accidental, empty `.git` repository in `C:\Users\KANHA\`.
   - A clean local Git repository was initialized directly within `c:\Users\KANHA\Desktop\SPSS PROJECT` on branch `main`.
2. **Remote Repository Inspection:**
   - Remote URL configured: `origin -> https://github.com/akshad1007/india-air-quality-analysis.git`.
   - Inspection of the remote via `git ls-remote` revealed an existing commit `9726b98` (`Initial commit` containing a single-line `# pune_hackathon`).
   - In accordance with safety rules, **no commits were pushed, no remote histories were overwritten, and no force-push operations were attempted**.

#### 2.2 Data Hygiene & Large File Exclusions
1. **File Size Hazard Identified:**
   - The five raw CSV files in `data/raw/` total approximately **296 MB**:
     * `station_hour.csv`: $219.67\text{ MB}$ (Exceeds GitHub's $100\text{ MB}$ hard file limit!)
     * `city_hour.csv`: $65.65\text{ MB}$
     * `station_day.csv`: $8.63\text{ MB}$
     * `city_day.csv`: $2.57\text{ MB}$
     * `stations.csv`: $14.47\text{ KB}$
   - Attempting to commit `station_hour.csv` would cause Git push rejections and repository bloat.
2. **Exclusion Guardrail Implemented:**
   - A robust `.gitignore` was authored to permanently exclude:
     * `data/raw/*.csv`
     * `data/interim/*.parquet`
     * `data/processed/*.parquet`
     * `.pytest_cache/`, `__pycache__/`, `*.log`, `.ipynb_checkpoints/`, virtual environments.
   - `.gitkeep` files were placed in `data/raw/`, `data/interim/`, and `data/processed/` to preserve directory hierarchy.
   - Exact, automated download instructions via `kagglehub` are documented in `README.md` to ensure any collaborator can re-acquire raw data without versioning 300 MB of CSVs in Git.

#### 2.3 Recommended GitHub Repository Metadata

- **Recommended Repository Description:**
  > "Academic PBL project for Predictive Modeling Using SPSS (IBM): analysis of Indian air-quality data from 2015–2020, exploratory data analysis, and planned next-day AQI forecasting."
- **Recommended Repository Topics:**
  `air-quality`, `india`, `data-science`, `exploratory-data-analysis`, `aqi-forecasting`, `predictive-modeling`, `academic-project`

---

### 3. Review of Completed Milestones (Phases 0–5) & Verification Audit

#### 3.1 Verified Accomplishments
1. **Raw Data Integrity Certified:**
   All five raw CPCB CSV files are verified in `data/raw/` with exact SHA-256 hashes matching `DATA_INVENTORY.md`.
2. **Continuous Calendar Preprocessing:**
   Raw observations were reindexed to a continuous $2,009$-day grid per city. Missing target AQI values ($4,681$ days) were strictly preserved without imputation. Predictor imputation was bounded to $\le 2$ days within each city.
3. **Artifact Completeness:**
   All 9 publication figures exist in `reports/figures/` (300 DPI PNGs) and 3 statistical tables in `reports/tables/`. All were regenerated via `src/analysis/eda_generator.py` in $14.2\text{ seconds}$.
4. **Notebook Execution:**
   `notebooks/02_exploratory_data_analysis.ipynb` executed head-to-tail from a clean kernel with zero errors, writing all execution outputs into the notebook artifact.
5. **Comprehensive Automated Test Suite:**
   17 unit and regression tests pass in $8.73\text{ seconds}$ via `pytest tests/`:
   - `test_data_validation.py` (7 tests: schemas, hashes, non-negativity, CPCB bounds)
   - `test_preprocessing.py` (6 tests: calendar continuity, ffill capping, target immutability)
   - `test_eda_artifacts.py` (4 tests: figure headers, table structure, clean execution, chunked sampling)

#### 3.2 Inconsistencies Audited & Resolved Prior to Phase 6
- **Hourly Sampling Bias:** Early draft sampled `nrows=100000`, capturing only alphabetical cities (Ahmedabad, Amaravati, Amritsar, Aizawl) while omitting Delhi, Kolkata, Mumbai, and Bengaluru. Resolved by implementing chunked streaming across all $707,875$ records, extracting $164,079$ valid records for all four target metros.
- **Station Descriptions Neutralized:** Early drafts used speculative labels ("Industrial Hub", "Forest-Buffered"). Audited against `stations.csv`, which contains zero land-use zoning fields. Replaced with neutral CPCB station names and geographic sectors.
- **Regional Comparisons Categorized:** Acknowledged that explanations involving maritime ventilation or thermal inversions are literature-based hypotheses rather than empirical findings, since the dataset lacks meteorological fields.
- **Seasonally Matched COVID-19 Baseline:** Replaced unstandardized multi-month estimates with exact March 25 – May 31 matched-window metrics ($N_b$ vs. $N_l$), reporting Welch's t-tests ($p < 10^{-6}$) and documenting baseline coverage disparities ($N_b=340$ for Delhi vs. $N_b=75$ for Kolkata).
- **Ahmedabad Extreme AQI Root Cause:** Verified that $412$ of $543$ days with $AQI > 500$ occurred in Ahmedabad due to uncapped linear extrapolation of the CPCB CO sub-index formula applied to extreme CO concentrations ($16 - 175\text{ mg/m}^3$).

#### 3.3 Status of the Claimed 40/100 Rubric Marks
The project tracking documents currently state a status of **40/100 marks verified** across Categories 1–4:
- Category 1: Problem Definition & Business Understanding (10 Marks) — Verified
- Category 2: Dataset & Data Understanding (10 Marks) — Verified
- Category 3: Data Preprocessing & Cleaning (10 Marks) — Verified
- Category 4: Exploratory Data Analysis (10 Marks) — Verified

> [!NOTE]
> **Audit Clarification:** This 40/100 status represents the project's **internal engineering self-assessment gate**, certifying that all required evidentiary artifacts, tests, and documentation for Categories 1–4 are complete and audited. It is not an officially awarded faculty grade, which will be determined by the faculty evaluator (Mr. Rohit Pawar) during formal project evaluation.

---

### 4. Phase 6 (Feature Engineering) Readiness Assessment

Before commencing Phase 6, the following eight readiness criteria were formally evaluated:

| No. | Readiness Requirement | Design Specification in Plan | Audit Verification | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Target Variable Definition** | Next-day composite AQI: $AQI_{t+1}$ | Shift $-1$ grouped strictly by city. Gaps preserve `NaN`. Never forward-filled. | **READY** |
| **2** | **Row & Date Alignment** | Features at row $t$ predict target at $t+1$ | Row $t$ contains observations up to calendar date $t$; target is date $t+1$. | **READY** |
| **3** | **Information Barrier / No Leakage** | Predictors use only information available at prediction time | Same-day contemporaneous AQI is never used to predict itself; future rolling windows strictly prohibited. | **READY** |
| **4** | **Autoregressive Lag Architecture** | Lags $T-1, T-2, T-7$ strictly within city | Grouped shift operations ensure no inter-city boundary contamination. | **READY** |
| **5** | **Rolling Statistical Windows** | Backward-looking rolling means/stds ($[t-2, t]$ and $[t-6, t]$) | `rolling(window=k, min_periods=...)` configured with `closed='right'` or backward indexing only. | **READY** |
| **6** | **Chronological Splitting Locked** | Train (2015–mid 2018), Val (mid–end 2018), Test (2019), Shock (2020) | Fully specified in `TRD.md` and `PRD.md`. Random splitting strictly banned. | **READY** |
| **7** | **Mandatory Baseline Defined** | Naive Persistence Model: $\widehat{AQI}_{t+1} = AQI_t$ | Established as primary hurdle. Proposed ML models must beat persistence MAE/RMSE on 2019 test set. | **READY** |
| **8** | **Missing Data Pipeline Strategy** | Dual variant handling (Variant A ffill vs. Variant B native trees) | `HistGradientBoostingRegressor` benchmarked against imputed RidgeCV/RandomForest pipelines. | **READY** |

---

### 5. Audit Verdict & Recommendation

- **Repository Configuration Status:** Complete. Clean local Git initialized, `.gitignore` active, large data excluded from Git, `README.md` updated to full academic specifications.
- **Preceding Phases Verification Status:** Complete. All 17 automated tests pass, all 9 figures and 3 tables verified, notebook executed cleanly.
- **Readiness Recommendation:** **CONDITIONAL APPROVAL TO PROCEED TO PHASE 6.**
- **Condition:** In accordance with project instructions, **Phase 6 remains ON HOLD** pending your review and explicit written approval. No feature engineering code, model training, or dashboard development will commence until approval is granted.
