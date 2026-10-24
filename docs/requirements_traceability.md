# Requirements Traceability Matrix & Rubric Evidence Tracker
## India Air Quality Analysis and Next-Day AQI Forecasting

- **Document Version:** 1.4.0 (Academic Alignment & Phase 6 Readiness Review)
- **Academic Subject:** Predictive Modeling Using SPSS (IBM) (`24UADPC302`)
- **Academic Term:** Semester 5 (Third Year B.Tech)
- **Department & Institution:** TY AIDS Department, Sanjivani University, Kopargaon, Maharashtra
- **Faculty Guide:** Mr. Rohit Pawar
- **Project Team:**
  1. Akshad Makhana (`2124UDSM1007`)
  2. Shubham Pawar (`2124UDSM1074`)
  3. Yash Kolhe (`2124UDSM1107`)
  4. Pranav Hadole (`2124UDSM1108`)
- **Evaluation Standard:** AIDS Department PBL Rubric (100 Marks Total across 9 Assessment Criteria)
- **Current Internal Self-Assessment Status:** **40/100 Marks Verified (Categories 1–4 Complete)**
  *(Note: This represents the project's internal engineering milestone verification gate, certifying that all evidentiary artifacts, tests, and documentation are complete and audited; it is not an officially awarded faculty grade).*
- **Readiness Review Reference:** [docs/review/REPO_AND_ACADEMIC_AUDIT.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/review/REPO_AND_ACADEMIC_AUDIT.md)

---

### 1. Faculty Rubric Traceability Matrix

| Rubric No. | Faculty Criterion | Marks | Required Faculty Evidence | Project Implementation Activity | Guiding Planning Document | Target Implementation Module / Notebook | Measurable Quantitative Evidence / Artifact Gate | Verification Status |
| :---: | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| **1** | **Problem Definition & Business Understanding** | **10** | Clear problem statement, objectives, stakeholders, business relevance, and KPIs | Formulation of 6 empirical research questions, stakeholder mapping, KPI definitions, and offline forecasting boundaries | [PRD.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/planning/PRD.md) | `docs/planning/PRD.md`, `src/config.py` | Research Questions (RQ1–RQ6) defined; 4 distinct stakeholder personas; 4 quantitative success criteria; offline disclaimer verified | **VERIFIED (10/10 Marks Evidence Complete)** |
| **2** | **Dataset & Data Understanding** | **10** | Dataset relevance, feature understanding, data dictionary, data types, and data quality | Ingestion of 5 CPCB CSV files, SHA-256 hashing, schema profiling, missingness heatmaps, and empirical gap length analysis | [DATA_SCHEMA.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/planning/DATA_SCHEMA.md) | `src/data/inventory.py`, `src/data/deep_profile.py`, `tests/test_data_validation.py` | [DATA_INVENTORY.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/reports/DATA_INVENTORY.md) (SHA-256 hashes, exact row counts); [DATA_QUALITY_REPORT.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/reports/DATA_QUALITY_REPORT.md) (missingness matrix, 2019 test proof); `pytest` 7/7 passed in 2.61s | **VERIFIED (10/10 Marks Evidence Complete)** |
| **3** | **Data Preprocessing & Cleaning** | **10** | Missing values, duplicates, outliers, encoding, scaling, and appropriate transformations | Continuous calendar generation, boundary-safe bounded forward-fill, zero target imputation, city usability matrix | [TRD.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/planning/TRD.md) | `src/preprocessing/cleaner.py`, `tests/test_preprocessing.py` | [PREPROCESSING_REPORT.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/reports/PREPROCESSING_REPORT.md); `data/interim/city_eligibility_matrix.csv`; `pytest tests/test_preprocessing.py` (6/6 passed); raw SHA-256 verified identical | **VERIFIED (10/10 Marks Evidence Complete)** |
| **4** | **Exploratory Data Analysis (EDA)** | **10** | Statistical analysis, meaningful visualisations, patterns, correlations, and insights | Multi-city trend analysis, seasonal decomposition, chemical precursor correlation matrices, lockdown impact, diurnal curves | [APP_FLOW.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/planning/APP_FLOW.md) | `src/analysis/eda_generator.py`, `notebooks/02_exploratory_data_analysis.ipynb`, `reports/figures/` | [EDA_REPORT.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/reports/EDA_REPORT.md); [EDA_QUALITY_AUDIT.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/review/EDA_QUALITY_AUDIT.md); 9 publication figures in `reports/figures/`; 3 statistical CSV tables in `reports/tables/`; executed Jupyter notebook; `pytest tests/test_eda_artifacts.py` (4/4 passed) | **VERIFIED (10/10 Marks Evidence Complete)** |
| **5** | **Feature Engineering / Selection** | **10** | Creation or selection of relevant features, handling data leakage, and justification | Calendar features, autoregressive lags ($T-1, T-2, T-7$), backward-looking rolling statistics ($T-6 \dots T$) | [DATA_SCHEMA.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/planning/DATA_SCHEMA.md) | `src/features/feature_builder.py`, `tests/test_features.py` | Automated unit test verifying zero temporal leakage (`pytest tests/test_features.py`); feature correlation matrix; feature importance ranking | **PENDING PHASE 6** |
| **6** | **Model Selection & Implementation** | **15** | Appropriate AI/ML algorithm, implementation, train-test split, pipeline, and methodology | Chronological train/val/test splitting, Naive Persistence baseline, RidgeCV, Random Forest, HistGradientBoosting | [TRD.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/planning/TRD.md) | `src/models/trainer.py`, `notebooks/04_model_evaluation.ipynb` | Chronological split executed (Train 15–mid 18, Val mid–end 18, Test 2019, Shock 2020); scikit-learn pipeline artifacts (`models/*.pkl`); TimeSeriesSplit CV logs | **PENDING PHASE 8** |
| **7** | **Model Evaluation & Validation** | **15** | Appropriate evaluation metrics, validation strategy, confusion matrix or other relevant metrics, and interpretation | Out-of-time test evaluation on 2019 full year; MAE, RMSE, $R^2$; Normalized Confusion Matrix for CPCB Buckets; 2020 shock diagnostics | [TRD.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/planning/TRD.md) | `src/evaluation/evaluator.py`, `tests/test_evaluation.py` | Metric scorecard comparing ML models vs. Naive Baseline on 2019; Normalized Confusion Matrix (Precision/Recall/Macro F1); Residual diagnostic plots ($e_t$ vs. $\hat{y}_t$); 2020 shock degradation table | **PENDING PHASE 9** |
| **8** | **Business Insights & Recommendations** | **10** | Conversion of data analysis and model results into actionable recommendations | Translation of seasonal pollution peaks and model error bounds into grounded policy and public advisory guidelines | [PRD.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/planning/PRD.md) | `docs/reports/business_insights_and_recommendations.md` | Grounded recommendations report linking empirical findings to 4 stakeholders, action plans, uncertainty bounds, and required future data | **PENDING PHASE 10** |
| **9** | **Documentation & Presentation** | **10** | Report quality, PPT, code organization, GitHub/README, project demonstration, presentation, and Q&A | Technical report, executive summary, presentation slide deck, viva defense Q&A, and interactive Streamlit app | [IMPLEMENTATION_PLAN.md](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/planning/IMPLEMENTATION_PLAN.md) | `README.md`, `app/main.py`, `docs/reports/`, `docs/viva/` | Root README.md; Academic Technical Report; Slide Deck Outline; Viva Defense Q&A; Verified multi-page Streamlit dashboard | **IN PROGRESS (Planning, Inventory, Preproc & EDA Complete)** |
| **TOTAL** | **Nine Rubric Categories** | **100** | **Complete lifecycle coverage** | **Full end-to-end Data Science & ML workflow** | | | | **ACTIVE (40/100 Marks Fully Verified)** |

---

### 2. Milestone Verification Gate Checklist

- [x] **Gate 0 (Faculty Documents Inspection):** Both original faculty PDFs copied to and verified in [docs/reference/faculty-guidelines/](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/reference/faculty-guidelines):
  - [Assignment_4144_Content_Document_20261019020930913PM.pdf](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/reference/faculty-guidelines/Assignment_4144_Content_Document_20261019020930913PM.pdf) (PBL Instructions)
  - [Assignment_4144_Content_SolutionDocument_20261019020930913PM (1).pdf](file:///c:/Users/KANHA/Desktop/SPSS%20PROJECT/docs/reference/faculty-guidelines/Assignment_4144_Content_SolutionDocument_20261019020930913PM%20(1).pdf) (100-Mark Rubric)
  - Faculty expectations confirmed: original Kaggle dataset, data science lifecycle demonstration, statistical justification, team contribution documentation, and optional deployment status.
- [x] **Gate 1 (Six Planning Documents & Audit):** PRD, TRD, APP_FLOW, UI_UX_BRIEF, DATA_SCHEMA, IMPLEMENTATION_PLAN, and PLANNING_AUDIT authored, audited, and updated.
- [x] **Gate 2 (Raw Data Acquisition):** 5 CSV files placed in `data/raw/` and SHA-256 checksums verified.
- [x] **Gate 3 (Data Quality Audit):** Missingness and schema profiling executed; 7/7 automated tests passed via `pytest`.
- [x] **Gate 4 (Cleaning Pipeline):** Continuous calendar reindexing and boundary-safe imputation passing 6/6 automated tests; raw immutability certified.
- [x] **Gate 5 (Statistical EDA & Quality Audit):** 9 publication figures generated in `reports/figures/`; 3 tables in `reports/tables/`; executed notebook created; `docs/review/EDA_QUALITY_AUDIT.md` passed; 17/17 automated tests passing via `pytest`.
- [ ] **Gate 6 (Feature Engineering):** Zero temporal leakage confirmed via `pytest tests/test_features.py` (ON HOLD).
- [ ] **Gate 7 (Baseline Modeling):** Naive persistence baseline benchmark calculated.
- [ ] **Gate 8 (Machine Learning Pipelines):** Supervised regression models trained on historical folds.
- [ ] **Gate 9 (Held-Out Evaluation):** Out-of-time test evaluation on 2019 and 2020 shock diagnostics completed.
- [ ] **Gate 10 (Insights & Recommendations):** Grounded policy and stakeholder recommendations drafted.
- [ ] **Gate 11 (Interactive Dashboard):** Streamlit application operating locally with all pages verified (Optional per faculty guidelines).
- [ ] **Gate 12 (Academic Reporting):** Technical report, slide deck outline, and viva guide completed.
- [ ] **Gate 13 (Final Repository Audit):** Clean modular repository verified, test suite 100% green, README polished.
