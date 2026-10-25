# India Air Quality Analysis and Next-Day AQI Forecasting
## Phase 8 Academic Submission Package (Phase 7 & Phase 8)

- **Course Code & Title:** `24UADPC302: Predictive Modeling Using SPSS (IBM)`
- **Academic Program:** Semester 5, TY AIDS Department
- **Institution:** Sanjivani University, Kopargaon, Maharashtra
- **Faculty Guide:** Mr. Rohit Pawar
- **Project Team:**
  1. Akshad Makhana (`2124UDSM1007`)
  2. Shubham Pawar (`2124UDSM1074`)
  3. Yash Kolhe (`2124UDSM1107`)
  4. Pranav Hadole (`2124UDSM1108`)
- **Repository Remote:** `https://github.com/akshad1007/india-air-quality-analysis.git`
- **Base Commit:** `4c9844309d58b91891e160fbadb872bc490552a4`
- **Milestone State:** **PHASE 8 ACCEPTED WITH LIMITATIONS & FROZEN**
- **Phase 9 Status:** **UNSTARTED (STRICT STOP BOUNDARY)**

---

## 1. Project Overview & Primary Objective

This academic submission package delivers the complete modeling implementation (Phase 7) and held-out empirical evaluation (Phase 8) for the **India Air Quality Analysis and Next-Day AQI Forecasting** project.

The objective is to forecast next-day Air Quality Index ($\widehat{\text{AQI}}_{c, t+1}$) across 18 major municipal jurisdictions in India using multi-pollutant surface station telemetry, rolling statistics, calendar indicators, and regional fixed effects.

### Critical Academic Metadata Note
The course title `24UADPC302: Predictive Modeling Using SPSS (IBM)` represents institutional course registration metadata. The technical execution is implemented purely in Python using `scikit-learn`, `pandas`, and `numpy` in compliance with modern data science practices.

---

## 2. Directory Structure Guide

This self-contained academic submission package is organized as follows:

```text
phase8_academic_submission/
├── README.md                                  # Top-level academic submission guide
├── 01_Project_Overview/
│   └── project_overview.md                    # Academic paper-style project executive summary
├── 02_Modeling_and_Evaluation/
│   ├── PHASE_7_MODELING_REPORT.md             # Phase 7 chronological validation & model selection report
│   └── PHASE_8_EVALUATION_REPORT.md           # Phase 8 final held-out test & stress evaluation report
├── 03_Acceptance_Audits/
│   ├── PHASE_6_READINESS_REVIEW.md            # Feature engineering leakage & readiness audit
│   ├── PHASE_7_ACCEPTANCE_AUDIT.md            # Independent audit of Phase 7 model selection
│   └── PHASE_8_ACCEPTANCE_AUDIT.md            # Independent audit of Phase 8 evaluation metrics
├── 04_Evaluation_Results/
│   ├── model_selection_summary.csv            # Phase 7 validation period metric comparison
│   ├── phase8_test_metrics.csv                # Primary 2019 test set regression metrics
│   ├── phase8_stress_metrics.csv              # 2020 COVID-19 lockdown stress test metrics
│   ├── phase8_classification_metrics.csv      # CPCB 6-tier hazard category diagnostic metrics
│   ├── phase8_ahmedabad_sensitivity.csv       # 18-city vs 17-city (Ahmedabad-excluded) sensitivity
│   ├── phase8_residual_by_city.csv            # Municipal residual diagnostics across all 18 cities
│   ├── phase8_residual_by_season.csv          # Meteorological seasonal error distributions
│   └── phase8_residual_by_bucket.csv          # Error distribution across CPCB hazard severity tiers
├── 05_Prediction_Evidence/
│   ├── phase8_test_predictions.csv            # 6,286 row-by-row 2019 predictions & ground truth
│   ├── phase8_stress_predictions.csv          # 3,163 row-by-row 2020 stress predictions & ground truth
│   └── validation_predictions.csv             # Validation period predictions supporting Phase 7
├── 06_Source_Code/
│   ├── features/                              # Feature engineering modules (feature_builder.py)
│   ├── models/                                # Model pipelines, baseline, training, and evaluation
│   └── tests/                                 # Complete pytest verification test suite (51 tests)
├── 07_Model_Artifacts/
│   ├── random_forest_val_model.joblib         # Champion Random Forest pipeline (5.16 MB)
│   ├── ridge_regression_val_model.joblib      # Baseline Ridge pipeline (4.98 KB)
│   └── histgradientboosting_val_model.joblib  # Candidate HistGradientBoosting pipeline (406 KB)
└── 08_Verification/
    └── verification_summary.md                # Independent verification & SHA-256 integrity log
```

---

## 3. Summary of Completed Phase 7 and Phase 8 Work

### Phase 7: Model Selection & Chronological Validation
- **Protocol:** Evaluated expanding-window temporal cross-validation across historical data ($\le 2018-06-30$) with final confirmation on the designated validation period (`2018-07-01` to `2018-12-31`, Shared $N=2,982$).
- **Selection Criterion:** **Validation MAE**.
- **Outcome:** **Random Forest Regressor** (`n_estimators=100, max_depth=10, min_samples_split=5`) achieved the lowest validation error: **MAE $27.3559\text{ AQI points}$** (+7.70% improvement over Persistence), **RMSE $59.7368\text{ AQI points}$** (+14.78% error reduction), and **$R^2 = 0.8774$**.
- **Isolation:** The 2019 test set and 2020 stress set remained 100% untouched during Phase 7.

### Phase 8: Final Model Evaluation & Held-Out Validation
- **Frozen Configurations:** Models were loaded directly from Phase 7 serialized binaries without retraining or hyperparameter retuning.
- **2019 Primary Held-Out Test Set (Shared Cohort A, $N=6,201$):**
  - **Naive Persistence Baseline:** MAE = $31.0419$, RMSE = $70.2360$, $R^2 = 0.7445$, MBE = $+0.3614$
  - **Random Forest (Champion):** MAE = **$26.2993\text{ AQI points}$** (**$+15.28\%$ error reduction**), RMSE = **$55.6234\text{ AQI points}$** (**$+20.81\%$ error reduction**), $R^2 = \mathbf{0.8398}$, MBE = **$+4.8666$**
  - **HistGradientBoosting:** MAE = $27.1402$ ($+12.57\%$), RMSE = **$54.9397\text{ AQI points}$** (**$+21.78\%$ error reduction**), $R^2 = \mathbf{0.8437}$, MBE = $+5.6109$
  - **Ridge Regression:** MAE = $29.0261$ ($+6.49\%$), RMSE = $55.8918$ ($+20.42\%$), $R^2 = 0.8382$, MBE = $+2.8414$
- **Model Selection Trade-Off Disclosed:** Random Forest was designated champion based on the predeclared validation MAE criterion. On the 2019 test set, RF confirmed superior MAE ($26.30$ vs $27.14$) and higher classification Macro F1 ($0.6678$ vs $0.6057$), while HistGradientBoosting achieved a slightly lower test RMSE ($54.94$ vs $55.62$). No retroactive winner re-selection was made.
- **2020 Lockdown Stress Period (Shared Cohort A, $N=3,124$):**
  - Random Forest demonstrated robust generalization: MAE = **$19.3288\text{ AQI points}$** (**$+10.64\%$ error reduction** vs Persistence $21.6306$), RMSE = **$39.1893\text{ AQI points}$** (**$+14.69\%$ error reduction** vs Persistence $45.9400$), $R^2 = \mathbf{0.8060}$.
- **Ahmedabad Anomaly Secondary Diagnostic:**
  - In 2019 Cohort A, Ahmedabad accounts for $347$ rows ($5.6\%$), but contributes **$72.59\%$ of total network squared error** ($13,926,980.55 / 19,185,641.08$) due to extreme sensor saturation ($\text{CO} > 400\text{ mg/m}^3$). Excluding Ahmedabad drops Random Forest RMSE to **$29.97\text{ AQI points}$** with **$R^2 = 0.8931$**.

---

## 4. How to Run Automated Verification Tests

### Environment Prerequisites
- Python 3.10+
- Dependencies: `pytest>=7.4.0`, `scikit-learn>=1.3.0`, `pandas>=2.0.0`, `numpy>=1.24.0`, `pyarrow>=12.0.0`, `joblib>=1.3.0`

### Execution Command
From the root of the project repository (`C:\Users\KANHA\Desktop\SPSS PROJECT\`):
```powershell
pytest tests/ -v
```

### Verified Test Outcome
- **Exit Code:** `0`
- **Total Tests Collected:** `51`
- **Passed:** `51 (100%)`
- **Failed / Skipped:** `0 / 0`
- **Execution Runtime:** ~15.5 to 16.2 seconds

---

## 5. How to Reproduce Inference from Serialized Artifacts

Inference can be reproduced directly using the serialized pipelines in `07_Model_Artifacts/` and the processed feature dataset `data/processed/city_day_features.parquet`:

```python
import joblib
import pandas as pd
from src.features.feature_builder import get_feature_columns

# Load model pipeline
rf_model = joblib.load("07_Model_Artifacts/random_forest_val_model.joblib")

# Load processed feature dataset
df = pd.read_parquet("data/processed/city_day_features.parquet")
test_df = df[(df["split"] == "test") & df["is_target_valid"]].copy()

# Feature column alignment
num_cols, cat_cols, _ = get_feature_columns()
f_cols = cat_cols + num_cols

# Generate predictions
predictions = rf_model.predict(test_df[f_cols])
# Confirmed bit-for-bit identical to phase8_test_predictions.csv (max diff = 0.000000)
```

---

## 6. Currency, Units, and Academic Limitations

1. **Units Convention:**
   - All forecasting errors (MAE, RMSE, MBE) are measured strictly in **AQI points**.
   - $R^2$, Accuracy, Macro F1, and Balanced Accuracy are dimensionless statistical quantities.
   - **Indian Rupee (`₹`, `INR`):** No financial expenses, cloud hosting invoices, compute budgets, or operational salaries were tracked or generated for this study. In compliance with academic integrity, **no artificial monetary values were invented**.
2. **Mean Bias Error (MBE) Sign Convention:**
   $$\text{MBE} = \frac{1}{N}\sum_{i=1}^N (\widehat{y}_i - y_i)$$
   Positive MBE indicates average overprediction ($\widehat{y} > y$) in AQI points; negative MBE indicates average underprediction ($\widehat{y} < y$).
3. **CPCB Category Mapping Convention:**
   The statutory CPCB scale terminates at 500. Continuous model predictions $>400$ are mapped post-hoc to `Severe` (and $\le 50$ to `Good`). This is an operational categorization convention for continuous forecasts, not an extension of the official CPCB index.
4. **Non-Causal Interpretation:**
   Observed performance shifts between 2019 and 2020 coincide with the COVID-19 nationwide lockdown and seasonal meteorology, but cannot be causally proven from observational telemetry alone without counterfactual atmospheric modeling.

---

## 7. Submission Sign-Off & Freeze Boundary

This package is complete, self-contained, and frozen at Phase 8.  
**Phase 9 (Explainability / Deployment) has NOT been started.**
