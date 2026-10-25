# Phase 9.1 — SHAP Explainability Engine Report
## Academic Deliverable — Controlled, Evidence-Driven Model Interpretability

**Project:** India Air Quality Analysis and Next-Day AQI Forecasting  
**Course:** `24UADPC302 — Predictive Modeling Using SPSS (IBM)`  
**Institution:** Sanjivani University, Kopargaon, Maharashtra  
**Department:** Third Year B.Tech, Artificial Intelligence & Data Science (AIDS), Semester 5  
**Faculty Guide:** Mr. Rohit Pawar  
**Student Team:**  
- Akshad Makhana (`2124UDSM1007`)  
- Shubham Pawar (`2124UDSM1074`)  
- Yash Kolhe (`2124UDSM1107`)  
- Pranav Hadole (`2124UDSM1108`)  

**Repository:** `C:\Users\KANHA\Desktop\SPSS PROJECT\`  
**Target Milestone:** Phase 9.1 (SHAP Explainability Engine)  
**Status:** IMPLEMENTED, VERIFIED & AUDITED  
**Date:** October 2026  

---

## 1. Executive Summary & Objective

The objective of Phase 9.1 is to establish a rigorous, mathematically verified, and academically defensible explainability module for the frozen Random Forest champion model selected during Phase 7 and independently evaluated in Phase 8.

The module implements the **SHAP (SHapley Additive exPlanations)** framework ([Lundberg & Lee, 2017](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html); [Lundberg et al., Nature Machine Intelligence 2020](https://www.nature.com/articles/s42256-019-0138-9)) to decompose next-day continuous composite AQI forecasts into additive feature attributions.

### Key Governance Guarantees
1. **Model Immutability:** The frozen model artifact (`models/random_forest_val_model.joblib`) was loaded strictly read-only. No retraining, refitting, parameter alteration, or hyperparameter searching occurred.
2. **Preservation of Phase 8 Evaluation:** Phase 7 model selection, Phase 8 held-out metrics (2019 test MAE $26.2993$, RMSE $55.6234$), predictions, classification tables, and the academic submission package (`submission/phase8_academic_submission/`) remain completely untouched.
3. **Chronological Quarantines:** Background sampling and validation explanations were strictly confined to pre-2019 data ($\le 2018-12-31$). The 2019 primary test set and 2020 COVID-19 lockdown stress test set were strictly isolated and never accessed to construct or calibrate explanations.
4. **Non-Causal Epistemic Guardrail:** SHAP attributions quantify model prediction mechanics (how input features shift the model's conditional forecast relative to its baseline expected value). They **do not prove physical atmospheric causation**, pollutant chemical kinetics, or source-emission shares.

---

## 2. Frozen Model Architecture & Provenance

The explained model is the Phase 7 validation champion, preserved in `models/random_forest_val_model.joblib`.

### Provenance Specifications
- **Pipeline Structure:** `sklearn.pipeline.Pipeline` with two sequential stages:
  1. `preprocessor` (`ColumnTransformer`):
     - Categorical Transformer (`cat`): `OneHotEncoder(handle_unknown='ignore')` applied to `['City']` ($18$ Tier-1 qualified Indian cities).
     - Numerical Transformer (`num`): `SimpleImputer(strategy='median')` applied to the $18$ continuous feature columns.
     - Transformed Output Dimensionality: $36$ numeric columns ($18$ one-hot city indicators + $18$ numerical predictors).
  2. `regressor` (`RandomForestRegressor`):
     - Number of Trees: $100$
     - Maximum Depth: $10$
     - Minimum Samples Split: $5$
     - Minimum Samples Leaf: $2$
     - Random State: $42$
     - Number of Parallel Jobs: $-1$
     - Input Feature Dimension (`n_features_in_`): $36$
- **Prediction Target:** Continuous next-day composite Air Quality Index $y_{c, t+1} \in [0, \infty)$ (strictly continuous, evaluated in continuous AQI points).
- **Prediction Interface:** Accepts raw DataFrame containing $19$ predictor columns (`City` + $18$ chemical, lag, and calendar features). Target columns (`Target_AQI`, `Target_Date`) are strictly excluded.

---

## 3. Dataset Periods & Background Reference Methodology

In accordance with strict chronological partitioning rules, data partitions are segmented by `Target_Date`:

| Split Name | Calendar Date Range | Sample Count (Shared Cohort) | Role in Phase 9.1 |
| :--- | :--- | :--- | :--- |
| **Historical Training** | 2015-01-02 to 2018-06-30 | $10,042$ | Source for reproducible background reference distribution ($N=100$) |
| **Designated Validation** | 2018-07-01 to 2018-12-31 | $2,982$ | Target cohort for global feature importance and local case explanations |
| **Primary Test (Held-Out)** | 2019-01-01 to 2019-12-31 | $6,201$ | **Strictly Quarantined** (Untouched) |
| **Exogenous Stress Test** | 2020-01-01 to 2020-07-01 | $3,124$ | **Strictly Quarantined** (Untouched) |

### Background Sample Selection
- **Sampling Protocol:** Simple random sample of $N=100$ observations drawn strictly from the historical training split (`split == 'train'`, `is_target_valid == True`, `is_shared_cohort == True`).
- **Seed:** Fixed `random_state = 42`.
- **Temporal Span of Background Sample:** Earliest `2015-02-02`, Latest `2018-06-23`.
- **Scientific Rationale:** Drawing the background sample strictly from observations on or before `2018-06-30` guarantees that the reference baseline distribution reflects historical atmospheric training priors without temporal leakage across the validation barrier.

---

## 4. Feature Mapping & Technical TreeSHAP Implementation

### Transformed Feature Representation
The pipeline transforms the $19$ raw input features into $36$ authoritative features:
- $18$ One-Hot Categorical Features: `cat__City_Ahmedabad`, `cat__City_Amaravati`, `cat__City_Amritsar`, `cat__City_Bengaluru`, `cat__City_Brajrajnagar`, `cat__City_Chennai`, `cat__City_Delhi`, `cat__City_Gurugram`, `cat__City_Hyderabad`, `cat__City_Jaipur`, `cat__City_Jorapokhar`, `cat__City_Kolkata`, `cat__City_Lucknow`, `cat__City_Mumbai`, `cat__City_Patna`, `cat__City_Talcher`, `cat__City_Thiruvananthapuram`, `cat__City_Visakhapatnam`.
- $18$ Numerical Features:
  - Autoregressive Lags: `AQI_t` (lag 1 to target), `AQI_t_minus_1` (lag 2), `AQI_t_minus_6` (lag 7 weekly anchor).
  - Chemical Precursors: `PM25_t`, `PM10_t`, `NO2_t`, `SO2_t`, `CO_t`, `O3_t`, `PM_ratio_t` ($\text{PM}_{2.5} / \text{PM}_{10}$).
  - Backward Rolling Moments: `AQI_roll_mean_3d`, `AQI_roll_mean_7d`, `AQI_roll_std_7d`.
  - Deterministic Calendar Features: `target_sin_doy`, `target_cos_doy`, `target_dow`, `target_is_weekend`, `target_month`.

### TreeSHAP Algorithmic Mechanics
We utilized `shap.TreeExplainer` operating on the fitted `regressor` (`RandomForestRegressor`).
- **Algorithmic Mode:** Path-dependent TreeSHAP (`feature_perturbation="tree_path_dependent"`).
- **Baseline Expected Value ($\phi_0$):** $193.2521$ AQI points. This corresponds to the exact weighted leaf output expectation across the 100 decision trees in the ensemble.
- **Empirical Additivity Finding:** Path-dependent TreeSHAP calculates exact conditional expectations based on internal tree node weights. Across the entire $2,982$-sample validation cohort, the maximum absolute reconstruction error was:
  $$\max |f(x) - (\phi_0 + \sum_{j=1}^{36} \phi_j)| = 5.91 \times 10^{-12} \text{ AQI points}$$
  with a mean error of $2.90 \times 10^{-13}$ AQI points.
- **Interventional Comparison & Diagnostic Transparency:** When `TreeExplainer` was initialized with interventional perturbation on the finite $N=100$ background sample, Monte Carlo split approximations introduced an expected sampling variance of up to $\approx 1.5$ AQI points ($0.3\%$ of the 500 AQI scale). To guarantee exact mathematical additivity and avoid heuristic suppression, path-dependent TreeSHAP was established as the primary explainability engine.

---

## 5. Global Feature Importance Findings

Across the complete validation cohort ($N=2,982$), global feature importance was computed as the mean absolute SHAP value:
$$I_j = \frac{1}{N} \sum_{k=1}^N |\phi_{k, j}|$$

All 36 features are archived in `reports/shap_feature_importance.csv`. The top 15 features are summarized below:

| Rank | Feature Name | Mean \|SHAP\| (AQI Points) | Feature Class | Mechanistic Role in RF Predictor |
| :---: | :--- | :---: | :--- | :--- |
| **1** | `AQI_t` | **73.3110** | Autoregressive Lag | Primary persistent baseline for next-day air quality |
| **2** | `PM25_t` | **21.1213** | Chemical Precursor | Fine particulate concentration driving acute CPCB spikes |
| **3** | `CO_t` | **18.4227** | Chemical Precursor | Primary tracer for combustion and the Ahmedabad sensor anomaly |
| **4** | `AQI_roll_mean_7d` | **8.2258** | Rolling Moment | Medium-term weekly smoothed background trend |
| **5** | `PM10_t` | **3.3341** | Chemical Precursor | Coarse inhalable particulate matter |
| **6** | `NO2_t` | **2.2019** | Chemical Precursor | Vehicular and industrial nitrogen dioxide precursor |
| **7** | `City_Ahmedabad` | **1.4982** | Geographic Indicator | Captures localized baseline shift for Ahmedabad cohort |
| **8** | `AQI_t_minus_6` | **1.4702** | Autoregressive Lag | Weekly periodic anchor |
| **9** | `O3_t` | **1.2927** | Chemical Precursor | Ground-level photochemical oxidant |
| **10** | `AQI_roll_mean_3d` | **1.2494** | Rolling Moment | Short-term 3-day momentum |
| **11** | `target_cos_doy` | **1.1922** | Cyclical Calendar | Winter peak vs. summer monsoon annual seasonality |
| **12** | `AQI_t_minus_1` | **1.0246** | Autoregressive Lag | 2-day-lagged autoregressive memory |
| **13** | `SO2_t` | **0.8662** | Chemical Precursor | Industrial sulfur dioxide precursor |
| **14** | `target_sin_doy` | **0.7571** | Cyclical Calendar | Spring vs. autumn transition cycle |
| **15** | `City_Delhi` | **0.6606** | Geographic Indicator | Indo-Gangetic Plain regional elevation offset |

### Global Visualizations
- **SHAP Summary Beeswarm Plot:** Saved to `reports/figures/shap_summary.png`. Illustrates the directional distribution: high observed values of `AQI_t`, `PM25_t`, and `CO_t` (red dots) exert strong positive attributions (pushing predicted AQI upwards), whereas low values (blue dots) push predicted AQI downwards.
- **Mean Absolute Importance Bar Chart:** Saved to `reports/figures/shap_importance.png`. Highlights the steep dominance of `AQI_t` ($73.31$ points) followed by `PM25_t` ($21.12$ points) and `CO_t` ($18.42$ points).

---

## 6. Local Case Explanations

Two deterministic local case studies were selected from the pre-2019 validation cohort ($N=2,982$) using documented, reproducible selection rules.

```
Residual Convention: e_i = y_true - y_pred
Positive residual (+) = Underprediction (model predicted lower than observed)
Negative residual (-) = Overprediction (model predicted higher than observed)
```

### A. Typical Case (Moderate Hazard Tier)
- **Selection Rule:** Pre-2019 validation observation ($N=2,982$) with observed AQI in the CPCB Moderate category ($100 < y \le 200$) minimizing absolute distance to the validation median observed AQI ($112.0$ AQI points). Ties broken by earliest `Target_Date`, then alphabetical `City`.
- **Selected Observation:**
  - City: **Gurugram**
  - Observation Date ($t$): `2018-07-30`
  - Target Date ($t+1$): `2018-07-31`
  - Observed AQI ($y$): **112.00**
  - Model Forecast ($\hat{y}$): **115.4546**
  - Residual ($e = y - \hat{y}$): **-3.4546 AQI points** (minor overprediction by $3.45$ points)
- **Additive Reconstruction:**
  - Baseline Expected Value ($\phi_0$): **193.2521**
  - Sum of Attributions ($\sum \phi_i$): **-77.7975**
  - Reconstructed Prediction: $193.2521 - 77.7975 = \mathbf{115.4546}$
  - Absolute Reconstruction Discrepancy: $1.28 \times 10^{-13}$ AQI points.
- **Attribution Dynamics:**
  - The model baseline is $193.25$ points. Because Gurugram's observed air quality on July 30 was relatively clean (`AQI_t = 109.0`, `PM25_t = 63.41 \mu\text{g/m}^3`, `CO_t = 0.59 \text{mg/m}^3`), the features pull the forecast down towards moderate levels:
    - `AQI_t` attribution: **-45.9721**
    - `AQI_roll_mean_7d` ($91.0$): **-11.1950**
    - `CO_t` ($0.59$): **-6.2080**
    - `PM25_t` ($63.41$): **-5.8552**
    - `AQI_roll_mean_3d` ($74.0$): **-1.5211**
  - Visualization: Archived in `reports/figures/shap_waterfall_typical.png`.

---

### B. Extreme Case (Severe Hazard Tier / CO Sensor Anomaly)
- **Selection Rule:** Pre-2019 validation observation ($N=2,982$) with the maximum observed AQI across the validation split. Ties broken by earliest `Target_Date`, then alphabetical `City`.
- **Selected Observation:**
  - City: **Ahmedabad**
  - Observation Date ($t$): `2018-11-17`
  - Target Date ($t+1$): `2018-11-18`
  - Observed AQI ($y$): **1672.00**
  - Model Forecast ($\hat{y}$): **1062.6818**
  - Residual ($e = y - \hat{y}$): **+609.3182 AQI points** (underprediction by $609.32$ points)
- **Additive Reconstruction:**
  - Baseline Expected Value ($\phi_0$): **193.2521**
  - Sum of Attributions ($\sum \phi_i$): **+869.4297**
  - Reconstructed Prediction: $193.2521 + 869.4297 = \mathbf{1062.6818}$
  - Absolute Reconstruction Discrepancy: $2.73 \times 10^{-12}$ AQI points.
- **Attribution Dynamics:**
  - Extreme precursor concentrations push the forecast dramatically upwards by $+869.43$ AQI points above baseline:
    - `AQI_t` ($1354.0$): **+388.1507**
    - `CO_t` ($68.52 \text{ mg/m}^3$): **+377.9746**
    - `NO2_t` ($84.54 \mu\text{g/m}^3$): **+30.3998**
    - `PM25_t` ($93.73 \mu\text{g/m}^3$): **+18.1012**
    - `O3_t` ($50.29 \mu\text{g/m}^3$): **+17.6542**
- **Academic Insight into Tree Mechanics:**
  - This extreme case clearly illustrates why Random Forest underpredicts on extreme spikes ($e = +609.32$): decision trees are piecewise-constant estimators bounded by training leaf means. Even with `max_depth=10`, leaf predictions cannot extrapolate beyond $\approx 1062$ AQI points.
  - Crucially, SHAP confirms that `CO_t` contributes $+377.97$ AQI points to this prediction, directly corroborating the Phase 8 sensitivity finding regarding the Ahmedabad CO sensor anomaly.
  - Visualization: Archived in `reports/figures/shap_waterfall_extreme.png`.

---

## 7. Mathematical Additivity Verification Results

For all observations $x$, the SHAP local accuracy property mandates:
$$f(x) = \phi_0 + \sum_{i=1}^{p} \phi_i$$

### Quantitative Audit Results

| Diagnostic Metric | Measured Value | Declared Acceptance Criterion | Evaluation Result |
| :--- | :---: | :---: | :---: |
| **Evaluated Cohort** | Validation Split ($N=2,982$) | Pre-2019 Designated Validation Split | PASS |
| **Baseline Expected Value ($\phi_0$)** | $193.25205974$ AQI points | Finite real scalar | PASS |
| **Max Absolute Reconstruction Error** | **$5.9117 \times 10^{-12}$ AQI points** | $\le 0.50$ AQI points | **PASS (Machine Precision)** |
| **Mean Absolute Reconstruction Error** | **$2.8994 \times 10^{-13}$ AQI points** | $\le 0.05$ AQI points | **PASS (Machine Precision)** |
| **Prediction Concordance** | Identical to `pipeline.predict(X)` | Max absolute diff $< 10^{-6}$ | PASS |

---

## 8. Limitations, Computational Performance & Reproducibility

### Limitations
1. **Multicollinearity & Credit Splitting:** In urban atmospheres, $\text{PM}_{2.5}$, $\text{PM}_{10}$, and lagged AQI are highly collinear ($r > 0.85$). In tree-based splitting, correlated features split credit across tree ensembles depending on tree partition random seeds. A high attribution for `AQI_t` reflects its utility as an autoregressive summary index, but does not mean fine particles are unimportant.
2. **Piecewise Saturation:** Decision tree ensembles cannot extrapolate beyond the upper envelope of training leaf averages, as demonstrated by the Ahmedabad extreme case.
3. **No Retraining Guarantee:** If the underlying training dataset is modified, SHAP explanations would change; however, within this frozen project, artifacts are fixed.

### Computational Performance
- **Environment:** Windows 11, Python 3.10.11, `shap==0.48.0`, `scikit-learn==1.7.2`, `numpy==2.2.6`, `matplotlib==3.10.3`.
- **TreeExplainer Initialization:** $0.02$ seconds.
- **Validation Set Evaluation ($N=2,982$ rows):** $31.3$ seconds on multithreaded CPU.
- **Unit Test Execution (`pytest tests/test_interpretability.py`):** $7.52$ seconds.

### Reproducibility Command
To reproduce the complete explainability engine, plots, and CSV deliverables:
```bash
python -m src.models.interpretability
```

---

## 9. Non-Causal Epistemic Guardrail Statement

> [!IMPORTANT]
> **Statement on Non-Causal Interpretation:**  
> SHAP feature attributions describe how the mathematical regression function $f(x)$ adjusts its predicted composite AQI value based on the values of the input features relative to the model's baseline training expectation.  
> **SHAP values do NOT represent physical atmospheric causation.**  
> A positive attribution of $+377.97$ AQI points for `CO_t` in Ahmedabad indicates that the model heavily used the elevated CO signal to scale up its forecast; it **does not prove** that carbon monoxide physically caused the severe air pollution event or that reducing ambient CO by $1 \text{ mg/m}^3$ would reduce ambient AQI by a corresponding amount. Atmospheric chemical transformation, boundary layer meteorology, and source emission apportionment require specialized deterministic chemical transport models (such as WRF-Chem or CMAQ), which are beyond the scope of statistical machine learning.

---

## 10. Integrity Audit: Preservation of Phase 8 & Submissions

To verify that Phase 9.1 maintained strict immutability over prior work, an audit was conducted:
1. **Model Artifact Immutability:** `models/random_forest_val_model.joblib` was verified using SHA-256 and parameter inspection. Zero retraining or rewriting occurred.
2. **Phase 8 Deliverables Unaltered:**
   - `reports/phase8_test_metrics.csv`: Unmodified.
   - `reports/phase8_stress_metrics.csv`: Unmodified.
   - `reports/phase8_test_predictions.csv`: Unmodified.
   - `reports/phase8_stress_predictions.csv`: Unmodified.
   - `reports/phase8_ahmedabad_sensitivity.csv`: Unmodified.
   - `docs/reports/PHASE_8_EVALUATION_REPORT.md`: Unmodified.
   - `docs/review/PHASE_8_ACCEPTANCE_AUDIT.md`: Unmodified.
3. **Academic Submission Package Unaltered:**
   - All 37 files in `submission/phase8_academic_submission/` were checked and confirmed intact. Zero files modified or added to the frozen package.
4. **Git Tree Cleanliness:** No `git add`, `git commit`, `git push`, `git checkout`, or `git reset` commands were executed.

---

## 11. Deliverables & Test Verification Matrix

### File Deliverables

| Deliverable Path | Description | Verification Status |
| :--- | :--- | :---: |
| `src/models/interpretability.py` | Complete SHAP explainability engine implementation | VERIFIED & EXECUTED |
| `tests/test_interpretability.py` | 10 focused unit and integration tests | VERIFIED (10/10 PASSED) |
| `reports/shap_feature_importance.csv` | Machine-readable table of 36 features, ranks, and mean \|SHAP\| | GENERATED ($2,007$ bytes) |
| `reports/figures/shap_summary.png` | Global SHAP beeswarm summary plot ($N=2,982$) | GENERATED ($377,886$ bytes) |
| `reports/figures/shap_importance.png` | Global Top 20 feature importance horizontal bar chart | GENERATED ($218,572$ bytes) |
| `reports/figures/shap_waterfall_typical.png` | Local waterfall attribution plot for Gurugram Typical case | GENERATED ($263,134$ bytes) |
| `reports/figures/shap_waterfall_extreme.png` | Local waterfall attribution plot for Ahmedabad Extreme case | GENERATED ($297,176$ bytes) |
| `docs/reports/PHASE_9_INTERPRETABILITY_REPORT.md` | Comprehensive academic interpretability deliverable | GENERATED & AUDITED |

### Test Suite Execution Summary
- **Focused Test Suite:** `pytest tests/test_interpretability.py -v`  
  - Result: **10 passed in 7.52s** (Exit Code: `0`).
- **Full Repository Test Suite:** `pytest tests/ -v`  
  - Result: **61 passed in 16.31s** (Exit Code: `0`).
  - Breakdown: 51 legacy tests (Data validation, Preprocessing, Features, Models, Evaluation) + 10 new Interpretability tests.
- **Git Formatting Check:** `git diff --check`  
  - Result: **Exit Code 0** (No whitespace errors or line-ending defects).

---
*End of Phase 9.1 SHAP Explainability Engine Report. Ready for Faculty Guide review.*
