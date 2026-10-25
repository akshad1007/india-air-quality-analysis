# Academic Project Overview: India Air Quality Analysis & Next-Day AQI Forecasting

## 1. Problem Statement
Air pollution is an acute environmental and public health crisis across urban India. Ground-level telemetry frequently records extreme concentrations of fine particulate matter ($\text{PM}_{2.5}$, $\text{PM}_{10}$), nitrogen oxides ($\text{NO}_x$), and carbon monoxide ($\text{CO}$), exceeding World Health Organization (WHO) and Central Pollution Control Board (CPCB) air quality standards. Traditional next-day forecasting systems often rely on naive inertia (assuming tomorrow's air quality equals today's) or complex numerical chemical transport models that require heavy supercomputing infrastructure and meteorological assimilation, often unavailable at municipal scale.

## 2. Project Objective
To develop, validate, and evaluate an empirical, leak-free machine learning system capable of generating reliable next-calendar-day Air Quality Index forecasts ($\widehat{\text{AQI}}_{c, t+1}$) across 18 major Indian cities using multi-pollutant telemetry, historical moving statistics, seasonal calendar indicators, and municipal fixed effects.

## 3. Dataset & Temporal Scope
The investigation utilizes official daily air quality telemetry provided by the Central Pollution Control Board (CPCB) spanning from **January 1, 2015 through July 1, 2020** ($N=29,531$ city-day observations across 18 cities).

The temporal architecture strictly enforces four chronological partitions:
- **Historical Training Partition:** Target dates from `2015-01-02` through `2018-06-30` (~3.5 years).
- **Designated Validation Period:** Target dates from `2018-07-01` through `2018-12-31` (6 months; used exclusively for Phase 7 model confirmation).
- **Primary Held-Out Test Period:** Target dates from `2019-01-01` through `2019-12-31` (1 full year; untouched until Phase 8).
- **Exogenous Stress Period:** Target dates from `2020-01-01` through `2020-07-01` (covers COVID-19 national lockdown onset).

## 4. Feature Engineering Approach (Phase 6)
To ensure zero lookahead leakage:
- **Strict Target Alignment:** All features available at prediction origin $t$ strictly predict next calendar day $t+1$ ($\Delta D \equiv 1\text{ day}$). Unobserved targets are never imputed.
- **Predictor Set (52 Features):**
  - **Lags:** 1-day, 2-day, 3-day, and 7-day backward lags for AQI and primary pollutants ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$).
  - **Rolling Window Statistics:** Backward-looking 7-day and 14-day rolling means and standard deviations computed exclusively over historical windows ending at origin $t$.
  - **Calendar Indicators:** Target month, day of week, day of year, and cyclical Fourier terms ($\sin, \cos$) for seasonal periodicity.
  - **Categorical:** One-hot or ordinal encoding of `City` to capture regional municipal baselines.
- **Leakage Prevention:** Learned preprocessing (median imputers, scalers, target encodings) was fit strictly on historical training data.

## 5. Forecasting Models Evaluated (Phase 7)
1. **Naive Persistence Baseline:** Deterministic benchmark ($\widehat{\text{AQI}}_{c, t+1} = \text{AQI}_{c, t}$).
2. **Ridge Regression:** Linear regularized model ($\alpha=100.0$) with one-hot city encoding, training median imputation, and standard scaling.
3. **Random Forest Regressor (Champion):** Non-linear ensemble (`n_estimators=100, max_depth=10, min_samples_split=5`) with one-hot city encoding and training median imputation.
4. **HistGradientBoostingRegressor:** Non-linear gradient boosting (`lr=0.03, max_iter=200, max_depth=8, l2=1.0`) with ordinal city encoding and native histogram binning for missing values.

## 6. Validation and Test Design
Model selection was governed by a 3-fold expanding window cross-validation protocol over training data, with final architecture selection confirmed on the designated validation period (`2018-07-01` to `2018-12-31`).

Two evaluation cohorts were established:
- **Cohort A (Shared Benchmark Cohort):** Rows where both origin $\text{AQI}_t$ and $\text{Target\_AQI}_{t+1}$ are observed ($\Delta D = 1\text{ day}$), evaluated across all models on identical row identifiers.
- **Cohort B (Model-Only Operational Cohort):** All target-valid rows, including monitor outage days where origin $\text{AQI}_t$ was unobserved.

## 7. Key Verified Results

### Validation Confirmation (Phase 7, Shared Cohort $N=2,982$)
- **Random Forest** achieved the lowest validation error: **MAE $27.3559\text{ AQI points}$** (+7.70% improvement over Persistence), **RMSE $59.7368\text{ AQI points}$** (+14.78% error reduction), and **$R^2 = 0.8774$**, securing designation as the project champion.
- **HistGradientBoosting** achieved validation MAE $28.1849$ (+4.90% vs Persistence) and RMSE $60.2068$ (+14.11%).

### Primary Held-Out Test Evaluation (Phase 8, 2019 Full Year, Shared Cohort $N=6,201$)
- **Naive Persistence Baseline:** MAE = $31.0419$, RMSE = $70.2360$, $R^2 = 0.7445$, MBE = $+0.3614$
- **Ridge Regression:** MAE = $29.0261$ ($+6.49\%$), RMSE = $55.8918$ ($+20.42\%$), $R^2 = 0.8382$, MBE = $+2.8414$
- **Random Forest (Champion):** MAE = **$26.2993\text{ AQI points}$** (**$+15.28\%$ error reduction**), RMSE = **$55.6234\text{ AQI points}$** (**$+20.81\%$ error reduction**), $R^2 = \mathbf{0.8398}$, MBE = **$+4.8666$**
- **HistGradientBoosting:** MAE = $27.1402$ ($+12.57\%$), RMSE = **$54.9397\text{ AQI points}$** (**$+21.78\%$ error reduction**), $R^2 = \mathbf{0.8437}$, MBE = $+5.6109$

*Model Trade-off Disclosed:* Random Forest confirmed the lowest MAE and superior classification fidelity, while HistGradientBoosting achieved a slightly lower RMSE ($54.94$ vs $55.62$). In adherence to evaluation protocols, Random Forest was preserved as champion without retroactive re-selection.

### 2020 COVID-19 Lockdown Stress Period (Shared Cohort $N=3,124$)
- Random Forest achieved **MAE $19.3288\text{ AQI points}$** (+10.64% over Persistence $21.6306$) and **RMSE $39.1893\text{ AQI points}$** (+14.69% over Persistence $45.9400$), demonstrating operational robustness during unprecedented socio-economic disruption.

## 8. Classification and Residual Diagnostics

### CPCB Hazard Category Mapping
Continuous predictions were mapped into the 6 statutory CPCB hazard classes:
- Good ($0\text{--}50$), Satisfactory ($51\text{--}100$), Moderate ($101\text{--}200$), Poor ($201\text{--}300$), Very Poor ($301\text{--}400$), Severe ($>400$).
- **Random Forest Performance (2019):** Overall Accuracy = **$74.36\%$**, Macro F1 = **$0.6678$**, Balanced Accuracy = **$0.6545$**.
- **Severe Hazard Recall ($>400$):** **$83.68\%$** ($282 / 337$ catastrophic pollution events correctly anticipated).

### Subgroup & Anomaly Analysis
- **Seasonal Patterns:** Errors peaked in Winter (MAE $33.23$, RMSE $71.44$) due to boundary-layer inversions and agricultural burning; lowest errors occurred during Monsoon (MAE $19.67$, RMSE $41.75$) due to precipitation scavenging.
- **Ahmedabad Anomaly:** In 2019 Cohort A, Ahmedabad accounts for $347$ rows ($5.6\%$), but contributes **$72.59\%$ of total network squared error** ($13,926,980.55 / 19,185,641.08$) due to extreme sensor saturation ($\text{CO} > 400\text{ mg/m}^3$). Excluding Ahmedabad drops Random Forest RMSE from $55.62$ to **$29.97\text{ AQI points}$** ($R^2 = \mathbf{0.8931}$).

## 9. Limitations
1. **Non-Causal Architecture:** The models capture empirical temporal correlations; they do not establish causal atmospheric mechanisms.
2. **Statutory Scale Ceiling:** The CPCB index caps sub-indices at 500; unbounded regressors extrapolate beyond 500 on extreme days. Mapping continuous predictions $>400$ to `Severe` is an operational convention for forecasts.
3. **Observational Incompleteness:** Monitor telemetry contains non-random sensor outages, addressed via bounded historical forward-filling and median imputation.

## 10. Conclusion
The project successfully demonstrates that a non-linear ensemble (Random Forest) trained on temporal lag dynamics and multi-pollutant telemetry outperforms naive persistence by **$+15.28\%$ in MAE** and **$+20.81\%$ in RMSE** across major Indian urban centers. All deliverables, serialized pipelines, and evaluation metrics are verified, reproducible, and ready for academic submission.
