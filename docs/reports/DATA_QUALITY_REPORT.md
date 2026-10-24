# Data Quality & Empirical Assessment Report (DATA_QUALITY_REPORT.md)
## India Air Quality Intelligence: Exploratory Analysis, Pollution Pattern Discovery and AQI Forecasting

- **Report Date:** 2026-10-24
- **Author:** Senior Data Scientist & Analytics Engineer
- **Scope:** Complete Empirical Quality Audit of CPCB India Air Quality Data (2015–2020)
- **Status:** **Verified Quantitative Findings (Pre- and Post-Preprocessing)**

---

### 1. Executive Summary of Data Quality Findings

An empirical audit of all 29,531 daily city observations and 108,035 station records revealed high structural integrity alongside distinct environmental missingness patterns:
1. **Zero Negative Values:** Across all 12 chemical pollutants and AQI, there are **0 negative values** ($<0$), confirming that sensor calibration errors resulting in negative readings were already truncated or removed at the raw CPCB ingestion stage.
2. **Missingness Stratification:** Missingness ranges from **6.97%** for Carbon Monoxide ($CO$) to **61.32%** for $Xylene$. The primary target variable, composite **$AQI$**, is missing in **15.85%** of records (4,681 days), which occurs precisely when fewer than three pollutant sub-indices are available.
3. **CPCB Bucket Fidelity:** Every recorded `AQI_Bucket` corresponds with 100% mathematical fidelity to CPCB statutory bounds (e.g., Good: 13–50, Moderate: 101–200, Severe: 401–2049).
4. **Investigation of AQI > 500 (Ahmedabad Anomaly):** Exactly 543 days exhibit $AQI > 500$. Of these, **412 days (75.9%) occurred in Ahmedabad**, reaching a maximum of $2,049.0$. Deep profiling revealed that Ahmedabad's recorded Carbon Monoxide ($CO$) concentrations are an order of magnitude higher than all other cities ($\text{Median} = 16.25\text{ mg/m}^3$, $\text{Max} = 175.81\text{ mg/m}^3$ vs. Delhi Median $1.24\text{ mg/m}^3$). When the standard linear sub-index formula is applied without an upper clamp, this produces mathematical AQI extrapolations exceeding 1,500.
5. **Staggered Station Inception Dates:** Stations were commissioned between 2015 and 2020. A global 2015–2020 completeness filter would erroneously disqualify major centers like Mumbai (active from May 2018) and Kolkata (active from April 2018).
6. **Empirical Validation of the 2019 Test Period:** In the primary test year (2019), **8 major metropolitan centers exhibit 100.0% continuous daily coverage (365/365 days)**, and 16 cities exhibit $>83\%$ coverage, confirming 2019 as an empirically superior, representative annual out-of-time test period.

---

### 2. Comprehensive Missingness & Distribution Profile (`city_day.csv`)

| Field Name | Missing Count | Missing % | Minimum | Median | Maximum | Physical Units | Negative Count | CPCB NAAQS 24h Standard |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`CO`** | 2,059 | **6.97%** | 0.00 | 0.89 | 175.81 | $mg/m^3$ | 0 | $4.0 mg/m^3$ |
| **`NO`** | 3,582 | **12.13%** | 0.02 | 9.89 | 390.68 | $\mu g/m^3$ | 0 | None (Precursor) |
| **`NO2`** | 3,585 | **12.14%** | 0.01 | 21.69 | 362.21 | $\mu g/m^3$ | 0 | $80 \mu g/m^3$ |
| **`SO2`** | 3,854 | **13.05%** | 0.01 | 9.16 | 193.86 | $\mu g/m^3$ | 0 | $80 \mu g/m^3$ |
| **`O3`** | 4,022 | **13.62%** | 0.01 | 30.84 | 257.73 | $\mu g/m^3$ | 0 | $100 \mu g/m^3$ (8h) |
| **`NOx`** | 4,185 | **14.17%** | 0.00 | 23.52 | 467.63 | $ppb$ / $\mu g/m^3$ | 0 | None (Sum of oxides) |
| **`PM2.5`** | 4,598 | **15.57%** | 0.04 | 48.57 | 949.99 | $\mu g/m^3$ | 0 | $60 \mu g/m^3$ |
| **`AQI`** | 4,681 | **15.85%** | 13.00 | 118.00 | 2049.00 | Dimensionless | 0 | Standard Index |
| **`Benzene`** | 5,623 | **19.04%** | 0.00 | 1.07 | 455.03 | $\mu g/m^3$ | 0 | $5 \mu g/m^3$ (Annual) |
| **`Toluene`** | 8,041 | **27.23%** | 0.00 | 2.97 | 454.85 | $\mu g/m^3$ | 0 | None |
| **`NH3`** | 10,328 | **34.97%** | 0.01 | 15.85 | 352.89 | $\mu g/m^3$ | 0 | $400 \mu g/m^3$ |
| **`PM10`** | 11,140 | **37.72%** | 0.01 | 95.68 | 1000.00 | $\mu g/m^3$ | 0 | $100 \mu g/m^3$ |
| **`Xylene`** | 18,109 | **61.32%** | 0.00 | 0.98 | 170.37 | $\mu g/m^3$ | 0 | None |

---

### 3. CPCB Statutory AQI Bucket Verification & Extrapolation Diagnostic

Cross-tabulation between numerical `AQI` and categorical `AQI_Bucket` validates statutory CPCB definitions:

| CPCB AQI Bucket | Record Count | Observed Min AQI | Observed Max AQI | Statutory CPCB Band | Empirical Alignment |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Good** | 1,341 | 13.0 | 50.0 | $0 - 50$ | 100% Consistent |
| **Satisfactory** | 8,224 | 51.0 | 100.0 | $51 - 100$ | 100% Consistent |
| **Moderate** | 8,829 | 101.0 | 200.0 | $101 - 200$ | 100% Consistent |
| **Poor** | 2,781 | 201.0 | 300.0 | $201 - 300$ | 100% Consistent |
| **Very Poor** | 2,337 | 301.0 | 400.0 | $301 - 400$ | 100% Consistent |
| **Severe** | 1,338 | 401.0 | 2049.0 | $401 - 500+$ | 100% Consistent (Linear Uncapped) |
| **Missing (`NaN`)** | 4,681 | N/A | N/A | N/A | Exactly matches missing AQI |

*Investigation Note:* The CPCB National Air Quality Index report defines 500 as the top category cutoff ('Severe' covers $>400$). In mathematical sub-index calculation, uncapped linear extrapolation generates indices above 500 for extreme pollutant concentrations. Rather than arbitrarily censoring these records, they are preserved in the dataset and examined via robust loss functions (Huber loss and MAE) during model training.

---

### 4. City-Level Coverage and Station Commissioning Analysis

The audit uncovered that cities in the CPCB dataset did not begin monitoring simultaneously:

```
Station Monitoring Onset Timeline:
2015: Delhi (Jan), Ahmedabad (Jan), Bengaluru (Mar), Chennai (Mar), Lucknow (Mar), Hyderabad (Mar), Patna (Oct), Gurugram (Nov)
2016: Visakhapatnam (Jul)
2017: Amritsar (Feb), Jorapokhar (Apr), Jaipur (Jun), Thiruvananthapuram (Jun), Amaravati (Nov), Brajrajnagar (Dec)
2018: Talcher (Feb), Kolkata (Apr), Mumbai (May)
2019: Guwahati (Feb), Coimbatore (Jun), Shillong (Aug), Chandigarh (Sep), Bhopal (Sep)
2020: Kochi (Jan), Ernakulam (Jan), Aizawl (Mar)
```

---

### 5. Empirical Gap Length Distribution Analysis

To evaluate whether a 2-day forward-fill is appropriate, continuous calendar gap lengths were calculated for major metropolitan centers:

| City | Total Calendar Days | Valid Days | Missing Days | Isolated 1-Day Gaps | 2-Day Gaps | 3 to 7-Day Gaps | Gaps > 7 Days | Max Gap (Days) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Delhi** | 2,009 | 1,999 | 10 | 2 | 2 | 1 | 0 | 4 days |
| **Bengaluru** | 2,009 | 1,910 | 99 | 3 | 2 | 3 | 1 (Pre-inception) | 79 days |
| **Chennai** | 2,009 | 1,884 | 125 | 5 | 2 | 2 | 3 (Pre-inception) | 82 days |
| **Hyderabad** | 2,009 | 1,880 | 129 | 7 | 1 | 3 | 2 (Pre-inception) | 89 days |
| **Ahmedabad** | 2,009 | 1,334 | 675 | 16 | 7 | 9 | 4 | 340 days |
| **Mumbai** | 2,009 | 775 | 1,234 | 2 | 1 | 2 | 1 (Pre-May 2018) | 1,222 days |
| **Kolkata** | 2,009 | 754 | 1,255 | 0 | 0 | 0 | 2 (Pre-Apr 2018) | 1,196 days |

#### Empirical Imputation Decisions:
1. **Short Gaps ($\le 2$ days):** Across active monitoring periods, short gaps (1–2 days) account for over 70% of all sensor outages. Forward-filling these isolated dropouts preserves temporal continuity without inventing artificial trends.
2. **Long Gaps ($> 2$ days):** Long gaps represent either pre-commissioning periods or major station shutdowns. These **must not be filled**. They will remain `NaN` and boundary rows will be excluded from target generation.
3. **Native Tree Handling Ablation:** `HistGradientBoostingRegressor` (which natively handles missing values) will be compared directly with and without bounded forward-filling.

---

### 6. Empirical Validation of the 2019 Primary Out-of-Time Test Set

The data proves that **2019 is the single most defensible held-out test year** in the entire dataset:
- **8 major cities had 100.0% complete 365-day monitoring in 2019:** Bengaluru, Mumbai, Lucknow, Hyderabad, Jaipur, Delhi, Chennai, Kolkata.
- **6 additional cities had $>93\%$ complete monitoring in 2019:** Gurugram, Amritsar, Patna, Ahmedabad, Thiruvananthapuram, Visakhapatnam.
- Testing on 2019 guarantees evaluation across all 365 calendar days and all 4 meteorological seasons (Winter, Pre-Monsoon, Monsoon, Stubble Burning/Early Winter), completely free from pandemic anomalies.

---

### 7. Recommendations for Preprocessing and Modeling

1. **City Retention Rule:** Filter cities based on having $\ge 80\%$ valid days during their active monitoring window (or specifically across 2018–2019). Discard cities with $<365$ total records from multi-year predictive modeling, analyzing them in cross-sectional EDA.
2. **Missing Feature Strategy:** High-missingness precursors ($Xylene$ 61.3%, $PM_{10}$ 37.7%, $NH_3$ 35.0%) should not be used as mandatory features for linear models without imputation, but can be leveraged natively by gradient boosting trees.
3. **Target Leakage Prohibition:** Always compute rolling statistics and autoregressive lags on day $T$ shifted by $+1$ relative to target day $T+1$.
