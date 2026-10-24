# Data Preprocessing & Pipeline Report (PREPROCESSING_REPORT.md)
## India Air Quality Intelligence: Exploratory Analysis, Pollution Pattern Discovery and AQI Forecasting

- **Report Date:** 2026-10-24
- **Lead Data Engineer & ML Architect:** Senior ML Engineer
- **Code Artifacts:** `src/preprocessing/cleaner.py`, `tests/test_preprocessing.py`
- **Output Artifacts:** `data/interim/city_day_continuous_raw.parquet`, `data/processed/city_day_preprocessed_ffill.parquet`, `data/processed/city_day_preprocessed_native.parquet`, `data/interim/city_eligibility_matrix.csv`
- **Status:** **Completed & Verified (6/6 Preprocessing Unit Tests Passing)**

---

### 1. Executive Summary

The data preprocessing stage standardizes the raw Central Pollution Control Board (CPCB) observations into continuous, leak-free, model-ready formats while strictly preserving raw data immutability. 

#### Key Methodological Accomplishments:
1. **Raw Data Immutability Certified:** All five raw CSV files in `data/raw/` were verified via automated SHA-256 checksums before and after pipeline execution, guaranteeing zero modification to source data.
2. **Strict Target Non-Imputation:** Under no circumstances are missing target values ($AQI$ or $AQI\_Bucket$) imputed. The pipeline rigorously separates missing predictors from missing labels.
3. **Continuous Calendar Reindexing:** Reindexed each of the 26 monitored cities from its first observation date through 2020-07-01, creating a strictly continuous daily sequence ($\Delta t = 1\text{ day}$) without hidden calendar drops.
4. **Boundary-Safe Predictor Handling:** Evaluated and generated two parallel processed datasets:
   - *Variant A (`city_day_preprocessed_ffill.parquet`):* Bounded forward-fill ($\le 2$ consecutive days) grouped strictly by city. Gaps $> 2$ days remain `NaN`. Zero cross-city temporal spillover.
   - *Variant B (`city_day_preprocessed_native.parquet`):* Zero imputation, preserving native `NaN` values for tree-based estimators (`HistGradientBoostingRegressor` and `LightGBM`).
5. **Empirical City Usability Tiers:** Established a 3-tier eligibility framework based on active monitoring completeness in 2018 (training/val) and 2019 (primary test). No city is excluded from EDA, but only qualified cities participate in multi-year predictive splits.

---

### 2. Raw Data Immutability & Checksum Audit

Before and after executing the cleaning pipeline, SHA-256 hashes were calculated against disk files. All hashes matched the baseline registry with 100% precision:

| Raw CSV File Name | Pre-Execution SHA-256 | Post-Execution SHA-256 | Verification Status |
| :--- | :--- | :--- | :---: |
| **`stations.csv`** | `ce2b7081c3e3c89145613eb724d7d2149ef79c6e16f36bb8a835747a673ed410` | `ce2b7081c3e3c89145613eb724d7d2149ef79c6e16f36bb8a835747a673ed410` | **UNTOUCHED / VERIFIED** |
| **`city_day.csv`** | `0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847` | `0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847` | **UNTOUCHED / VERIFIED** |
| **`city_hour.csv`** | `fae474062ec6b10af7c5b2ac4e32f5c240a0ef80835ee3a50332fec06e2190a4` | `fae474062ec6b10af7c5b2ac4e32f5c240a0ef80835ee3a50332fec06e2190a4` | **UNTOUCHED / VERIFIED** |
| **`station_day.csv`**| `f07787eb4c36c53b07763f591ee29fa01fbaa4d18df1a470447d04def79677b9` | `f07787eb4c36c53b07763f591ee29fa01fbaa4d18df1a470447d04def79677b9` | **UNTOUCHED / VERIFIED** |
| **`station_hour.csv`**| `26a423c66fe9763064d580ffb007150375cfa2056e62b29d52b1b7857af54fde` | `26a423c66fe9763064d580ffb007150375cfa2056e62b29d52b1b7857af54fde` | **UNTOUCHED / VERIFIED** |

---

### 3. Continuous Calendar Reindexing

In the raw dataset, days with total station outages were omitted as missing rows rather than explicit `NaN` entries. Reindexing against continuous date sequences per city exposed these missing dates:
- Raw `city_day.csv` records: `29,531`
- Continuous reindexed records: `29,531` (The raw CPCB dataset had already populated missing calendar rows per city between their initial start date and 2020-07-01).
- Timestamp monotonicity: `100% verified` across all 26 cities (each consecutive row is strictly $\text{Date}_{t} - \text{Date}_{t-1} = 1\text{ day}$).

---

### 4. Target Integrity: Separation of Predictors and Target Labels

A fundamental rule of scientifically defensible forecasting is that **target labels must never be synthetically imputed**.
- If $AQI$ at day $T+1$ is missing, that day cannot serve as a supervised training label or an evaluation ground-truth point.
- **Verification:** In both `city_day_preprocessed_ffill.parquet` and `city_day_preprocessed_native.parquet`, the number of valid $AQI$ values is **exactly 24,850**, identical to the raw dataset. Zero target rows were filled.

---

### 5. Missing Predictor Imputation Audit

For input features ($PM_{2.5}, NO_2, SO_2, CO$, etc.), bounded forward-filling was compared against zero imputation:

| Chemical Pollutant | Pre-Imputation Missing Count | Pre-Imputation Missing % | Post-Imputation Missing Count (Variant A) | Post-Imputation Missing % (Variant A) | Net Recovered Days |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`CO`** | 2,059 | 6.97% | 1,446 | 4.90% | +613 days |
| **`NO`** | 3,582 | 12.13% | 2,827 | 9.57% | +755 days |
| **`NO2`** | 3,585 | 12.14% | 2,818 | 9.54% | +767 days |
| **`SO2`** | 3,854 | 13.05% | 3,074 | 10.41% | +780 days |
| **`O3`** | 4,022 | 13.62% | 3,212 | 10.88% | +810 days |
| **`NOx`** | 4,185 | 14.17% | 3,463 | 11.73% | +722 days |
| **`PM2.5`** | 4,598 | 15.57% | 3,842 | 13.01% | +756 days |
| **`Benzene`** | 5,623 | 19.04% | 4,896 | 16.58% | +727 days |
| **`Toluene`** | 8,041 | 27.23% | 7,143 | 24.19% | +898 days |
| **`NH3`** | 10,328 | 34.97% | 9,335 | 31.61% | +993 days |
| **`PM10`** | 11,140 | 37.72% | 10,211 | 34.58% | +929 days |
| **`Xylene`** | 18,109 | 61.32% | 17,294 | 58.56% | +815 days |
| **`AQI` (Target)** | **4,681** | **15.85%** | **4,681** | **15.85%** | **0 (Strictly Preserved)** |

#### Rules Enforced:
1. **City Boundary Protection:** `ffill` was executed inside `.groupby('City')`. Test `test_no_cross_city_leakage` confirms zero data bled across city transitions.
2. **Gap Threshold Enforcement:** Gaps exceeding 2 consecutive missing days remained `NaN`. Test `test_bounded_ffill_gap_limit` verified that the 3rd missing day is strictly preserved as `NaN`.

---

### 6. Three-Tier City Usability Matrix

To avoid discarding valid real-world data while preventing models from training on unmonitored historical voids, cities were partitioned into three functional tiers:

```
+-------------------------------------------------------------------------------------------------------+
|                                    CITY USABILITY STRATIFICATION                                      |
+-------------------------------------------------------------------------------------------------------+
| TIER 1: PRIMARY MODELING QUALIFIED (18 Cities)                                                        |
| Criteria: >= 200 valid AQI days in 2018 (Train/Val) AND >= 300 valid AQI days in 2019 (Primary Test).  |
| Cities: Chennai (365/365), Bengaluru (361/365), Kolkata (206/365), Jaipur (365/365),                 |
|         Mumbai (227/365), Lucknow (365/365), Hyderabad (365/365), Delhi (365/365),                    |
|         Gurugram (340/364), Amritsar (324/362), Patna (361/359), Ahmedabad (357/352),                |
|         Thiruvananthapuram (361/351), Visakhapatnam (326/341), Brajrajnagar (260/322),                |
|         Amaravati (312/309), Jorapokhar (225/303), Talcher (239/303).                                |
|                                                                                                       |
| TIER 2: SECONDARY TEST QUALIFIED (1 City)                                                             |
| Criteria: >= 250 valid AQI days in 2019 (Test), but lacking 2018 historical training depth.           |
| City: Guwahati (2018: 0 days, 2019: 312 days).                                                        |
|                                                                                                       |
| TIER 3: EDA & CROSS-SECTIONAL ONLY (7 Cities)                                                         |
| Criteria: Insufficient operational depth (<365 total historical records). Included in full EDA,      |
|           excluded from multi-year time-series modeling.                                              |
| Cities: Coimbatore, Chandigarh, Bhopal, Shillong, Aizawl, Kochi, Ernakulam.                           |
+-------------------------------------------------------------------------------------------------------+
```

---

### 7. Automated Test Suite Execution Results

Automated unit tests in `tests/test_preprocessing.py` verified the pipeline's correctness:

```bash
$ pytest tests/test_preprocessing.py
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-7.4.3, pluggy-1.6.0
rootdir: C:\Users\KANHA\Desktop\SPSS PROJECT
configfile: pytest.ini
plugins: anyio-4.10.0, cov-6.2.1, mock-3.12.0
collected 6 items

tests\test_preprocessing.py ......                                       [100%]

============================== 6 passed in 5.04s ==============================
```

- `test_raw_data_immutability`: **PASSED** (Raw hashes unaltered).
- `test_target_aqi_never_imputed`: **PASSED** (Valid AQI count = 24,850 in all variants).
- `test_no_cross_city_leakage`: **PASSED** (Zero values bled across city transitions).
- `test_bounded_ffill_gap_limit`: **PASSED** (Gaps > 2 days preserved as `NaN`).
- `test_monotonic_daily_continuity`: **PASSED** (Continuous 1-day step verified).
- `test_key_uniqueness`: **PASSED** (Zero duplicate `(City, Date)` pairs).
