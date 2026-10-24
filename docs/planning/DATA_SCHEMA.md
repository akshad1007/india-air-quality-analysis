# Data Schema & Data Dictionary (DATA_SCHEMA.md)
## India Air Quality Intelligence: Exploratory Analysis, Pollution Pattern Discovery and AQI Forecasting

- **Document Version:** 1.2.0 (Verified Empirical Specification)
- **Document History:**
  - `v1.0.0` (2026-10-24): Initial provisional schema specification.
  - `v1.1.0` (2026-10-24): Revised following planning audit (removed station-type claims).
  - `v1.2.0` (2026-10-24): **Verified against actual downloaded raw CPCB files.** Updated exact SHA-256 hashes, confirmed row counts, zero-negative concentration audits, foreign key cardinalities, and empirical null percentages.
- **Data Engineering Lead:** Senior Analytics Engineer & Data Architect
- **Source Repository:** Kaggle / CPCB Air Quality Data in India (2015–2020) by Rohan Rao
- **Source URL:** `https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india`
- **Status:** **100% Empirically Verified against Raw CSVs**

---

### 1. Verified Source Dataset Inventory

| File Name | Size (Bytes) | Exact Row Count | Columns | Date Range | Verified SHA-256 Checksum | Join Integrity |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **`stations.csv`** | 14,467 | 230 | 5 | Metadata | `ce2b7081c3e3c89145613eb724d7d2149ef79c6e16f36bb8a835747a673ed410` | Unique PK: `StationId` |
| **`city_day.csv`** | 2,574,056 | 29,531 | 16 | 2015-01-01 to 2020-07-01 | `0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847` | Unique Composite Key: `(City, Date)` |
| **`city_hour.csv`** | 65,648,581 | 707,875 | 16 | 2015-01-01 01:00:00 to 2020-07-01 00:00:00 | `fae474062ec6b10af7c5b2ac4e32f5c240a0ef80835ee3a50332fec06e2190a4` | Unique Composite Key: `(City, Datetime)` |
| **`station_day.csv`** | 8,634,105 | 108,035 | 16 | 2015-01-01 to 2020-07-01 | `f07787eb4c36c53b07763f591ee29fa01fbaa4d18df1a470447d04def79677b9` | Unique Composite Key: `(StationId, Date)` |
| **`station_hour.csv`**| 219,675,434 | 2,589,083 | 16 | 2015-01-01 01:00:00 to 2020-07-01 00:00:00 | `26a423c66fe9763064d580ffb007150375cfa2056e62b29d52b1b7857af54fde` | Unique Composite Key: `(StationId, Datetime)` |

---

### 2. Verified Schema & Dictionary: `city_day.csv` (Primary Modeling Dataset)

| Field Name | Raw Type | Target Type | Observed Range [Min, Max] | Empirical Null % | Units | NAAQS 24h Standard | Physical / Domain Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`City`** | `object` | `category` | 26 Unique Cities | **0.00%** | String | N/A | Monitored Indian metropolitan centers. Unique key component. |
| **`Date`** | `object` | `datetime64[ns]`| [2015-01-01, 2020-07-01] | **0.00%** | `YYYY-MM-DD`| N/A | Observation calendar day. Continuous per city. |
| **`CO`** | `float64` | `float32` | [0.00, 175.81] | **6.97%** | $mg/m^3$ | $4.0 mg/m^3$ | Carbon monoxide. Note unit is $mg$ (not $\mu g$). Zero negatives. |
| **`NO`** | `float64` | `float32` | [0.02, 390.68] | **12.13%** | $\mu g/m^3$ | None | Nitric Oxide. Combustion tracer. Zero negatives. |
| **`NO2`** | `float64` | `float32` | [0.01, 362.21] | **12.14%** | $\mu g/m^3$ | $80 \mu g/m^3$ | Nitrogen Dioxide. Major traffic marker. Zero negatives. |
| **`SO2`** | `float64` | `float32` | [0.01, 193.86] | **13.05%** | $\mu g/m^3$ | $80 \mu g/m^3$ | Sulfur Dioxide. Industrial/power emissions. Zero negatives. |
| **`O3`** | `float64` | `float32` | [0.01, 257.73] | **13.62%** | $\mu g/m^3$ | $100 \mu g/m^3$ (8h)| Ground-level ozone. Secondary photochemical oxidant. Zero negatives. |
| **`NOx`** | `float64` | `float32` | [0.00, 467.63] | **14.17%** | $ppb$ / $\mu g$ | None | Total nitrogen oxides ($NO + NO_2$). Zero negatives. |
| **`PM2.5`** | `float64` | `float32` | [0.04, 949.99] | **15.57%** | $\mu g/m^3$ | $60 \mu g/m^3$ | Fine particulate matter ($<2.5 \mu m$). Primary driver of winter smog. |
| **`AQI`** | `float64` | `float32` | [13.00, 2049.00] | **15.85%** | Index | Standard | Composite CPCB Air Quality Index. Target variable. Zero negatives. |
| **`Benzene`** | `float64` | `float32` | [0.00, 455.03] | **19.04%** | $\mu g/m^3$ | $5 \mu g/m^3$ (Annual)| Toxic aromatic hydrocarbon (VOC). Zero negatives. |
| **`Toluene`** | `float64` | `float32` | [0.00, 454.85] | **27.23%** | $\mu g/m^3$ | None | Industrial solvent tracer (VOC). Zero negatives. |
| **`NH3`** | `float64` | `float32` | [0.01, 352.89] | **34.97%** | $\mu g/m^3$ | $400 \mu g/m^3$ | Ammonia. Agricultural and industrial precursor. Zero negatives. |
| **`PM10`** | `float64` | `float32` | [0.01, 1000.00] | **37.72%** | $\mu g/m^3$ | $100 \mu g/m^3$ | Respirable particulate matter ($<10 \mu m$). Dust and mechanical. |
| **`Xylene`** | `float64` | `float32` | [0.00, 170.37] | **61.32%** | $\mu g/m^3$ | None | Petrochemical aromatic solvent. Highest missingness. Zero negatives. |
| **`AQI_Bucket`**| `object`| `category` | 6 Valid Categories | **15.85%** | Category | 6 Bands | Exact CPCB classification: Good, Satisfactory, Moderate, Poor, Very Poor, Severe. |

---

### 3. Verified Metadata Schema: `stations.csv`

| Field Name | Raw Type | Target Type | Observed Values & Cardinality | Verified Status |
| :--- | :--- | :--- | :--- | :--- |
| **`StationId`** | `object` | `string` | 230 Unique IDs (e.g., `AP001`, `DL001`) | **Strict Primary Key** (100% unique, 0 duplicates). |
| **`StationName`**| `object` | `string` | Geographical landmark (e.g., "Anand Vihar, Delhi - DPCC") | Verified. |
| **`City`** | `object` | `category` | 127 Unique Cities across India | Matches all 26 cities in `city_day.csv`. |
| **`State`** | `object` | `category` | 28 States and Union Territories | Used for regional climatic mapping. |
| **`Status`** | `object` | `category` | Active: 131, Unspecified (`NaN`): 97, Inactive: 2 | Operational status metadata. |

> [!IMPORTANT]
> **Audit Confirmation:** `stations.csv` does **NOT** contain a `StationType` column. Any zoning categorization (e.g. industrial vs. residential) is unsupported by raw data.

---

### 4. Verified Join Cardinality & Data Lineage

- **`station_day.csv` $\bowtie$ `stations.csv` on `StationId`:**
  - Exactly 108,035 rows before and after the join.
  - Zero unmatched station identifiers (`unmatched = 0`).
  - Cardinality is strictly $N:1$.
- **City Mapping:**
  - All 26 cities in `city_day.csv` exist in `stations.csv`.

---

### 5. Engineered Features & Leakage Protection Dictionary

| Feature Name | Type | Formula / Origin | Source Columns | Availability Time | Leakage Risk & Prevention |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`day_of_week`** | Calendar | `Date.dt.dayofweek` | `Date` | Forecast Origin $T$ | Zero risk (Deterministic calendar). |
| **`month`** | Calendar | `Date.dt.month` | `Date` | Forecast Origin $T$ | Zero risk (Deterministic calendar). |
| **`day_of_year`** | Calendar | `Date.dt.dayofyear` | `Date` | Forecast Origin $T$ | Zero risk (Deterministic calendar). |
| **`is_weekend`** | Binary | `1 if day_of_week >= 5 else 0` | `Date` | Forecast Origin $T$ | Zero risk. |
| **`aqi_lag_1`** | Autoregressive | $AQI_{t}$ | `AQI` | Day $T$ (Available for predicting $T+1$) | Shifted by $+1$ day relative to target. Verified safe. |
| **`aqi_lag_2`** | Autoregressive | $AQI_{t-1}$ | `AQI` | Day $T-1$ | Shifted by $+2$ days. Verified safe. |
| **`aqi_lag_7`** | Autoregressive | $AQI_{t-6}$ | `AQI` | Day $T-6$ | Shifted by $+7$ days (Weekly seasonality). Verified safe. |
| **`aqi_roll_mean_7`**| Statistical | $\frac{1}{7} \sum_{i=0}^6 AQI_{t-i}$ | `AQI` | Computed over $T-6$ to $T$ | Window strictly excludes $T+1$. Computed with `shift(1).rolling(7).mean()`. |
| **`aqi_roll_std_7`** | Statistical | $\text{StdDev}(AQI_{t-6 \dots t})$ | `AQI` | Computed over $T-6$ to $T$ | Excludes $T+1$. Measures atmospheric volatility. |
| **`aqi_roll_max_7`** | Statistical | $\max(AQI_{t-6 \dots t})$ | `AQI` | Computed over $T-6$ to $T$ | Excludes $T+1$. Captures recent severe spikes. |
| **`pm25_lag_1`** | Pollutant Lag | $PM2.5_t$ | `PM2.5` | Day $T$ | Shifted by $+1$ day. Primary particulate driver. |
| **`no2_lag_1`** | Pollutant Lag | $NO2_t$ | `NO2` | Day $T$ | Shifted by $+1$ day. Vehicle emission proxy. |

---

### 6. Target Variable Definitions

- **Primary Continuous Regression Target:** `target_aqi_next_day` = $AQI_{t+1}$
- **Secondary Categorical Evaluation Target:** Evaluated by applying standard statutory CPCB cuts to predicted continuous $\hat{y}_{t+1}$:
  - Good: $\le 50$
  - Satisfactory: $51 - 100$
  - Moderate: $101 - 200$
  - Poor: $201 - 300$
  - Very Poor: $301 - 400$
  - Severe: $> 400$
