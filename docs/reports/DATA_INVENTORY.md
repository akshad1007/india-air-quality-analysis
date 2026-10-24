# Data Source Inventory & Verification Registry (DATA_INVENTORY.md)
## India Air Quality Intelligence: Exploratory Analysis, Pollution Pattern Discovery and AQI Forecasting

- **Dataset Acquisition Date:** 2026-10-24
- **Source Repository:** Kaggle / Central Pollution Control Board (CPCB), India
- **Dataset Author / Curator:** Rohan Rao (`rohanrao/air-quality-data-in-india`)
- **Dataset URL:** `https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india`
- **Dataset License:** CC BY-SA 4.0 (Open Data Attribution)
- **Local Immutable Raw Storage:** `c:\Users\KANHA\Desktop\SPSS PROJECT\data\raw\`
- **Verification Status:** 100% Acquired, Verified, Preserved Unmodified

---

### 1. Verified Raw File Inventory Registry

Every raw file was acquired, verified via automated SHA-256 hashing, and stored in `data/raw/`. The raw files are write-protected and will remain permanently unmodified throughout the project lifecycle.

| File Name | Size (Bytes) | Exact Row Count | Column Count | Date / Time Range | Primary / Composite Key | Duplicate Keys | SHA-256 Checksum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`stations.csv`** | 14,467 | 230 | 5 | N/A (Metadata) | `StationId` | 0 | `ce2b7081c3e3c89145613eb724d7d2149ef79c6e16f36bb8a835747a673ed410` |
| **`city_day.csv`** | 2,574,056 | 29,531 | 16 | 2015-01-01 to 2020-07-01 | `(City, Date)` | 0 | `0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847` |
| **`city_hour.csv`** | 65,648,581 | 707,875 | 16 | 2015-01-01 01:00:00 to 2020-07-01 00:00:00 | `(City, Datetime)` | 0 | `fae474062ec6b10af7c5b2ac4e32f5c240a0ef80835ee3a50332fec06e2190a4` |
| **`station_day.csv`** | 8,634,105 | 108,035 | 16 | 2015-01-01 to 2020-07-01 | `(StationId, Date)` | 0 | `f07787eb4c36c53b07763f591ee29fa01fbaa4d18df1a470447d04def79677b9` |
| **`station_hour.csv`** | 219,675,434 | 2,589,083 | 16 | 2015-01-01 01:00:00 to 2020-07-01 00:00:00 | `(StationId, Datetime)` | 0 | `26a423c66fe9763064d580ffb007150375cfa2056e62b29d52b1b7857af54fde` |

---

### 2. File-by-File Column Schema and Identification

#### 2.1 `stations.csv` (Station Metadata Registry)
- **Columns (5):** `StationId`, `StationName`, `City`, `State`, `Status`
- **Total Unique Stations:** 230
- **Total Unique Cities in Metadata:** 127
- **Operational Status Counts:** Active: 131, Unspecified (`NaN`): 97, Inactive: 2
- **Key Architectural Confirmation:** Raw metadata confirms **absence** of any `StationType` column (e.g., Industrial, Residential, Commercial).

#### 2.2 `city_day.csv` (Primary Modeling Dataset)
- **Columns (16):** `City`, `Date`, `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`, `AQI`, `AQI_Bucket`
- **Temporal Resolution:** 24-hour daily aggregate
- **Total Unique Cities (26):** Ahmedabad, Aizawl, Amaravati, Amritsar, Bengaluru, Bhopal, Brajrajnagar, Chandigarh, Chennai, Coimbatore, Delhi, Ernakulam, Gurugram, Guwahati, Hyderabad, Jaipur, Jorapokhar, Kochi, Kolkata, Lucknow, Mumbai, Patna, Shillong, Talcher, Thiruvananthapuram, Visakhapatnam.
- **Key Integrity:** Zero duplicate records on `(City, Date)`.

#### 2.3 `city_hour.csv` (Hourly City Breakdown)
- **Columns (16):** `City`, `Datetime`, `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`, `AQI`, `AQI_Bucket`
- **Temporal Resolution:** 1-hour interval
- **Total Rows:** 707,875 across the same 26 cities.
- **Key Integrity:** Zero duplicate records on `(City, Datetime)`.

#### 2.4 `station_day.csv` (Daily Station Breakdown)
- **Columns (16):** `StationId`, `Date`, `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`, `AQI`, `AQI_Bucket`
- **Active Stations with Daily Data:** 110 stations (out of 230 registered stations in `stations.csv`).
- **Key Integrity:** Zero duplicate records on `(StationId, Date)`.

#### 2.5 `station_hour.csv` (Hourly Station Breakdown)
- **Columns (16):** `StationId`, `Datetime`, `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`, `AQI`, `AQI_Bucket`
- **Total Rows:** 2,589,083
- **Active Stations:** 110 stations
- **Key Integrity:** Zero duplicate records on `(StationId, Datetime)`.

---

### 3. Join Viability & Foreign Key Cardinality Audit

To prevent silent multiplication of records and memory leaks, joins were tested empirically:

1. **`station_day.csv` $\bowtie$ `stations.csv` on `StationId`:**
   - Pre-join row count: `108,035`
   - Post-join row count: `108,035`
   - Unmatched station IDs: `0`
   - **Result:** $N:1$ cardinality is strictly preserved. Zero row multiplication.
2. **City Identifier Consistency:**
   - All 26 unique cities present in `city_day.csv` match city names in `stations.csv` with 100% typographical alignment.
3. **Daily $\bowtie$ Hourly Aggregation Rule:**
   - Direct joins between daily and hourly tables are strictly prohibited. Hourly tables will only be analyzed in standalone diurnal workflows or aggregated using explicit 24-hour mean operations.

---

### 4. Preservation & Reproducibility Certification

- The five source CSVs in `data/raw/` are designated immutable.
- Any cleaning, transformation, or feature generation is saved to separate tiers (`data/interim/` and `data/processed/`).
- Checksum validation is enforced automatically in CI/testing via `pytest tests/test_data_validation.py`.
