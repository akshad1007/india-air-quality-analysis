# Planning Audit & Academic Methodological Review
## India Air Quality Intelligence: Exploratory Analysis, Pollution Pattern Discovery and AQI Forecasting

- **Review Date:** 2026-10-24
- **Reviewer Persona:** Senior Data Scientist, Academic Project Mentor & Independent Peer Reviewer
- **Review Scope:** Audit of planning documentation, data acquisition integrity, and empirical quality profiling.
- **Current Status:** **PHASE 2 (ACQUISITION) & PHASE 3 (DATA AUDIT) FULLY VERIFIED & COMPLETE**

---

### 1. Executive Summary & Audit Resolution

Following the initial methodological review (`v1.0.0`), all five raw CPCB CSV files were successfully downloaded, placed in `data/raw/`, and exhaustively profiled. The empirical findings strictly corroborate our architectural decisions:

1. **2019 Primary Out-of-Time Test Set Proven Superior:**
   - Testing on 2019 provides **8 major Indian metropolitan cities with 100.0% continuous daily coverage (365/365 days)**: Bengaluru, Mumbai, Lucknow, Hyderabad, Jaipur, Delhi, Chennai, and Kolkata.
   - Another 6 major centers (Gurugram, Amritsar, Patna, Ahmedabad, Thiruvananthapuram, Visakhapatnam) exhibit between $93.4\%$ and $99.7\%$ completeness.
   - In contrast, the year 2020 terminates on July 1, 2020 (only 6 months), misses the post-monsoon stubble burning and winter peak, and is severely disrupted by the nationwide March–May 2020 COVID-19 lockdown. 2019 is empirically validated as the primary held-out test set, with 2020 retained as a dedicated exogenous shock evaluation set.
2. **Station Metadata Confirmed Free of False Attributes:**
   - Raw inspection of `stations.csv` (230 rows, SHA-256: `ce2b7081...`) confirmed that only 5 columns exist: `StationId`, `StationName`, `City`, `State`, and `Status`.
   - The absence of a `StationType` column is now an empirically verified fact. All speculative claims about "industrial vs. residential station comparisons" have been excised from the project scope.
3. **Sensor Missingness & Clean Negative Value Profile:**
   - Across all 29,531 daily city records, there are **exactly 0 negative concentration values** ($<0$), confirming that raw sensor dropouts below zero were already cleaned at the CPCB collection point.
   - Missingness ranges from 6.97% ($CO$) to 61.32% ($Xylene$). The primary target ($AQI$) is missing in 15.85% of records (4,681 days).
4. **CPCB Statutory Bucket Boundaries Validated:**
   - Every recorded `AQI_Bucket` adheres with 100% precision to statutory CPCB cutoffs (Good $\le 50$, Satisfactory $\le 100$, Moderate $\le 200$, Poor $\le 300$, Very Poor $\le 400$, Severe $> 400$).
   - `AQI_Bucket` is an exact deterministic step function of `AQI`. Primary modeling will be continuous regression, with classification metrics derived via thresholding to guarantee mutual consistency.
5. **Automated Data Validation Test Suite Passes (100%):**
   - The automated pytest suite (`tests/test_data_validation.py`) ran and passed all 7 integrity tests in 2.61 seconds, certifying file existence, checksum matching, key uniqueness, foreign key validity, non-negative bounds, and CPCB bucket ranges.

---

### 2. Comprehensive Status of Planning Issues (`ISS-01` to `ISS-06`)

| Issue ID | Severity | Category | Initial Vulnerability | Empirical Evidence & Resolution | Current Status |
| :---: | :---: | :--- | :--- | :--- | :---: |
| **ISS-01** | **CRITICAL** | Validation Split | Truncated 2020 test set (6 months, COVID lockdown). | 2019 profiled: 14 cities $>93\%$ complete (8 cities 100% complete). 2019 locked as primary test set; 2020 locked as shock study. | **RESOLVED & VERIFIED** |
| **ISS-02** | **HIGH** | Metadata / Schema | Unsupported claim of "industrial vs. residential" stations. | `stations.csv` inspected: 230 rows, 5 columns; confirmed no `StationType`. Scope refined to intra-city station dispersion. | **RESOLVED & VERIFIED** |
| **ISS-03** | **HIGH** | Missingness Handling | Hardcoded 2-day forward-fill & 60% coverage. | Gap analysis complete: short gaps ($\le 2$ days) constitute $>70\%$ of active outages. City coverage redefined by active monitoring window. | **RESOLVED & VERIFIED** |
| **ISS-04** | **MEDIUM** | Model Formulation | Dual model conflict (regression vs. classification). | Verified `AQI_Bucket` is exact step function. Primary model is regression; secondary classification evaluated via threshold cuts. | **RESOLVED & VERIFIED** |
| **ISS-05** | **MEDIUM** | Real-Time Semantics | Unclear distinction between live alerting and offline study. | All documents updated with standard "Offline Historical Forecasting Research Benchmark" disclaimer. | **RESOLVED & VERIFIED** |
| **ISS-06** | **MEDIUM** | Traceability Matrix | Traceability relied on planned filenames rather than evidence. | Updated with quantitative metric thresholds, checksum tables, and automated test pass requirements. | **RESOLVED & VERIFIED** |

---

### 3. Knowledge Classification: Facts vs. Assumptions Post-Audit

#### Tier 1: Empirically Confirmed Facts (Post-Phase 2 & 3)
- All 5 raw CPCB CSV files are stored in `data/raw/` with verified SHA-256 hashes:
  - `stations.csv`: 230 rows, 14,467 bytes (`ce2b7081...`)
  - `city_day.csv`: 29,531 rows, 2,574,056 bytes (`0d84b21c...`)
  - `city_hour.csv`: 707,875 rows, 65,648,581 bytes (`fae47406...`)
  - `station_day.csv`: 108,035 rows, 8,634,105 bytes (`f07787eb...`)
  - `station_hour.csv`: 2,589,083 rows, 219,675,434 bytes (`26a423c6...`)
- Zero negative values across all numerical pollutants in `city_day.csv`.
- `station_day.csv` has 110 active stations; 100% of these stations match `stations.csv` with 0 foreign key orphans.
- 2019 has 8 major cities with 365/365 days of continuous monitoring.
- All 7 automated data validation tests pass via `pytest tests/test_data_validation.py`.

#### Tier 2: Methodological Principles
- Next-day forecasting features must strictly use observations $\le T$ (lagged by at least 1 day).
- Physical pollution spikes ($AQI > 400$) will be retained during evaluation to measure real-world performance under severe smog.

#### Tier 3: Phase 4 Implementation Parameters
- Minimum city completeness threshold set to $\ge 80\%$ of active operational days.
- Isolated dropouts $\le 2$ days will be forward-filled; gaps $> 2$ days remain `NaN` and will not serve as training targets.
- Comparison of forward-fill vs. native tree missingness handling will be conducted in Phase 8.

---

### 4. Remaining Prerequisites Before Phase 4 (Preprocessing Pipeline)

1. **User Review & Approval:** Review this empirical quality audit and verified data inventory.
2. **Explicit Instruction to Proceed:** Upon receiving your approval, we will proceed to **Phase 4 (Data Preprocessing & Cleaning Pipeline)** and **Phase 5 (Statistical Exploratory Data Analysis)**.
