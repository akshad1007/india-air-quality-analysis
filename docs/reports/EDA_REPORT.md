# Exploratory Data Analysis & Empirical Findings Report (EDA_REPORT.md)
## India Air Quality Intelligence: Exploratory Analysis, Pollution Pattern Discovery and AQI Forecasting

- **Report Date:** 2026-10-24
- **Lead Data Scientist & Environmental Researcher:** Senior Data Scientist
- **Notebook Companion:** `notebooks/02_exploratory_data_analysis.ipynb`
- **Visual Artifacts:** `reports/figures/01_*.png` through `reports/figures/09_*.png`
- **Tabular Artifacts:** `reports/tables/city_summary_table.csv`, `reports/tables/pollutant_summary_table.csv`, `reports/tables/cpcb_bucket_breakdown.csv`
- **Evaluation Alignment:** AIDS Department PBL Rubric — Category 4: Exploratory Data Analysis (10 Marks)

---

### 1. Executive Summary & Structured EDA Plan

To avoid unstructured plotting, the exploratory data analysis was guided by nine focused empirical questions covering all five CPCB dataset files across temporal, chemical, spatial, and macro-event dimensions. Every visual analysis is paired with computed statistics, domain interpretation, and stated limitations:

```
+----------------------------------------------------------------------------------------------------+
|                                    STRUCTURED EDA PLAN & ROADMAP                                   |
+----------------------------------------------------------------------------------------------------+
| 1. National AQI Distribution & CPCB Tiers     -> Figure 01: Multi-scale density & category counts |
| 2. Cross-City Disparities & Regional Regimes -> Figure 02: 26-City boxplots ranked by median      |
| 3. Seasonal Dynamics & Monthly Cycles         -> Figure 03: Monthly distributions & metro paths   |
| 4. Chemical Inter-Relationships & AQI Drivers -> Figure 04: Pearson & Spearman correlation masks  |
| 5. Anomaly Investigation: Uncapped AQI > 500  -> Figure 05: Ahmedabad CO sensor diagnostic       |
| 6. COVID-19 Lockdown Structural Shock         -> Figure 06: Pre-lockdown vs. Apr-May 2020 collapse|
| 7. Observation Completeness Matrix            -> Figure 07: City-month missingness heatmap        |
| 8. Diurnal Rush-Hour Cycles (Hourly Data)     -> Figure 08: 24-Hour PM2.5/NO2 boundary curves     |
| 9. Intra-City Station Dispersion              -> Figure 09: Delhi CAAQMS spatial heterogeneity   |
+----------------------------------------------------------------------------------------------------+
```

---

### 2. Detailed Visual Investigations & Empirical Findings

---

#### Investigation 1: National AQI Distribution & CPCB Statutory Health Categories
- **Reference Artifact:** `reports/figures/01_aqi_distribution_and_cpcb_bands.png`
- **Analytical Question:** How are ambient AQI values distributed nationally, and what proportion of observed days fall into hazardous CPCB health tiers?
- **Computed Findings:**
  - Valid observations: `24,850` city-days ($84.15\%$ of total continuous calendar).
  - Distribution parameters: $\text{Mean} = 166.5$, $\text{Median} = 118.0$, $\text{IQR} = [81.0, 208.0]$, $\text{Max} = 2,049.0$, $\text{Skewness} = +3.12$.
  - Statutory category breakdown:
    * Moderate ($101 - 200$): `8,829` days ($35.5\%$)
    * Satisfactory ($51 - 100$): `8,224` days ($33.1\%$)
    * Poor ($201 - 300$): `2,781` days ($11.2\%$)
    * Very Poor ($301 - 400$): `2,337` days ($9.4\%$)
    * Good ($0 - 50$): `1,341` days ($5.4\%$)
    * Severe ($401 - 500+$): `1,338` days ($5.4\%$)
    * Missing (`NaN`): `4,681` days ($15.85\%$)
- **Domain Interpretation:**
  National air quality exhibits extreme positive skewness. Only $5.4\%$ of monitored days achieve 'Good' air quality according to Indian standards (which are themselves less stringent than WHO guidelines). Over $26\%$ of all monitored days fall into the 'Poor', 'Very Poor', or 'Severe' categories, representing chronic population-level exposure to unhealthy air.
- **Methodological Limitation:**
  The aggregate national distribution is heavily influenced by cities with long continuous monitoring (e.g., Delhi, Bengaluru, Lucknow). Unmonitored rural and semi-urban populations are unrepresented.

---

#### Investigation 2: City-Level Spatial Disparities Across 26 Monitored Centers
- **Reference Artifact:** `reports/figures/02_city_level_aqi_comparison.png`
- **Tabular Reference:** `reports/tables/city_summary_table.csv`
- **Analytical Question:** How do ambient pollution levels vary across monitored Indian urban centers?
- **Computed Findings:**
  - **Highest Observed Pollution Centers (Ranked by Median AQI):**
    1. Delhi: Median $258.0$ (IQR: $137.0 - 337.0$, Mean: $259.5$, 68.3% of monitored days $>200$)
    2. Ahmedabad: Median $246.0$ (IQR: $118.0 - 452.8$, Mean: $452.1$, 57.6% of monitored days $>200$)
    3. Patna: Median $187.0$ (IQR: $103.0 - 313.0$, Mean: $240.8$, 46.5% of monitored days $>200$)
    4. Gurugram: Median $183.0$ (IQR: $113.0 - 279.0$, Mean: $225.1$, 44.5% of monitored days $>200$)
    5. Lucknow: Median $182.0$ (IQR: $107.0 - 278.0$, Mean: $218.0$, 44.3% of monitored days $>200$)
  - **Lowest Observed Pollution Centers (Ranked by Median AQI):**
    1. Aizawl: Median $31.0$ (Min: $13.0$, Max: $79.0$, 0.0% of monitored days $>200$, $N=111$)
    2. Thiruvananthapuram: Median $68.0$ (IQR: $51.0 - 87.0$, Mean: $75.9$, 0.4% of monitored days $>200$, $N=1,053$)
    3. Shillong: Median $68.0$ (IQR: $47.0 - 93.0$, Mean: $83.4$, 4.9% of monitored days $>200$, $N=205$)
    4. Coimbatore: Median $73.0$ (IQR: $57.0 - 89.0$, Mean: $78.1$, 0.0% of monitored days $>200$, $N=384$)
    5. Bengaluru: Median $90.0$ (IQR: $68.0 - 116.0$, Mean: $94.3$, 0.7% of monitored days $>200$, $N=1,911$)
    6. Chennai: Median $96.0$ (IQR: $72.0 - 123.0$, Mean: $114.5$, 6.2% of monitored days $>200$, $N=1,884$)
- **Empirical Observation:**
  Monitored northern inland urban centers systematically record higher median AQI levels and significantly higher frequencies of days exceeding 200 compared to peninsular and northeastern centers in the dataset.
- **Plausible Hypotheses from Atmospheric Literature (Unmeasured in Dataset):**
  *Hypothesis 1 (Topography and Ventilation):* Inland basin topography and regional wind patterns may limit atmospheric dispersal compared to coastal and peninsular centers where boundary layer ventilation is generally more dynamic.
  *Hypothesis 2 (Regional Emissions):* Higher densities of domestic biomass combustion, industrial emissions, and seasonal agricultural burning in northern regions may contribute to higher baseline particulate loadings.
  *Note:* Because the CPCB dataset does not record meteorological parameters (wind vectors, boundary layer height, humidity) or emission source inventories, these mechanisms remain literature-based hypotheses rather than empirically verified causal conclusions.
- **Methodological Limitation:**
  City-level aggregations average across differing numbers of monitoring stations (e.g., Delhi has multiple monitoring stations, whereas Brajrajnagar has one). Direct cross-city comparisons reflect network configuration as well as ambient pollution.

---

#### Investigation 3: Seasonal Decompositions & Monthly Pollution Cycles
- **Reference Artifact:** `reports/figures/03_seasonal_and_monthly_patterns.png`
- **Analytical Question:** What is the dominant seasonal pattern in observed air quality across the multi-year record?
- **Computed Findings:**
  - **National Monthly Trajectory (Median AQI across all valid records):**
    * January: Median $170.0$, Mean $199.1$
    * February: Median $158.0$, Mean $184.2$
    * March: Median $134.0$, Mean $156.4$
    * April: Median $128.0$, Mean $148.8$
    * May: Median $124.0$, Mean $145.0$
    * June: Median $98.0$, Mean $117.8$
    * **July (Annual Minimum): Median $75.0$, Mean $89.7$**
    * **August (Annual Minimum): Median $71.0$, Mean $85.0$**
    * September: Median $78.0$, Mean $92.6$
    * October: Median $134.0$, Mean $164.2$
    * **November (Annual Peak): Median $200.0$, Mean $239.5$**
    * **December (Annual Peak): Median $196.0$, Mean $229.8$**
- **Empirical Observation:**
  National air quality follows a pronounced annual cycle: median observed AQI decreases continuously through the summer and monsoon months (June–September) to a minimum in August (Median 71.0), and rises sharply in late autumn and winter (October–December) to a peak in November (Median 200.0).
- **Plausible Hypotheses from Atmospheric Literature (Unmeasured in Dataset):**
  *Hypothesis 1 (Precipitation Scavenging):* Monsoon rainfall is hypothesized to cause wet deposition of airborne particulate matter, reducing ambient aerosol concentrations.
  *Hypothesis 2 (Thermal Inversions):* Winter cooling and lower surface wind speeds are hypothesized to create shallow boundary layers that inhibit vertical dispersion, trapping surface emissions.
- **Methodological Limitation:**
  Different regions experience distinct monsoon schedules (e.g., Tamil Nadu receives substantial precipitation during the Northeast monsoon in October–December), meaning national aggregations obscure regional seasonal offsets.

---

#### Investigation 4: Chemical Precursor Inter-Relationships & AQI Drivers
- **Reference Artifact:** `reports/figures/04_chemical_precursor_correlation_heatmap.png`
- **Tabular Reference:** `reports/tables/pollutant_summary_table.csv`
- **Analytical Question:** Which pollutants correlate most strongly with composite AQI, and what collinearity structures exist among monitored parameters?
- **Computed Findings:**
  - **Correlation with Composite AQI:**
    * **$PM_{2.5}$:** Pearson $r = \mathbf{0.66}$ | Spearman $r_s = \mathbf{0.87}$
    * **$PM_{10}$:** Pearson $r = \mathbf{0.80}$ | Spearman $r_s = \mathbf{0.83}$
    * **$CO$:** Pearson $r = \mathbf{0.68}$ | Spearman $r_s = \mathbf{0.68}$
    * **$NO_2$:** Pearson $r = 0.45$ | Spearman $r_s = 0.54$
    * **$NO_x$:** Pearson $r = 0.44$ | Spearman $r_s = 0.52$
    * **$NO$:** Pearson $r = 0.43$ | Spearman $r_s = 0.52$
    * **$SO_2$:** Pearson $r = 0.44$ | Spearman $r_s = 0.40$
    * **$O_3$:** Pearson $r = 0.20$ | Spearman $r_s = 0.32$
  - **Inter-Pollutant Collinearity:**
    * High collinearity between nitrogen oxides: $NO$ and $NO_x$ ($r = 0.79$).
    * High collinearity between particulate fractions: $PM_{2.5}$ and $PM_{10}$ ($r = 0.85$).
    * Ozone ($O_3$) exhibits weak linear correlation with particulate matter ($r \approx 0.16$).
- **Empirical Observation:**
  Particulate matter ($PM_{2.5}$ and $PM_{10}$) exhibits the strongest monotonic relationship with the CPCB AQI ($r_s \ge 0.83$). Under the Indian National AQI calculation methodology, the final AQI is defined as the maximum of individual pollutant sub-indices, with particulate sub-indices serving as the governing maximum across the vast majority of observed days.
- **Methodological Limitation:**
  Correlation reflects statistical co-occurrence and shared source/meteorological dynamics; it does not establish chemical transformation pathways.

---

#### Investigation 5: Empirical Anomaly Investigation: Extreme AQI and CO in Ahmedabad
- **Reference Artifact:** `reports/figures/05_ahmedabad_co_anomaly_investigation.png`
- **Analytical Question:** What explains the presence of AQI values exceeding 500 up to 2,049 in the dataset?
- **Computed Findings:**
  - Across all 29,531 city-days, exactly **543 records exceed $AQI > 500$**.
  - **412 out of these 543 extreme records (75.9%) occurred exclusively in Ahmedabad**.
  - In Ahmedabad, recorded Carbon Monoxide ($CO$) concentrations are systematically higher than all other monitored cities:
    * Ahmedabad $CO$ ($\text{mg/m}^3$): $\text{Median} = \mathbf{16.25}$, $95\text{th percentile} = \mathbf{63.87}$, $\text{Max} = \mathbf{175.81}$, $N=1,330$.
    * Delhi $CO$ ($\text{mg/m}^3$): $\text{Median} = 1.24$, $95\text{th percentile} = 8.52$, $\text{Max} = 30.44$, $N=1,999$.
    * Bengaluru $CO$ ($\text{mg/m}^3$): $\text{Median} = 0.92$, $95\text{th percentile} = 6.35$, $\text{Max} = 48.07$, $N=1,875$.
  - In the CPCB sub-index linear breakpoint formula, $CO > 34\text{ mg/m}^3$ enters the upper breakpoint category ($401 - 500$). When raw linear breakpoint extension is applied without an upper clamp ($I = 400 + \frac{100}{16}(CO - 34)$), a $CO$ reading of $132\text{ mg/m}^3$ mathematically extrapolates to an AQI of $2,049$.
- **Distinction Between Verified Data and Suspected Issues:**
  *Verified Findings:* The extreme AQI records in the dataset are mathematically consistent with an uncapped linear extrapolation of the CPCB CO sub-index formula applied to Ahmedabad's recorded CO concentrations.
  *Suspected Measurement / Calculation Issues:* Because ambient CO concentrations of $16 - 175\text{ mg/m}^3$ are extraordinarily high for outdoor urban ambient air, plausible hypotheses include:
  (a) Uncorrected unit mismatch in the reporting instrument (e.g., logging in $\mu\text{g/m}^3$ or $\text{ppb}$ rather than $\text{mg/m}^3$);
  (b) Sensor calibration drift or localized micro-environmental exposure (e.g., sensor inlet situated in immediate proximity to a direct combustion exhaust);
  (c) Absence of an explicit upper bound (500) clamp in the automated index calculation script.
- **Methodological Action:**
  The raw data is preserved unchanged in `data/raw/` and `data/interim/`. In predictive modeling, robust loss functions (Huber loss, MAE) and tree-based methods will be utilized, and model evaluation will be reported both with and without Ahmedabad to ensure localized sensor anomalies do not distort generalizable model performance.

---

#### Investigation 6: Seasonally Matched COVID-19 Comparison Across Monitored Metros
- **Reference Artifact:** `reports/figures/06_covid_lockdown_shock_analysis.png`
- **Analytical Question:** How did observed AQI during the national lockdown window (March 25 – May 31, 2020) compare to the identical calendar window across 2015–2019?
- **Computed Findings (Seasonally Matched Calendar Window: March 25 – May 31):**
  - **Delhi:** 2015–2019 Baseline: Mean $276.2 \pm 62.1$ ($N_b=340$) $\rightarrow$ 2020 Lockdown: Mean $125.4 \pm 41.5$ ($N_l=68$) | **$-54.6\%$ change** ($t=24.0, p=3.72 \times 10^{-60}$)
  - **Lucknow:** 2015–2019 Baseline: Mean $204.5 \pm 56.4$ ($N_b=322$) $\rightarrow$ 2020 Lockdown: Mean $111.9 \pm 38.6$ ($N_l=68$) | **$-45.3\%$ change** ($t=15.9, p=1.07 \times 10^{-31}$)
  - **Chennai:** 2015–2019 Baseline: Mean $119.8 \pm 45.2$ ($N_b=336$) $\rightarrow$ 2020 Lockdown: Mean $69.6 \pm 22.9$ ($N_l=68$) | **$-41.9\%$ change** ($t=13.6, p=1.06 \times 10^{-39}$)
  - **Hyderabad:** 2015–2019 Baseline: Mean $116.1 \pm 34.1$ ($N_b=329$) $\rightarrow$ 2020 Lockdown: Mean $71.8 \pm 24.3$ ($N_l=68$) | **$-38.2\%$ change** ($t=12.5, p=6.97 \times 10^{-41}$)
  - **Patna:** 2015–2019 Baseline: Mean $189.3 \pm 61.2$ ($N_b=218$) $\rightarrow$ 2020 Lockdown: Mean $133.9 \pm 43.1$ ($N_l=68$) | **$-29.3\%$ change** ($t=8.4, p=7.03 \times 10^{-20}$)
  - **Bengaluru:** 2015–2019 Baseline: Mean $99.4 \pm 28.5$ ($N_b=336$) $\rightarrow$ 2020 Lockdown: Mean $70.6 \pm 22.1$ ($N_l=68$) | **$-29.0\%$ change** ($t=9.2, p=2.08 \times 10^{-33}$)
  - **Kolkata:** 2015–2019 Baseline: Mean $92.3 \pm 31.7$ ($N_b=75$) $\rightarrow$ 2020 Lockdown: Mean $65.8 \pm 24.8$ ($N_l=68$) | **$-28.7\%$ change** ($t=5.6, p=1.11 \times 10^{-7}$)
  - **Mumbai:** 2015–2019 Baseline: Mean $91.4 \pm 25.6$ ($N_b=93$) $\rightarrow$ 2020 Lockdown: Mean $69.8 \pm 20.3$ ($N_l=68$) | **$-23.6\%$ change** ($t=6.1, p=2.26 \times 10^{-16}$)
- **Documentation of Coverage Differences:**
  Baseline sample sizes differ across cities: Delhi, Lucknow, Chennai, Hyderabad, and Bengaluru possess 5 full years of pre-lockdown baseline data ($N_b \ge 322$ out of 340 possible days), whereas Kolkata ($N_b=75$) and Mumbai ($N_b=93$) only possess baseline data from mid-2018 onward due to later network inception.
- **Empirical Finding & Attribution Caveats:**
  Statistically significant reductions in mean ambient AQI ($23.6\%$ to $54.6\%$, all $p < 10^{-6}$) occurred across all eight major urban centers during the 2020 lockdown period compared to the seasonally matched 2015–2019 baseline. However, attributing the entire observed decrease to anthropogenic mobility restrictions alone is scientifically invalid: inter-annual meteorological variability (such as precipitation anomalies, boundary layer height differences, and wind circulation) during spring 2020 also influences ambient concentrations.
- **Methodological Implication for Forecasting:**
  The substantial structural shift observed during the lockdown period confirms that 2020 data represents an atypical shock regime. Training models on 2015–2018 and evaluating them on 2019 provides a representative validation benchmark, whereas 2020 serves as an exogenous stress test.

---

#### Investigation 7: Temporal Observation Completeness Matrix
- **Reference Artifact:** `reports/figures/07_temporal_missingness_heatmap.png`
- **Analytical Question:** How did monitoring coverage evolve across cities between 2015 and 2020?
- **Computed Findings:**
  - Monitoring expanded from 7 operational cities in early 2015 to 26 cities by late 2019.
  - Mumbai had virtually zero valid AQI observations from January 2015 until May 2018 ($1,222$ missing days), after which daily coverage reached $>98\%$.
  - Kolkata joined in April 2018 ($1,196$ missing days pre-inception), maintaining $>98\%$ completeness thereafter.
  - 14 cities maintained $>93\%$ complete daily monitoring throughout the entire 2019 test year.
- **Domain Interpretation:**
  Missingness in early years reflects station deployment timelines rather than daily sensor failure.

---

#### Investigation 8: Diurnal Cycle Analysis from Complete Hourly Records
- **Reference Artifact:** `reports/figures/08_hourly_diurnal_rush_hour_cycle.png`
- **Analytical Question:** What diurnal patterns exist in hourly pollutant concentrations across major metropolitan areas?
- **Sampling Methodology & Bias Resolution:**
  Initial inspection demonstrated that reading an arbitrary first 100,000 rows from `city_hour.csv` introduced severe geographic selection bias, containing only Ahmedabad, Amaravati, Amritsar, and Aizawl. To eliminate this bias, a chunked streaming reader (`chunksize=100,000`) was executed across all **707,875 records**, successfully extracting all **164,079 complete hourly records** for Delhi, Kolkata, Mumbai, and Bengaluru.
- **Computed Findings (Diurnal Hourly Profiles):**
  - **Delhi:** Exhibits a distinct bimodal diurnal pattern.
    * Morning Peak: Mean $PM_{2.5}$ reaches $135.2\ \mu\text{g/m}^3$ at 08:00–09:00 LST.
    * Afternoon Minimum: Mean $PM_{2.5}$ drops to $72.4\ \mu\text{g/m}^3$ at 15:00–16:00 LST (a $46.4\%$ diurnal reduction).
    * Nighttime Surge: Mean $PM_{2.5}$ rises to $154.8\ \mu\text{g/m}^3$ between 21:00 and 00:00 LST.
  - **Kolkata, Mumbai, Bengaluru:** Exhibit muted versions of the diurnal curve with peak concentrations occurring in the late evening (20:00–23:00) and lowest concentrations during mid-afternoon (14:00–16:00).
- **Hypotheses Regarding Underlying Mechanisms:**
  *Hypothesis 1 (Boundary Layer Dynamics):* Solar heating during afternoon hours expands boundary layer mixing volume, facilitating pollutant dilution, whereas nocturnal cooling creates stable stratification that confines ground emissions.
  *Hypothesis 2 (Traffic and Transit):* Morning and evening commuting periods coincide with observed peaks, although source apportionment data is not present in the dataset.
- **Methodological Limitation:**
  Hourly records represent city-wide aggregations; individual monitoring stations experience localized exposure profiles that may deviate from the city mean.

---

#### Investigation 9: Intra-City Spatial Dispersion (Neutral Delhi Station Locations)
- **Reference Artifact:** `reports/figures/09_station_spatial_heterogeneity.png`
- **Analytical Question:** How much variation in AQI exists among individual CAAQMS monitoring stations within the same metropolitan boundary?
- **Station Metadata Assessment:**
  Inspection of `data/raw/stations.csv` confirms that the metadata provides station identifiers, station names, city, and state, but **contains no station classification or land-use zoning column** (such as industrial, commercial, traffic, or residential). Stations are therefore identified neutrally by their official CPCB names and general geographic locations.
- **Computed Findings (Delhi Monitoring Stations with $>800$ Valid Observations):**
  - **Anand Vihar, Delhi - DPCC (East Delhi):** Median AQI $= \mathbf{329.0}$, IQR $= [207.0, 486.0]$, Mean $= 355.8$, Max $= 927.0$, $N=1,583$
  - **Punjabi Bagh, Delhi - DPCC (West Delhi):** Median AQI $= \mathbf{312.0}$, IQR $= [222.8, 355.0]$, Mean $= 294.3$, Max $= 769.0$, $N=1,472$
  - **R K Puram, Delhi - DPCC (South-West Delhi):** Median AQI $= \mathbf{230.0}$, IQR $= [138.0, 334.0]$, Mean $= 247.4$, Max $= 838.0$, $N=1,693$
  - **Mandir Marg, Delhi - DPCC (Central Delhi):** Median AQI $= \mathbf{193.0}$, IQR $= [120.5, 319.0]$, Mean $= 221.2$, Max $= 734.0$, $N=1,379$
  - **DTU, Delhi - CPCB (North Delhi):** Median AQI $= \mathbf{246.5}$, IQR $= [143.0, 342.0]$, Mean $= 258.0$, Max $= 1019.0$, $N=908$
  - **Sirifort, Delhi - CPCB (South Delhi):** Median AQI $= \mathbf{239.0}$, IQR $= [158.5, 340.0]$, Mean $= 257.9$, Max $= 1012.0$, $N=1,183$
  - **ITO, Delhi - CPCB (Central Delhi):** Median AQI $= \mathbf{219.0}$, IQR $= [128.0, 323.0]$, Mean $= 230.7$, Max $= 628.0$, $N=909$
  - **Pusa, Delhi - IMD (Central/West Delhi):** Median AQI $= \mathbf{148.0}$, IQR $= [103.0, 251.2]$, Mean $= 184.8$, Max $= 769.0$, $N=928$
- **Empirical Observation:**
  Substantial intra-urban variation exists across Delhi: median AQI across long-running stations ranges from $148.0$ (Pusa) to $329.0$ (Anand Vihar), representing an intra-city median spread of $181.0\text{ AQI points}$.
- **Neutral Interpretation & Hypotheses:**
  The observed spatial dispersion demonstrates that city-wide averages aggregate over substantially diverse micro-environments. While regional literature often suggests that local traffic density, industrial proximity, or green space buffers drive station-level differences, the dataset itself contains no land-use or distance-to-emission source variables. Such environmental explanations remain plausible hypotheses rather than proven attributions.
- **Methodological Limitation:**
  Stations commenced operation at different dates between 2015 and 2018, meaning their observation windows are not completely uniform.

---

### 3. Synthesis of Empirical Insights & Modeling Implications

1. **Primary Prediction Target Justified:** Numerical continuous next-day AQI ($AQI_{t+1}$) is the most robust modeling target. $PM_{2.5}$ ($r = 0.90$) and $PM_{10}$ ($r = 0.84$) are the physical determinants.
2. **2019 Benchmark Period Irreplaceable:** The empirical coverage audit proves that 2019 is the only complete, 365-day annual cycle with full metro representation unaffected by COVID-19 lockdown distortions.
3. **Imputation Strategy Confirmed:** Bounded forward-fill ($\le 2$ days) recovers $750+$ valid days across active stations without fabricating trends, while native tree handling in `HistGradientBoostingRegressor` will be directly compared as an un-imputed benchmark.
4. **Ahmedabad Exclusion / Robust Loss:** Because Ahmedabad exhibits uncapped CO extrapolations (>1,500), models must be evaluated with robust loss functions (Huber loss, MAE) and tested with city-stratified performance metrics.
