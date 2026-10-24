# UI/UX Design Brief & Visualization Specification (UI_UX_BRIEF.md)
## India Air Quality Intelligence: Exploratory Analysis, Pollution Pattern Discovery and AQI Forecasting

- **Design System Version:** 1.1.0 (Audited & Methodologically Revised)
- **Document History:**
  - `v1.0.0` (2026-10-24): Initial UI/UX specification.
  - `v1.1.0` (2026-10-24): Revised following planning audit. Replaced real-time forecasting semantics with explicit "Offline Historical Model Benchmark" labels; integrated dual evaluation views (2019 full-year test vs. 2020 shock analysis) into dashboard visual layouts.
- **Lead Visual Designer & Interface Engineer:** Senior Data Visualization Specialist
- **Interface Framework:** Streamlit (Python 3.10) with Plotly / Matplotlib integration
- **Theme Philosophy:** Modern Scientific Precision — Dark Mode Environmental Intelligence
- **Status:** Approved Post-Audit Baseline

---

### 1. Design Philosophy & Guiding Principles

The India Air Quality Intelligence dashboard is an analytical research instrument, not a marketing SaaS interface. The interface prioritizes **visual integrity, cognitive efficiency, accessible color-coding, and honest uncertainty representation**.

#### 1.1 Four Core Pillars
1. **Analytical Purpose Over Decoration:** Every visual element must answer a specific research question. Decorative graphics, meaningless 3D charts, and low-information widgets are prohibited.
2. **Standardized Environmental Semantics:** All AQI visualizations strictly adhere to the statutory Central Pollution Control Board (CPCB) color bands while maintaining dual visual cues (text labels + numeric values) for accessibility.
3. **Transparent Uncertainty & Baseline Context:** Forecasts must never be presented as single-point deterministic certainties. Confidence ranges, historical baseline comparisons, and model error bands ($\pm \text{MAE}$) must accompany predictions.
4. **Data Density & Restraint:** High information-to-ink ratio (Edward Tufte principle). Clear visual hierarchies, subdued gridlines, and uncluttered typography.

---

### 2. Statutory CPCB AQI Color System & Accessible Standards

The interface maps national air quality categories to statutory CPCB hues with WCAG AA compliant text pairings:

| CPCB Category | AQI Range | Palette Hex Code | Text Contrast Pair | Accessible Icon & Semantic Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Good** | $0 - 50$ | `#10B981` (Emerald Green) | `#FFFFFF` on Dark | 🟢 Minimal health impact |
| **Satisfactory** | $51 - 100$ | `#84CC16` (Lime Green) | `#0E1117` on Light | 🟡 Minor breathing discomfort to sensitive people |
| **Moderate** | $101 - 200$ | `#EAB308` (Amber Gold) | `#0E1117` on Light | 🟠 Breathing discomfort to lungs/asthma patients |
| **Poor** | $201 - 300$ | `#F97316` (Deep Orange) | `#FFFFFF` on Dark | 🔴 Breathing discomfort to most people on prolonged exposure |
| **Very Poor** | $301 - 400$ | `#EF4444` (Crimson Red) | `#FFFFFF` on Dark | 🟣 Respiratory illness on prolonged exposure |
| **Severe** | $401 - 500+$ | `#7F1D1D` (Dark Maroon / Purple) | `#FFFFFF` on Dark | 🟤 Affects healthy people and seriously impacts sensitive |

*Critical Accessibility Rule:* Color is never used as the sole conveyor of information. Every colored badge or chart marker must display the textual category name and numerical value alongside the color.

---

### 3. Visual Layout & Page Blueprints

```
+-----------------------------------------------------------------------------------------+
| [SideBar]                   INDIA AIR QUALITY INTELLIGENCE                              |
| - Overview                  Historical Analytics & Offline Forecasting Benchmark        |
| - Data Quality       +---------------------------------------------------------------+  |
| - Exploratory EDA    | KPI 1: Total Days | KPI 2: Active Cities | KPI 3: 2019 Test   |  |
| - Model Benchmark    | 2,000+ per city   | 10 Validated Metros  | HistGB (MAE: 26.1) |  |
| - Forecaster (Demo)  +---------------------------------------------------------------+  |
| - Policy & Viva                                                                         |
|                      +----------------------------------+----------------------------+  |
| Global Filters:      | 2019 REPRESENTATIVE BENCHMARK    | 2020 COVID SHOCK BENCHMARK |  |
| [City: Delhi      v] | Grouped Bar Chart                | Degradation Scatter / Line |  |
| [Date: 2015-2020  v] | Baseline vs. Ridge vs. HistGB    | Normal vs. Lockdown Error  |  |
|                      |                                  |                            |  |
| [Status: Verified v] +----------------------------------+----------------------------+  |
|                      | INTERPRETATION & METHODOLOGICAL EVIDENCE                      |  |
|                      | "Model outperforms baseline across winter; degrades in 2020..."|  |
+-----------------------------------------------------------------------------------------+
```

---

### 4. Chart Selection & Visualization Standards

| Analytical Objective | Chart Type | Configuration Rules & Constraints | Misleading Pitfalls Prevented |
| :--- | :--- | :--- | :--- |
| **AQI Temporal Trajectory** | Line Chart with Shaded Threshold Bands | X: Continuous Date; Y: AQI; Shaded horizontal bands for Good, Moderate, Severe; Plotly interactive hover. | No truncated Y-axes; Y-axis must start at 0; no linear interpolation over long sensor outages (>7 days). |
| **Seasonal Pollution Cycles** | Monthly Box-and-Whisker Plots | Grouped by Month (Jan–Dec); Median line, IQR box, 1.5 IQR whiskers, outlier points. | Box plots display true skewness and dispersion; avoid simple mean bars that hide extreme winter smog episodes. |
| **Chemical Precursor Cross-Correlation** | Lower-Triangle Correlation Heatmap | Diverging colormap (`RdBu_r` or `Viridis`); Display Pearson $r$ value inside each cell. | Never show full redundant symmetric matrix; mask diagonal ($r=1.0$); sort by hierarchical clustering. |
| **Model vs. Baseline Benchmark** | Grouped Horizontal Bar Charts | Side-by-side grouped bars for Naive, Ridge, RF, HistGB across MAE and RMSE on 2019 test data. | Scale clearly marked in "AQI Units" rather than unitless scores; includes Naive baseline comparator. |
| **Categorical Forecast Accuracy** | Normalized Multi-Class Confusion Matrix | Rows: True CPCB Bucket; Columns: Predicted Bucket; Values: Normalized percentage + count. | Avoid unnormalized counts that mislead due to class imbalance. |
| **Residual Diagnostics** | Scatter Plot ($e_i$ vs. $\hat{y}_i$) & Residual Histogram | Horizontal zero-error reference line; KDE density curve overlaid on histogram. | Assesses heteroscedasticity; reveals underprediction of high pollution peaks. |

---

### 5. Mandatory Disclaimer Policy

All user-facing views containing forecasts must include the following standard banner:
> **Academic Research Prototype:** This offline forecasting benchmark is trained on historical CPCB observations (2015–2020). Predictions are statistical estimates derived from historical autoregressive lags and do not incorporate real-time meteorology or provide operational public health warnings.
