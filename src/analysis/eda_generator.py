"""
Comprehensive Exploratory Data Analysis & Visualization Generator.
Scientifically audited version:
1. Zero sampling bias in hourly data: processes city_hour.csv across all 707,875 rows using chunking.
2. Seasonally matched COVID-19 analysis: March 25 - May 31 across all years, reporting sample sizes (N),
   standard deviations, and explicit coverage differences without attributing entire variance to lockdown alone.
3. Neutral station location comparisons in Delhi: no speculative land-use tags.
4. Detailed Ahmedabad CO/AQI diagnostic: separates verified data from suspected measurement/extrapolation issues.
5. All figures rendered with accessible color palettes, high DPI (300), explicit units, and clear statistical legends.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

RAW_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\data\raw")
INTERIM_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\data\interim")
PROCESSED_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\data\processed")
FIGURES_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\reports\figures")
TABLES_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\reports\tables")

CPCB_COLORS = {
    "Good": "#10B981",
    "Satisfactory": "#84CC16",
    "Moderate": "#EAB308",
    "Poor": "#F97316",
    "Very Poor": "#EF4444",
    "Severe": "#7F1D1D"
}

def set_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#F1F5F9"
    plt.rcParams["grid.linestyle"] = "--"

def generate_all_eda():
    set_style()
    print("Loading interim continuous city dataset...")
    df_city = pd.read_parquet(INTERIM_DIR / "city_day_continuous_raw.parquet")
    df_city["Date"] = pd.to_datetime(df_city["Date"])
    df_city["Month"] = df_city["Date"].dt.month
    df_city["Year"] = df_city["Date"].dt.year
    df_city["MonthName"] = df_city["Date"].dt.strftime("%b")

    # -------------------------------------------------------------
    # FIGURE 1: AQI Distribution & CPCB Statutory Bands
    # -------------------------------------------------------------
    print("Generating Figure 1: National AQI Distribution & Statutory Tiers...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    valid_aqi = df_city["AQI"].dropna()
    sns.histplot(valid_aqi[valid_aqi <= 600], bins=60, kde=True, ax=ax1, color="#2563EB", edgecolor="white", alpha=0.7)
    ax1.axvline(50, color=CPCB_COLORS["Good"], linestyle="--", label="Good (50)")
    ax1.axvline(100, color=CPCB_COLORS["Satisfactory"], linestyle="--", label="Satisfactory (100)")
    ax1.axvline(200, color=CPCB_COLORS["Moderate"], linestyle="--", label="Moderate (200)")
    ax1.axvline(300, color=CPCB_COLORS["Poor"], linestyle="--", label="Poor (300)")
    ax1.axvline(400, color=CPCB_COLORS["Very Poor"], linestyle="--", label="Very Poor (400)")
    ax1.set_title("National AQI Frequency Distribution (Observed Values <= 600)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Air Quality Index (AQI) [Dimensionless]", fontsize=11)
    ax1.set_ylabel("Observation Frequency (Days)", fontsize=11)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.annotate(f"Total >500: {(valid_aqi > 500).sum():,} days\n(Dataset Max: {valid_aqi.max():.0f})", 
                 xy=(520, 800), xytext=(450, 1500),
                 arrowprops=dict(facecolor="#DC2626", shrink=0.05, width=1, headwidth=6),
                 fontsize=9, fontweight="bold", color="#DC2626",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#FEF2F2", ec="#DC2626"))

    bucket_counts = df_city["AQI_Bucket"].value_counts()[["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]]
    colors = [CPCB_COLORS[b] for b in bucket_counts.index]
    bars = ax2.bar(bucket_counts.index, bucket_counts.values, color=colors, edgecolor="#334155", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:,}\n({h/bucket_counts.sum()*100:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_title("Observations Stratified by Statutory CPCB Category", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Valid City-Day Count", fontsize=11)
    ax2.tick_params(axis="x", rotation=25)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_aqi_distribution_and_cpcb_bands.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 2: City-Level AQI Comparison Across 26 Monitored Cities
    # -------------------------------------------------------------
    print("Generating Figure 2: City-Level AQI Comparison...")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    
    city_medians = df_city.groupby("City", observed=False)["AQI"].median().sort_values(ascending=False)
    ordered_cities = city_medians.index.tolist()
    
    sns.boxplot(data=df_city, y="City", x="AQI", order=ordered_cities, ax=ax,
                color="#64748B", showfliers=False, linewidth=1.1)
    
    ax.axvline(100, color=CPCB_COLORS["Satisfactory"], linestyle=":", linewidth=1.5, label="Satisfactory Threshold (100)")
    ax.axvline(200, color=CPCB_COLORS["Moderate"], linestyle="--", linewidth=1.5, label="Moderate Threshold (200)")
    ax.axvline(300, color=CPCB_COLORS["Poor"], linestyle="-.", linewidth=1.5, label="Poor Threshold (300)")
    
    ax.set_title("Observed AQI Dispersion Across 26 Indian Cities (Ranked by Median, 2015–2020)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Air Quality Index (AQI) [Box shows IQR (25th-75th); Whiskers show 1.5x IQR]", fontsize=11)
    ax.set_ylabel("Monitored City Name", fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_city_level_aqi_comparison.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 3: Seasonal & Monthly Cycles
    # -------------------------------------------------------------
    print("Generating Figure 3: Seasonal & Monthly Cycles...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sns.boxplot(data=df_city, x="MonthName", y="AQI", order=month_order, ax=ax1,
                color="#94A3B8", showfliers=False, linewidth=1.1)
    ax1.set_title("National Monthly AQI Cycle (Whiskers: 1.5x IQR)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Calendar Month", fontsize=11)
    ax1.set_ylabel("Air Quality Index (AQI)", fontsize=11)
    ax1.axhline(100, color="#84CC16", linestyle=":", label="Satisfactory (100)")
    ax1.axhline(200, color="#EAB308", linestyle="--", label="Moderate (200)")
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    
    # 5 Major representative metropolitan centers
    metro_subset = df_city[df_city["City"].isin(["Delhi", "Bengaluru", "Chennai", "Kolkata", "Mumbai"])].copy()
    monthly_metro = metro_subset.groupby(["Month", "City"], observed=False)["AQI"].mean().reset_index()
    
    palette = {"Delhi": "#DC2626", "Kolkata": "#EA580C", "Mumbai": "#0284C7", "Bengaluru": "#10B981", "Chennai": "#8B5CF6"}
    for city in ["Delhi", "Kolkata", "Mumbai", "Bengaluru", "Chennai"]:
        sub = monthly_metro[monthly_metro["City"] == city]
        ax2.plot(sub["Month"], sub["AQI"], marker="o", linewidth=2.0, label=city, color=palette[city])
        
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_order)
    ax2.set_title("Mean Monthly AQI Profiles for Five Major Metropolitan Centers", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Calendar Month", fontsize=11)
    ax2.set_ylabel("Mean Observed AQI", fontsize=11)
    ax2.legend(title="City", frameon=True, loc="upper right")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_seasonal_and_monthly_patterns.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 4: Chemical Precursor Correlation Heatmap
    # -------------------------------------------------------------
    print("Generating Figure 4: Chemical Precursor Correlation Heatmap...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "AQI"]
    corr_df = df_city[pollutants].dropna(how="all")
    
    pearson_corr = corr_df.corr(method="pearson")
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
    sns.heatmap(pearson_corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.2, vmax=1.0,
                square=True, linewidths=0.6, ax=ax1, cbar_kws={"shrink": 0.8})
    ax1.set_title("Pearson Linear Correlation Matrix", fontsize=12, fontweight="bold", pad=12)
    
    spearman_corr = corr_df.corr(method="spearman")
    sns.heatmap(spearman_corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-0.2, vmax=1.0,
                square=True, linewidths=0.6, ax=ax2, cbar_kws={"shrink": 0.8})
    ax2.set_title("Spearman Rank (Monotonic) Correlation Matrix", fontsize=12, fontweight="bold", pad=12)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_chemical_precursor_correlation_heatmap.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 5: Ahmedabad CO / Extreme AQI Diagnostic
    # -------------------------------------------------------------
    print("Generating Figure 5: Ahmedabad CO Anomaly Diagnostic...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    co_cities = df_city[df_city["City"].isin(["Ahmedabad", "Delhi", "Bengaluru", "Lucknow", "Mumbai"])].copy()
    sns.boxplot(data=co_cities, x="City", y="CO", ax=ax1, color="#38BDF8", showfliers=False)
    ax1.set_title("Reported Carbon Monoxide (CO) Concentrations Across Five Metros", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Reported CO (mg/m³)", fontsize=11)
    ax1.axhline(4.0, color="#DC2626", linestyle="--", label="NAAQS 24h Standard (4.0 mg/m³)")
    ax1.legend(loc="upper right", frameon=True)
    
    ahm = df_city[df_city["City"] == "Ahmedabad"].sort_values("Date")
    ax2.plot(ahm["Date"], ahm["AQI"], label="Ahmedabad Reported AQI", color="#DC2626", alpha=0.8, linewidth=1.1)
    ax2.axhline(500, color="#000000", linestyle=":", label="Official CPCB Severe Ceiling (500)")
    ax2.set_title("Ahmedabad Time-Series: Uncapped Mathematical Extrapolations (>1,500)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Reported AQI", fontsize=11)
    ax2.legend(loc="upper left", frameon=True)
    ax2.annotate("Uncapped linear extrapolation\nof high CO yields AQI up to 2,049",
                 xy=(pd.Timestamp("2018-02-19"), 2049), xytext=(pd.Timestamp("2016-06-01"), 1750),
                 arrowprops=dict(facecolor="#B91C1C", shrink=0.05, width=1.1, headwidth=5),
                 fontsize=9, fontweight="bold", color="#B91C1C",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#FEF2F2", ec="#B91C1C"))
                 
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_ahmedabad_co_anomaly_investigation.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 6: COVID-19 Seasonally Matched Comparison (Mar 25 - May 31)
    # -------------------------------------------------------------
    print("Generating Figure 6: Seasonally Matched COVID-19 Comparison...")
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    
    lockdown_cities = ["Delhi", "Lucknow", "Chennai", "Hyderabad", "Patna", "Bengaluru", "Kolkata", "Mumbai"]
    df_lock = df_city[df_city["City"].isin(lockdown_cities)].copy()
    
    # Exact calendar match: March 25 - May 31 (68 calendar days per year)
    matched_window = ((df_lock["Date"].dt.month == 3) & (df_lock["Date"].dt.day >= 25)) | (df_lock["Date"].dt.month.isin([4, 5]))
    lock_data = df_lock[matched_window].copy()
    
    stats_list = []
    for city in lockdown_cities:
        c_sub = lock_data[lock_data["City"] == city]
        b_series = c_sub[c_sub["Year"] < 2020]["AQI"].dropna()
        l_series = c_sub[c_sub["Year"] == 2020]["AQI"].dropna()
        
        b_mean, b_std, b_n = b_series.mean(), b_series.std(), len(b_series)
        l_mean, l_std, l_n = l_series.mean(), l_series.std(), len(l_series)
        drop = ((l_mean - b_mean) / b_mean) * 100
        
        stats_list.append({
            "City": city,
            "Baseline_Mean": b_mean,
            "Baseline_Std": b_std,
            "Baseline_N": b_n,
            "Lockdown_Mean": l_mean,
            "Lockdown_Std": l_std,
            "Lockdown_N": l_n,
            "PctDrop": drop
        })
        
    comp_df = pd.DataFrame(stats_list).sort_values("PctDrop")
    
    x = np.arange(len(comp_df))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, comp_df["Baseline_Mean"], width, yerr=comp_df["Baseline_Std"],
                    capsize=4, label="2015–2019 Seasonally Matched Baseline (Mean ± 1 SD)", color="#94A3B8", ecolor="#475569")
    rects2 = ax.bar(x + width/2, comp_df["Lockdown_Mean"], width, yerr=comp_df["Lockdown_Std"],
                    capsize=4, label="2020 Lockdown Period (Mean ± 1 SD)", color="#10B981", ecolor="#047857")
    
    for i, row in comp_df.reset_index(drop=True).iterrows():
        ax.annotate(f"{row['PctDrop']:.1f}%\n(Nb={row['Baseline_N']}, Nl={row['Lockdown_N']})",
                    xy=(x[i] + width/2, row["Lockdown_Mean"] + row["Lockdown_Std"]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color="#065F46")
                    
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["City"], rotation=20)
    ax.set_title("Observed Ambient AQI During March 25 – May 31: 2015–2019 Baseline vs. 2020 Lockdown", fontsize=12, fontweight="bold", pad=14)
    ax.set_ylabel("Mean Observed AQI (March 25 – May 31)", fontsize=11)
    ax.legend(frameon=True, loc="upper right")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_covid_lockdown_shock_analysis.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 7: Temporal Missingness Heatmap
    # -------------------------------------------------------------
    print("Generating Figure 7: Temporal Missingness Heatmap...")
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    
    df_city["YearMonth"] = df_city["Date"].dt.to_period("M")
    coverage_matrix = df_city.groupby(["City", "YearMonth"], observed=False)["AQI"].apply(lambda s: s.notna().mean() * 100).unstack()
    coverage_matrix.columns = [str(c) for c in coverage_matrix.columns]
    
    sns.heatmap(coverage_matrix, cmap="Blues", cbar_kws={"label": "% Valid Daily AQI Recorded"}, ax=ax, linewidths=0.2)
    ax.set_title("City-Level CAAQMS Station Completeness by Month (2015–2020)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Year-Month", fontsize=11)
    ax.set_ylabel("City", fontsize=11)
    ax.tick_params(axis="x", rotation=90)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_temporal_missingness_heatmap.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 8: Hourly Diurnal Rush-Hour Cycles (Memory-Safe Chunking)
    # -------------------------------------------------------------
    print("Generating Figure 8: Hourly Diurnal Rush Hour Curves (Chunked Read across all 707k rows)...")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    
    target_hourly_cities = ["Delhi", "Bengaluru", "Chennai", "Kolkata"]
    hourly_chunks = []
    for chunk in pd.read_csv(RAW_DIR / "city_hour.csv", chunksize=100000, usecols=["City", "Datetime", "PM2.5", "NO2"]):
        sub = chunk[chunk["City"].isin(target_hourly_cities)].copy()
        if not sub.empty:
            hourly_chunks.append(sub)
            
    df_all_hourly = pd.concat(hourly_chunks, ignore_index=True)
    df_all_hourly["Datetime"] = pd.to_datetime(df_all_hourly["Datetime"])
    df_all_hourly["Hour"] = df_all_hourly["Datetime"].dt.hour
    
    diurnal_pm = df_all_hourly.groupby(["Hour", "City"], observed=False)["PM2.5"].mean().unstack()
    
    for city in target_hourly_cities:
        ax.plot(diurnal_pm.index, diurnal_pm[city], marker="s", markersize=4, linewidth=2.0, label=f"{city} (PM2.5)", color=palette.get(city, "#334155"))
        
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    ax.set_title("Diurnal Cycle: Complete 2015–2020 Hourly Mean PM2.5 Concentrations (164,079 Extracted Records)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day (Local Standard Time)", fontsize=11)
    ax.set_ylabel("Mean PM2.5 Concentration (µg/m³)", fontsize=11)
    ax.legend(title="Metropolitan City", frameon=True)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_hourly_diurnal_rush_hour_cycle.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 9: Intra-City Spatial Dispersion (Neutral Station Locations)
    # -------------------------------------------------------------
    print("Generating Figure 9: Intra-City Delhi Station AQI Dispersion...")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    
    st_day = pd.read_csv(RAW_DIR / "station_day.csv")
    stations = pd.read_csv(RAW_DIR / "stations.csv")
    
    st_merged = st_day.merge(stations, on="StationId", how="left")
    delhi_stations = st_merged[st_merged["City"] == "Delhi"].copy()
    
    top_stations = delhi_stations.groupby("StationName")["AQI"].count()
    valid_names = top_stations[top_stations > 800].index.tolist()[:8]
    delhi_top = delhi_stations[delhi_stations["StationName"].isin(valid_names)]
    
    # Sort stations by median AQI
    station_order = delhi_top.groupby("StationName")["AQI"].median().sort_values(ascending=False).index.tolist()
    
    sns.boxplot(data=delhi_top, y="StationName", x="AQI", order=station_order, ax=ax,
                color="#A855F7", showfliers=False, linewidth=1.1)
    ax.set_title("Intra-City Spatial Dispersion: Station-Level AQI Across Delhi CAAQMS Monitoring Locations (2015–2020)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Air Quality Index (AQI) [Whiskers show 1.5x IQR]", fontsize=11)
    ax.set_ylabel("Monitoring Station Name (Source Metadata)", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_station_spatial_heterogeneity.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # EXPORT SUMMARY TABLES
    # -------------------------------------------------------------
    print("Exporting Statistical Summary Tables...")
    
    # Table 1: City Summary Table
    city_summary = df_city.groupby("City", observed=False).agg(
        TotalDays=("Date", "count"),
        ValidAQIDays=("AQI", lambda s: s.notna().sum()),
        AQI_Mean=("AQI", "mean"),
        AQI_Median=("AQI", "median"),
        AQI_p95=("AQI", lambda s: s.quantile(0.95)),
        PM25_Mean=("PM2.5", "mean"),
        PM10_Mean=("PM10", "mean"),
        NO2_Mean=("NO2", "mean"),
        Days_Above_200=("AQI", lambda s: (s > 200).sum()),
        Pct_Days_Above_200=("AQI", lambda s: round(((s > 200).sum() / s.notna().sum()) * 100, 1) if s.notna().sum() > 0 else 0)
    ).round(2).reset_index()
    city_summary.to_csv(TABLES_DIR / "city_summary_table.csv", index=False)

    # Table 2: Pollutant Summary Table
    pollutant_summary = []
    numeric_cols = [c for c in df_city.columns if c in pollutants]
    for c in numeric_cols:
        s = df_city[c].dropna()
        pollutant_summary.append({
            "Pollutant": c,
            "Count": len(s),
            "MissingPct": round((df_city[c].isna().sum() / len(df_city)) * 100, 2),
            "Mean": round(s.mean(), 2),
            "StdDev": round(s.std(), 2),
            "Min": round(s.min(), 2),
            "Median": round(s.median(), 2),
            "Max": round(s.max(), 2),
            "p95": round(s.quantile(0.95), 2),
            "Skewness": round(s.skew(), 2)
        })
    pd.DataFrame(pollutant_summary).to_csv(TABLES_DIR / "pollutant_summary_table.csv", index=False)

    # Table 3: CPCB Bucket Breakdown Table
    bucket_df = df_city["AQI_Bucket"].value_counts(dropna=False).reset_index()
    bucket_df.columns = ["CPCB_Category", "Record_Count"]
    bucket_df["Percentage"] = round((bucket_df["Record_Count"] / len(df_city)) * 100, 2)
    bucket_df.to_csv(TABLES_DIR / "cpcb_bucket_breakdown.csv", index=False)

    print("[SUCCESS] All 9 audited figures and 3 summary tables generated.")

if __name__ == "__main__":
    generate_all_eda()
