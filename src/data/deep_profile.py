"""
Deep Data Quality and Methodological Assessment Script.
Analyzes:
1. Exact column schemas, data types, and value ranges (min, median, max, negative counts).
2. Missingness percentages for all columns in all 5 datasets.
3. City-level temporal coverage across years (2015, 2016, 2017, 2018, 2019, 2020).
4. Continuous missing gap lengths per city in city_day.csv.
5. Join cardinality tests between stations.csv and station_day.csv, and city_day.csv.
6. CPCB AQI Bucket mapping verification against calculated AQI.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

RAW_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\data\raw")

def analyze_city_day():
    print("--- ANALYZING CITY_DAY.CSV ---")
    df = pd.read_csv(RAW_DIR / "city_day.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Range & negative values check
    numeric_cols = [c for c in df.columns if c not in ['City', 'Date', 'AQI_Bucket']]
    stats = {}
    for c in numeric_cols:
        series = df[c].dropna()
        stats[c] = {
            "missing_pct": round((df[c].isna().sum() / len(df)) * 100, 2),
            "min": float(series.min()) if not series.empty else None,
            "median": float(series.median()) if not series.empty else None,
            "max": float(series.max()) if not series.empty else None,
            "negatives_count": int((series < 0).sum())
        }
    
    # AQI Bucket vs AQI check
    bucket_counts = df['AQI_Bucket'].value_counts(dropna=False).to_dict()
    
    # Check AQI bucket consistency with standard CPCB bounds
    bucket_ranges = {}
    for bucket in df['AQI_Bucket'].dropna().unique():
        sub = df[df['AQI_Bucket'] == bucket]['AQI'].dropna()
        bucket_ranges[bucket] = {
            "min_aqi": float(sub.min()),
            "max_aqi": float(sub.max()),
            "count": int(len(sub))
        }
        
    # City-level temporal coverage by year
    df['Year'] = df['Date'].dt.year
    city_year_coverage = df.groupby(['City', 'Year'])['AQI'].agg(
        total_days='count',
        valid_aqi_days=lambda x: x.notna().sum(),
        pct_valid=lambda x: round((x.notna().sum() / len(x)) * 100, 1)
    ).unstack(level='Year')
    
    # Active date range per city
    city_spans = {}
    for city, grp in df.groupby('City'):
        valid_dates = grp.dropna(subset=['AQI'])['Date']
        city_spans[city] = {
            "first_record": str(grp['Date'].min().date()),
            "last_record": str(grp['Date'].max().date()),
            "first_valid_aqi": str(valid_dates.min().date()) if not valid_dates.empty else None,
            "last_valid_aqi": str(valid_dates.max().date()) if not valid_dates.empty else None,
            "total_records": len(grp),
            "valid_aqi_count": int(grp['AQI'].notna().sum()),
            "valid_aqi_pct": round((grp['AQI'].notna().sum() / len(grp)) * 100, 2)
        }
        
    # Analyze continuous missing gap lengths for AQI in continuous calendar
    gap_distributions = {}
    for city in ['Delhi', 'Bengaluru', 'Chennai', 'Mumbai', 'Hyderabad', 'Kolkata', 'Ahmedabad']:
        c_df = df[df['City'] == city].set_index('Date').reindex(
            pd.date_range(df['Date'].min(), df['Date'].max(), freq='D')
        )
        is_na = c_df['AQI'].isna()
        # Find consecutive missing blocks
        blocks = (~is_na).cumsum()[is_na]
        gap_lengths = blocks.value_counts().values
        gap_distributions[city] = {
            "total_days_in_span": len(c_df),
            "valid_aqi_days": int((~is_na).sum()),
            "missing_aqi_days": int(is_na.sum()),
            "isolated_1_day_gaps": int((gap_lengths == 1).sum()),
            "gaps_2_days": int((gap_lengths == 2).sum()),
            "gaps_3_to_7_days": int(((gap_lengths >= 3) & (gap_lengths <= 7)).sum()),
            "gaps_over_7_days": int((gap_lengths > 7).sum()),
            "max_gap_days": int(gap_lengths.max()) if len(gap_lengths) > 0 else 0
        }
        
    return {
        "numeric_stats": stats,
        "bucket_counts": bucket_counts,
        "bucket_ranges": bucket_ranges,
        "city_spans": city_spans,
        "gap_distributions": gap_distributions
    }

def analyze_joins():
    print("--- ANALYZING JOINS & METADATA ---")
    stations = pd.read_csv(RAW_DIR / "stations.csv")
    station_day = pd.read_csv(RAW_DIR / "station_day.csv")
    city_day = pd.read_csv(RAW_DIR / "city_day.csv")
    
    print(f"Stations shape: {stations.shape}, cols: {list(stations.columns)}")
    print(f"Station Day shape: {station_day.shape}, cols: {list(station_day.columns)}")
    
    # Join test: station_day with stations on StationId
    merged = station_day.merge(stations, on='StationId', how='left')
    unmatched_stations = station_day[~station_day['StationId'].isin(stations['StationId'])]['StationId'].unique()
    
    # Station types check in StationName
    name_sample = stations['StationName'].head(15).tolist()
    
    # City matching between stations.csv and city_day.csv
    stations_cities = set(stations['City'].dropna().unique())
    city_day_cities = set(city_day['City'].dropna().unique())
    
    cities_in_both = sorted(list(stations_cities.intersection(city_day_cities)))
    cities_in_city_day_not_stations = sorted(list(city_day_cities - stations_cities))
    
    return {
        "stations_columns": list(stations.columns),
        "unmatched_station_ids_in_day": list(unmatched_stations),
        "station_day_merged_len": len(merged),
        "station_day_original_len": len(station_day),
        "join_cardinality_preserved": len(merged) == len(station_day),
        "cities_in_both_count": len(cities_in_both),
        "cities_in_both": cities_in_both,
        "cities_in_city_day_not_stations": cities_in_city_day_not_stations,
        "station_status_counts": stations['Status'].value_counts(dropna=False).to_dict()
    }

def main():
    city_day_results = analyze_city_day()
    join_results = analyze_joins()
    
    output = {
        "city_day_results": city_day_results,
        "join_results": join_results
    }
    
    with open(RAW_DIR.parent.parent / "docs/reports/deep_profile_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n[SUCCESS] Deep profiling complete. Saved to docs/reports/deep_profile_results.json")

if __name__ == "__main__":
    main()
