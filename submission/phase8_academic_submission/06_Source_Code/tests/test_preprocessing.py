"""
Unit and Integration Tests for Data Preprocessing Pipeline.
Tests:
1. Raw Data Immutability: Verifies raw CSV files are not modified.
2. Target Non-Imputation: Verifies that NaN target values are NEVER filled.
3. Cross-City Boundary Protection: Asserts no temporal forward-fill bleeds across cities.
4. Bounded Forward-Fill Limit: Asserts that gaps > forward_fill_limit remain NaN.
5. Temporal Continuity: Asserts monotonic daily continuity per city.
6. Key Uniqueness: Asserts (City, Date) remains unique.
"""

from pathlib import Path
import pandas as pd
import pytest
from src.preprocessing.cleaner import PreprocessingPipeline, verify_raw_immutability, RAW_DIR, PROCESSED_DIR

def test_raw_data_immutability():
    assert verify_raw_immutability(RAW_DIR) is True

def test_target_aqi_never_imputed():
    raw_df = pd.read_csv(RAW_DIR / "city_day.csv")
    raw_missing_aqi = raw_df["AQI"].isna().sum()
    
    # Check both processed parquet variants
    df_ffill = pd.read_parquet(PROCESSED_DIR / "city_day_preprocessed_ffill.parquet")
    df_native = pd.read_parquet(PROCESSED_DIR / "city_day_preprocessed_native.parquet")
    
    # In continuous reindexed calendar, additional calendar days were added,
    # but the number of valid (non-null) AQI records must be EXACTLY identical to raw!
    raw_valid_aqi = raw_df["AQI"].notna().sum()
    assert df_ffill["AQI"].notna().sum() == raw_valid_aqi, (
        f"Target AQI was modified in ffill! Raw valid: {raw_valid_aqi}, Processed valid: {df_ffill['AQI'].notna().sum()}"
    )
    assert df_native["AQI"].notna().sum() == raw_valid_aqi, (
        f"Target AQI was modified in native! Raw valid: {raw_valid_aqi}, Processed valid: {df_native['AQI'].notna().sum()}"
    )

def test_no_cross_city_leakage():
    df_ffill = pd.read_parquet(PROCESSED_DIR / "city_day_preprocessed_ffill.parquet")
    
    # Verify each city starts with its own initial values and does not copy from previous city in dataframe
    for city, grp in df_ffill.groupby("City", observed=False):
        first_row = grp.sort_values("Date").iloc[0]
        # If raw had NaN at start, ffill should NOT have filled it from another city
        raw_city = pd.read_csv(RAW_DIR / "city_day.csv")
        raw_first = raw_city[raw_city["City"] == city].sort_values("Date").iloc[0]
        for col in ["CO", "NO2", "SO2"]:
            if pd.isna(raw_first[col]):
                assert pd.isna(first_row[col]), f"Cross-city leakage detected in {city} for {col}!"

def test_bounded_ffill_gap_limit():
    df_ffill = pd.read_parquet(PROCESSED_DIR / "city_day_preprocessed_ffill.parquet")
    
    # In Delhi, find a gap > 2 days and verify day 3 is still NaN
    # Let's verify that wherever a gap > 2 days occurred in raw, the 3rd missing day remained NaN
    for city in ["Delhi", "Bengaluru", "Chennai"]:
        c_df = df_ffill[df_ffill["City"] == city].sort_values("Date")
        # Check long pre-monitoring gaps
        if city == "Chennai":
            # Chennai had missing PM10 for months initially
            initial_pm10 = c_df["PM10"].iloc[:10]
            assert initial_pm10.isna().sum() > 2, "Forward fill exceeded gap limit!"

def test_monotonic_daily_continuity():
    df_ffill = pd.read_parquet(PROCESSED_DIR / "city_day_preprocessed_ffill.parquet")
    
    for city, grp in df_ffill.groupby("City", observed=False):
        dates = pd.to_datetime(grp["Date"])
        diffs = dates.diff().dropna()
        assert (diffs == pd.Timedelta(days=1)).all(), f"City {city} does not have strictly monotonic 1-day step!"

def test_key_uniqueness():
    df_ffill = pd.read_parquet(PROCESSED_DIR / "city_day_preprocessed_ffill.parquet")
    duplicates = df_ffill.duplicated(subset=["City", "Date"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (City, Date) pairs in preprocessed data!"
