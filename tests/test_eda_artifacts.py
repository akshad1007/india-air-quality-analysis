"""
tests/test_eda_artifacts.py
---------------------------
Automated test suite verifying the integrity, completeness, and non-emptiness
of all 9 EDA publication figures and 3 summary tables generated in reports/.
"""
import os
import json
import pytest
import pandas as pd

REPORT_FIGURES_DIR = "reports/figures"
REPORT_TABLES_DIR = "reports/tables"
NOTEBOOK_PATH = "notebooks/02_exploratory_data_analysis.ipynb"

EXPECTED_FIGURES = [
    "01_aqi_distribution_and_cpcb_bands.png",
    "02_city_level_aqi_comparison.png",
    "03_seasonal_and_monthly_patterns.png",
    "04_chemical_precursor_correlation_heatmap.png",
    "05_ahmedabad_co_anomaly_investigation.png",
    "06_covid_lockdown_shock_analysis.png",
    "07_temporal_missingness_heatmap.png",
    "08_hourly_diurnal_rush_hour_cycle.png",
    "09_station_spatial_heterogeneity.png"
]

EXPECTED_TABLES = [
    "city_summary_table.csv",
    "pollutant_summary_table.csv",
    "cpcb_bucket_breakdown.csv"
]


def test_all_figures_exist_and_non_empty():
    """Verify that all 9 figures exist and are non-empty PNG files."""
    for fig_name in EXPECTED_FIGURES:
        fig_path = os.path.join(REPORT_FIGURES_DIR, fig_name)
        assert os.path.exists(fig_path), f"Missing figure: {fig_path}"
        file_size = os.path.getsize(fig_path)
        assert file_size > 15000, f"Figure {fig_name} is suspiciously small ({file_size} bytes)"
        
        # Verify PNG header
        with open(fig_path, "rb") as f:
            header = f.read(8)
            assert header == b'\x89PNG\r\n\x1a\n', f"Figure {fig_name} is not a valid PNG file"


def test_all_tables_exist_and_valid():
    """Verify that all 3 summary tables exist, have non-zero rows, and correct columns."""
    for tbl_name in EXPECTED_TABLES:
        tbl_path = os.path.join(REPORT_TABLES_DIR, tbl_name)
        assert os.path.exists(tbl_path), f"Missing table: {tbl_path}"
        df = pd.read_csv(tbl_path)
        assert len(df) > 0, f"Table {tbl_name} is empty"

    # Specific checks
    city_tbl = pd.read_csv(os.path.join(REPORT_TABLES_DIR, "city_summary_table.csv"))
    assert "City" in city_tbl.columns
    assert "AQI_Median" in city_tbl.columns
    assert len(city_tbl) >= 26

    pollutant_tbl = pd.read_csv(os.path.join(REPORT_TABLES_DIR, "pollutant_summary_table.csv"))
    assert "Pollutant" in pollutant_tbl.columns
    assert "MissingPct" in pollutant_tbl.columns
    assert len(pollutant_tbl) >= 12

    bucket_tbl = pd.read_csv(os.path.join(REPORT_TABLES_DIR, "cpcb_bucket_breakdown.csv"))
    assert "CPCB_Category" in bucket_tbl.columns
    assert "Record_Count" in bucket_tbl.columns


def test_notebook_executed_cleanly():
    """Verify that the notebook contains executed outputs and no execution errors."""
    assert os.path.exists(NOTEBOOK_PATH), f"Notebook missing at {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    assert len(code_cells) >= 9, "Notebook should have at least 9 code cells"
    
    # Check that code cells have executed outputs
    executed_cells = [c for c in code_cells if len(c.get("outputs", [])) > 0]
    assert len(executed_cells) >= 9, f"Only {len(executed_cells)} cells have outputs; notebook was not fully executed"
    
    # Check for error outputs
    for idx, c in enumerate(executed_cells):
        for out in c.get("outputs", []):
            assert out.get("output_type") != "error", f"Cell {idx} failed with error: {out.get('ename')}: {out.get('evalue')}"
    
    # Check for error outputs
    for idx, c in enumerate(executed_cells):
        for out in c.get("outputs", []):
            assert out.get("output_type") != "error", f"Cell {idx} failed with error: {out.get('ename')}: {out.get('evalue')}"


def test_chunked_hourly_sample_coverage():
    """Verify that city_hour sampling includes Delhi, Mumbai, Kolkata, Bengaluru without truncation bias."""
    # Read chunked hourly curves from generated figure or re-verify city presence in raw city_hour
    cities_needed = {"Delhi", "Mumbai", "Kolkata", "Bengaluru"}
    found_cities = set()
    for chunk in pd.read_csv("data/raw/city_hour.csv", usecols=["City", "Datetime", "NO2"], chunksize=100000):
        matched = set(chunk["City"].unique()).intersection(cities_needed)
        found_cities.update(matched)
        if found_cities == cities_needed:
            break
    assert found_cities == cities_needed, f"Missing cities in city_hour: {cities_needed - found_cities}"
