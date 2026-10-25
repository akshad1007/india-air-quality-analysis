"""
Automated Data Validation and Integrity Tests.
Tests:
1. File existence and non-zero byte size.
2. SHA-256 checksum integrity against recorded registry.
3. Primary and composite key uniqueness across all datasets.
4. Foreign key integrity between station_day.csv and stations.csv.
5. Non-negative concentration boundaries.
6. Statutory CPCB AQI Bucket mapping correctness.
"""

import hashlib
from pathlib import Path
import pandas as pd
import pytest

RAW_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\data\raw")

EXPECTED_CHECKSUMS = {
    "stations.csv": "ce2b7081c3e3c89145613eb724d7d2149ef79c6e16f36bb8a835747a673ed410",
    "city_day.csv": "0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847",
    "city_hour.csv": "fae474062ec6b10af7c5b2ac4e32f5c240a0ef80835ee3a50332fec06e2190a4",
    "station_day.csv": "f07787eb4c36c53b07763f591ee29fa01fbaa4d18df1a470447d04def79677b9",
    "station_hour.csv": "26a423c66fe9763064d580ffb007150375cfa2056e62b29d52b1b7857af54fde"
}

def compute_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()

def test_raw_files_exist_and_non_empty():
    for filename in EXPECTED_CHECKSUMS.keys():
        filepath = RAW_DIR / filename
        assert filepath.exists(), f"Missing file: {filename}"
        assert filepath.stat().st_size > 0, f"Empty file: {filename}"

def test_sha256_checksums():
    for filename, expected_hash in EXPECTED_CHECKSUMS.items():
        filepath = RAW_DIR / filename
        actual_hash = compute_sha256(filepath)
        assert actual_hash == expected_hash, f"Checksum mismatch for {filename}: expected {expected_hash}, got {actual_hash}"

def test_stations_primary_key_uniqueness():
    df = pd.read_csv(RAW_DIR / "stations.csv")
    assert df["StationId"].is_unique, "StationId is not unique in stations.csv"
    assert len(df) == 230, f"Expected 230 stations, found {len(df)}"

def test_city_day_composite_key_uniqueness():
    df = pd.read_csv(RAW_DIR / "city_day.csv")
    duplicates = df.duplicated(subset=["City", "Date"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (City, Date) rows in city_day.csv"

def test_station_day_foreign_key_integrity():
    stations = pd.read_csv(RAW_DIR / "stations.csv")
    station_day = pd.read_csv(RAW_DIR / "station_day.csv")
    
    station_ids = set(stations["StationId"])
    day_ids = set(station_day["StationId"].unique())
    
    unmatched = day_ids - station_ids
    assert len(unmatched) == 0, f"Found unmatched station IDs in station_day.csv: {unmatched}"

def test_non_negative_pollutant_concentrations():
    df = pd.read_csv(RAW_DIR / "city_day.csv")
    numeric_cols = [c for c in df.columns if c not in ["City", "Date", "AQI_Bucket"]]
    
    for col in numeric_cols:
        negatives = (df[col].dropna() < 0).sum()
        assert negatives == 0, f"Column {col} contains {negatives} negative values"

def test_cpcb_aqi_bucket_boundaries():
    df = pd.read_csv(RAW_DIR / "city_day.csv")
    valid = df.dropna(subset=["AQI", "AQI_Bucket"])
    
    good = valid[valid["AQI_Bucket"] == "Good"]["AQI"]
    assert good.min() >= 0 and good.max() <= 50, f"Good bucket out of bounds: [{good.min()}, {good.max()}]"
    
    sat = valid[valid["AQI_Bucket"] == "Satisfactory"]["AQI"]
    assert sat.min() >= 51 and sat.max() <= 100, f"Satisfactory bucket out of bounds: [{sat.min()}, {sat.max()}]"
    
    mod = valid[valid["AQI_Bucket"] == "Moderate"]["AQI"]
    assert mod.min() >= 101 and mod.max() <= 200, f"Moderate bucket out of bounds: [{mod.min()}, {mod.max()}]"
    
    poor = valid[valid["AQI_Bucket"] == "Poor"]["AQI"]
    assert poor.min() >= 201 and poor.max() <= 300, f"Poor bucket out of bounds: [{poor.min()}, {poor.max()}]"
    
    vpoor = valid[valid["AQI_Bucket"] == "Very Poor"]["AQI"]
    assert vpoor.min() >= 301 and vpoor.max() <= 400, f"Very Poor bucket out of bounds: [{vpoor.min()}, {vpoor.max()}]"
    
    severe = valid[valid["AQI_Bucket"] == "Severe"]["AQI"]
    assert severe.min() >= 401, f"Severe bucket min out of bounds: {severe.min()}"


def test_faculty_guidelines_exist():
    """Verify that both original faculty PDF documents and README exist in docs/reference/faculty-guidelines/."""
    guide_dir = Path("docs/reference/faculty-guidelines")
    pdf1 = guide_dir / "Assignment_4144_Content_Document_20261019020930913PM.pdf"
    pdf2 = guide_dir / "Assignment_4144_Content_SolutionDocument_20261019020930913PM (1).pdf"
    readme = guide_dir / "README.md"
    
    assert pdf1.exists(), f"Missing faculty PDF 1: {pdf1}"
    assert pdf2.exists(), f"Missing faculty PDF 2: {pdf2}"
    assert readme.exists(), f"Missing faculty guidelines README: {readme}"
    
    assert pdf1.stat().st_size > 20000, f"Faculty PDF 1 size too small: {pdf1.stat().st_size}"
    assert pdf2.stat().st_size > 40000, f"Faculty PDF 2 size too small: {pdf2.stat().st_size}"
    
    # Verify PDF magic bytes (%PDF)
    with open(pdf1, "rb") as f:
        assert f.read(4) == b"%PDF", f"{pdf1} is not a valid PDF file"
    with open(pdf2, "rb") as f:
        assert f.read(4) == b"%PDF", f"{pdf2} is not a valid PDF file"

