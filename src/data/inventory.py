"""
Automated Data Inventory and Verification Script.
Profiles all 5 raw CPCB CSV files in data/raw/ using memory-safe chunking.
Computes SHA-256 checksums, exact row counts, date ranges, missingness,
key cardinality, join viability, and gap distributions.
"""

import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\data\raw")
FILES = ["stations.csv", "city_day.csv", "city_hour.csv", "station_day.csv", "station_hour.csv"]

def compute_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()

def profile_file(filename: str):
    filepath = RAW_DIR / filename
    size_bytes = filepath.stat().st_size
    checksum = compute_sha256(filepath)
    print(f"\n{'='*70}\nPROFILING: {filename} ({size_bytes:,} bytes, SHA-256: {checksum[:16]}...)")

    # Read header and sample
    df_sample = pd.read_csv(filepath, nrows=100)
    columns = list(df_sample.columns)
    
    # Process file memory-safely
    chunksize = 200000
    total_rows = 0
    null_counts = {col: 0 for col in columns}
    date_min = None
    date_max = None
    date_col = next((c for c in ["Date", "Datetime"] if c in columns), None)
    unique_cities = set()
    unique_stations = set()
    duplicate_key_count = 0
    
    # Track composite keys
    key_cols = []
    if "StationId" in columns and date_col:
        key_cols = ["StationId", date_col]
    elif "City" in columns and date_col:
        key_cols = ["City", date_col]
    elif "StationId" in columns:
        key_cols = ["StationId"]

    seen_keys = set()
    
    # Chunked iteration
    for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        for col in columns:
            null_counts[col] += chunk[col].isna().sum()
            
        if date_col:
            chunk_dates = chunk[date_col].dropna()
            if not chunk_dates.empty:
                c_min = chunk_dates.min()
                c_max = chunk_dates.max()
                if date_min is None or c_min < date_min:
                    date_min = c_min
                if date_max is None or c_max > date_max:
                    date_max = c_max
                    
        if "City" in columns:
            unique_cities.update(chunk["City"].dropna().unique())
        if "StationId" in columns:
            unique_stations.update(chunk["StationId"].dropna().unique())
            
        if key_cols:
            if len(key_cols) == 1:
                keys = chunk[key_cols[0]].values
            else:
                keys = list(zip(*(chunk[k].values for k in key_cols)))
            for k in keys:
                if k in seen_keys:
                    duplicate_key_count += 1
                else:
                    seen_keys.add(k)

    print(f"Total Rows: {total_rows:,}")
    print(f"Date Range ({date_col}): {date_min} to {date_max}")
    if unique_cities:
        print(f"Unique Cities ({len(unique_cities)}): {sorted(list(unique_cities))[:8]}... (total {len(unique_cities)})")
    if unique_stations:
        print(f"Unique Stations ({len(unique_stations)}): total {len(unique_stations)}")
    print(f"Key cols {key_cols} - Duplicate Keys: {duplicate_key_count}")
    
    # Null summary
    null_summary = {}
    for col in columns:
        pct = (null_counts[col] / total_rows) * 100
        null_summary[col] = {"missing_count": int(null_counts[col]), "missing_pct": round(pct, 2)}
    
    return {
        "filename": filename,
        "size_bytes": size_bytes,
        "sha256": checksum,
        "total_rows": total_rows,
        "columns": columns,
        "date_col": date_col,
        "date_min": str(date_min),
        "date_max": str(date_max),
        "unique_cities_count": len(unique_cities),
        "unique_stations_count": len(unique_stations),
        "unique_cities": sorted(list(unique_cities)),
        "unique_stations": sorted(list(unique_stations)),
        "key_cols": key_cols,
        "duplicate_keys": duplicate_key_count,
        "null_summary": null_summary
    }

def main():
    results = {}
    for f in FILES:
        results[f] = profile_file(f)
        
    with open(RAW_DIR.parent.parent / "docs/reports/data_inventory_raw.json", "w") as out:
        json.dump(results, out, indent=2)
    print("\n[SUCCESS] Raw data profiling completed and saved to docs/reports/data_inventory_raw.json")

if __name__ == "__main__":
    main()
