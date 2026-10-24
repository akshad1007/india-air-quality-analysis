"""
Modular Preprocessing & Cleaning Pipeline for India Air Quality Intelligence.

Key Design Principles:
1. Raw Data Immutability: Verifies SHA-256 checksums before and after processing.
2. Target Integrity: Strictly NEVER imputes missing target AQI values.
3. Separation of Concerns: Distinguishes missing predictors from missing labels.
4. City Boundary Protection: Grouping by City guarantees zero cross-city temporal spillover.
5. Configurable Gap Imputation: Bounded forward-fill (default: 2 days) applied only to features,
   never across gaps exceeding the threshold.
6. Explicit Audit Logging: Logs row counts, transformations, and exclusions.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\data\raw")
INTERIM_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\data\interim")
PROCESSED_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\data\processed")

RAW_CHECKSUMS = {
    "stations.csv": "ce2b7081c3e3c89145613eb724d7d2149ef79c6e16f36bb8a835747a673ed410",
    "city_day.csv": "0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847",
    "city_hour.csv": "fae474062ec6b10af7c5b2ac4e32f5c240a0ef80835ee3a50332fec06e2190a4",
    "station_day.csv": "f07787eb4c36c53b07763f591ee29fa01fbaa4d18df1a470447d04def79677b9",
    "station_hour.csv": "26a423c66fe9763064d580ffb007150375cfa2056e62b29d52b1b7857af54fde"
}

def verify_raw_immutability(raw_dir: Path = RAW_DIR) -> bool:
    """Verifies that all 5 raw CSV files in data/raw/ have unaltered SHA-256 hashes."""
    for filename, expected_hash in RAW_CHECKSUMS.items():
        filepath = raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Raw file missing: {filepath}")
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192 * 1024):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Raw data mutation detected in {filename}! Expected {expected_hash}, found {actual_hash}")
    return True

class PreprocessingPipeline:
    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        interim_dir: Path = INTERIM_DIR,
        processed_dir: Path = PROCESSED_DIR,
        forward_fill_limit: int = 2,
        min_active_days: int = 365
    ):
        self.raw_dir = raw_dir
        self.interim_dir = interim_dir
        self.processed_dir = processed_dir
        self.forward_fill_limit = forward_fill_limit
        self.min_active_days = min_active_days
        
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log: Dict = {}

    def load_and_standardize_city_day(self) -> pd.DataFrame:
        """Loads city_day.csv, parses dates, verifies zero negatives, and validates types."""
        verify_raw_immutability(self.raw_dir)
        
        filepath = self.raw_dir / "city_day.csv"
        df = pd.read_csv(filepath)
        initial_rows = len(df)
        
        df["Date"] = pd.to_datetime(df["Date"])
        df["City"] = df["City"].astype("category")
        
        # Verify negative values
        pollutant_cols = [c for c in df.columns if c not in ["City", "Date", "AQI", "AQI_Bucket"]]
        for col in pollutant_cols + ["AQI"]:
            neg_count = (df[col].dropna() < 0).sum()
            if neg_count > 0:
                raise ValueError(f"Negative values detected in {col}: {neg_count} records.")

        self.audit_log["raw_city_day"] = {
            "initial_rows": initial_rows,
            "unique_cities": df["City"].nunique(),
            "date_min": str(df["Date"].min().date()),
            "date_max": str(df["Date"].max().date()),
            "missing_aqi_labels": int(df["AQI"].isna().sum())
        }
        return df

    def enforce_continuous_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reindexes each city against a continuous daily calendar from its first recorded
        date to the global dataset end date (2020-07-01).
        This guarantees strict temporal continuity without gaps.
        """
        global_max_date = df["Date"].max()
        continuous_dfs = []
        
        for city, grp in df.groupby("City", observed=False):
            min_date = grp["Date"].min()
            full_idx = pd.date_range(min_date, global_max_date, freq="D", name="Date")
            
            c_df = grp.set_index("Date").reindex(full_idx).reset_index()
            c_df["City"] = city
            continuous_dfs.append(c_df)
            
        res = pd.concat(continuous_dfs, ignore_index=True)
        
        self.audit_log["continuous_calendar"] = {
            "reindexed_total_rows": len(res),
            "added_calendar_days": len(res) - len(df)
        }
        return res

    def apply_predictor_imputation(self, df: pd.DataFrame, strategy: str = "bounded_ffill") -> pd.DataFrame:
        """
        Imputes missing PREDICTOR values strictly per-city.
        CRITICAL RULE: The target 'AQI' and 'AQI_Bucket' columns are NEVER imputed.
        """
        pollutant_cols = [c for c in df.columns if c not in ["City", "Date", "AQI", "AQI_Bucket"]]
        df_imputed = df.copy()
        
        pre_missing = {col: int(df[col].isna().sum()) for col in pollutant_cols}
        
        if strategy == "bounded_ffill":
            # Group strictly by city to prevent cross-city leakage
            # Apply forward fill with limit
            imputed_parts = []
            for city, grp in df_imputed.groupby("City", observed=False):
                grp = grp.sort_values("Date").copy()
                grp[pollutant_cols] = grp[pollutant_cols].ffill(limit=self.forward_fill_limit)
                imputed_parts.append(grp)
            df_imputed = pd.concat(imputed_parts, ignore_index=True)
            
        elif strategy == "none":
            # Pass through without imputation (for native missing support algorithms)
            pass
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")
            
        post_missing = {col: int(df_imputed[col].isna().sum()) for col in pollutant_cols}
        
        # Verify target AQI was NOT imputed
        assert df["AQI"].isna().sum() == df_imputed["AQI"].isna().sum(), "Target AQI was modified during imputation!"
        
        self.audit_log["imputation"] = {
            "strategy": strategy,
            "forward_fill_limit_days": self.forward_fill_limit if strategy == "bounded_ffill" else 0,
            "pre_imputation_missing": pre_missing,
            "post_imputation_missing": post_missing
        }
        return df_imputed

    def compute_city_eligibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assesses city usability across:
        1. Full EDA eligibility (all cities eligible).
        2. Modeling eligibility based on active observations in 2018 (train/val) and 2019 (test).
        """
        eligibility = []
        for city, grp in df.groupby("City", observed=False):
            valid_aqi = grp["AQI"].dropna()
            total_days = len(grp)
            valid_days = len(valid_aqi)
            
            # Check 2018 coverage (pre-test)
            days_2018 = grp[(grp["Date"] >= "2018-01-01") & (grp["Date"] <= "2018-12-31")]
            valid_2018 = days_2018["AQI"].notna().sum()
            
            # Check 2019 coverage (test set)
            days_2019 = grp[(grp["Date"] >= "2019-01-01") & (grp["Date"] <= "2019-12-31")]
            valid_2019 = days_2019["AQI"].notna().sum()
            
            # Eligibility rules:
            # Full Modeling: >= 200 valid days in 2018 AND >= 300 valid days in 2019
            # Partial / Secondary Modeling: >= 250 valid days in 2019
            # EDA Only: < 200 days in 2018 or < 250 days in 2019
            if valid_2018 >= 200 and valid_2019 >= 300:
                tier = "Primary Modeling Qualified (Tier 1)"
            elif valid_2019 >= 250:
                tier = "Secondary Test Qualified (Tier 2)"
            else:
                tier = "EDA Only (Insufficient Historical Depth)"
                
            eligibility.append({
                "City": city,
                "TotalRecords": total_days,
                "TotalValidAQI": valid_days,
                "ValidAQI_2018": int(valid_2018),
                "ValidAQI_2019": int(valid_2019),
                "CoverageTier": tier
            })
            
        el_df = pd.DataFrame(eligibility).sort_values("ValidAQI_2019", ascending=False)
        self.audit_log["city_eligibility"] = el_df.to_dict(orient="records")
        return el_df

    def run_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Executes full preprocessing and saves artifacts."""
        logger.info("Starting Preprocessing Pipeline...")
        
        # 1. Standardize and load
        df_raw = self.load_and_standardize_city_day()
        
        # 2. Continuous calendar
        df_continuous = self.enforce_continuous_calendar(df_raw)
        
        # 3. Assess city eligibility
        eligibility_df = self.compute_city_eligibility(df_continuous)
        
        # 4. Generate two processed variants:
        # Variant A: Bounded forward fill (for baseline and linear models)
        df_ffill = self.apply_predictor_imputation(df_continuous, strategy="bounded_ffill")
        
        # Variant B: No imputation (retains NaNs for native gradient boosting tree support)
        df_native = self.apply_predictor_imputation(df_continuous, strategy="none")
        
        # Save interim and processed datasets
        df_continuous.to_parquet(self.interim_dir / "city_day_continuous_raw.parquet", index=False)
        df_ffill.to_parquet(self.processed_dir / "city_day_preprocessed_ffill.parquet", index=False)
        df_native.to_parquet(self.processed_dir / "city_day_preprocessed_native.parquet", index=False)
        
        # Save eligibility table
        eligibility_df.to_csv(self.interim_dir / "city_eligibility_matrix.csv", index=False)
        
        # Verify raw files were not modified
        verify_raw_immutability(self.raw_dir)
        self.audit_log["raw_immutability_verified"] = True
        
        # Write audit summary JSON
        with open(self.interim_dir / "preprocessing_audit_summary.json", "w") as f:
            json.dump(self.audit_log, f, indent=2)
            
        logger.info("Preprocessing complete. Artifacts saved.")
        return df_continuous, df_ffill, eligibility_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pipeline = PreprocessingPipeline()
    pipeline.run_pipeline()
    print("[SUCCESS] Preprocessing pipeline executed successfully.")
