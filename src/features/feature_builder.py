"""
Modular Feature Engineering Pipeline for India Air Quality Intelligence.

Forecasting Objective:
Predict continuous composite AQI for city c on calendar day t+1 using strictly
information available at or prior to day t:
    y_{c, t+1} = AQI_{c, t+1}
    Information set: I_{c, t} = { X_{c, tau}, AQI_{c, tau} | tau <= t }

Design Principles:
1. Strict Temporal Barrier: No feature may use observations from t+1.
2. Group Boundary Isolation: Grouping by City prevents inter-city temporal leakage.
3. Target Preservation: Target AQI (t+1) is strictly NEVER imputed.
4. Exact Calendar Continuity: Gaps in daily observation dates never bridge across multi-day intervals.
5. Deterministic Calendar Alignment: Target calendar features describe the known target date (t+1).
6. Chronological Partitioning: Partitions are indexed strictly by Target Date (t+1).
7. Training-Only Learned Preprocessing: Preprocessing fits strictly on the historical training set.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\data\processed")
INTERIM_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\data\interim")

# 18 Tier-1 Qualified Cities verified from city_eligibility_matrix.csv
TIER1_CITIES = [
    "Ahmedabad", "Amaravati", "Amritsar", "Bengaluru", "Brajrajnagar",
    "Chennai", "Delhi", "Gurugram", "Hyderabad", "Jaipur",
    "Jorapokhar", "Kolkata", "Lucknow", "Mumbai", "Patna",
    "Talcher", "Thiruvananthapuram", "Visakhapatnam"
]

SPLIT_BOUNDS = {
    "train": ("2015-01-02", "2018-06-30"),
    "val": ("2018-07-01", "2018-12-31"),
    "test": ("2019-01-01", "2019-12-31"),
    "stress": ("2020-01-01", "2020-07-01"),
}

NUMERICAL_FEATURES = [
    # Autoregressive target lags
    "AQI_t",
    "AQI_t_minus_1",
    "AQI_t_minus_6",
    # Chemical precursor measurements at prediction origin t
    "PM25_t",
    "PM10_t",
    "NO2_t",
    "SO2_t",
    "CO_t",
    "O3_t",
    "PM_ratio_t",
    # Backward rolling moments (closed at t)
    "AQI_roll_mean_3d",
    "AQI_roll_mean_7d",
    "AQI_roll_std_7d",
    # Deterministic calendar features for target day t+1
    "target_sin_doy",
    "target_cos_doy",
    "target_dow",
    "target_is_weekend",
    "target_month",
]

CATEGORICAL_FEATURES = ["City"]
TARGET_COL = "Target_AQI"


def load_tier1_cities(eligibility_path: Optional[Path] = None) -> List[str]:
    """Loads Tier 1 cities dynamically from eligibility matrix if present, else fallback."""
    path = eligibility_path or (INTERIM_DIR / "city_eligibility_matrix.csv")
    if path.exists():
        df = pd.read_csv(path)
        col = "CoverageTier" if "CoverageTier" in df.columns else "Eligibility_Tier"
        tier1 = df[df[col].str.contains("Tier 1")]["City"].tolist()
        return sorted(tier1)
    return sorted(TIER1_CITIES)


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the 1-day-ahead target AQI (t+1) and target date.
    Strictly grouped by City so the terminal row of each city becomes NaN/NaT.
    Guarantees calendar continuity: target date must be exactly 1 calendar day ahead.
    If a calendar gap exists, target becomes NaN/NaT so multi-day gaps never bridge.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    next_date = df.groupby("City")["Date"].shift(-1)
    next_aqi = df.groupby("City")["AQI"].shift(-1)

    # Check exact calendar continuity (must be exactly +1 calendar day)
    is_consecutive = (next_date - df["Date"]) == pd.Timedelta(days=1)

    df["Target_Date"] = np.where(is_consecutive, next_date, pd.NaT)
    df["Target_Date"] = pd.to_datetime(df["Target_Date"])
    df["Target_AQI"] = np.where(is_consecutive, next_aqi, np.nan)
    return df


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs autoregressive AQI lags relative to prediction origin t.
    - AQI_t: observed AQI on date t (lag 1 to target t+1)
    - AQI_t_minus_1: observed AQI on date t-1 (lag 2 to target t+1)
    - AQI_t_minus_6: observed AQI on date t-6 (lag 7 to target t+1; weekly anchor)

    Verifies calendar continuity:
    (Date_t - Date_{t-1}) == 1 day and (Date_t - Date_{t-6}) == 6 days.
    If multi-day gaps exist, the lag feature evaluates to NaN.
    """
    df = df.copy()
    df["AQI_t"] = df["AQI"]

    prev_date_1 = df.groupby("City")["Date"].shift(1)
    prev_aqi_1 = df.groupby("City")["AQI"].shift(1)
    is_lag1_consecutive = (df["Date"] - prev_date_1) == pd.Timedelta(days=1)
    df["AQI_t_minus_1"] = np.where(is_lag1_consecutive, prev_aqi_1, np.nan)

    prev_date_6 = df.groupby("City")["Date"].shift(6)
    prev_aqi_6 = df.groupby("City")["AQI"].shift(6)
    is_lag6_consecutive = (df["Date"] - prev_date_6) == pd.Timedelta(days=6)
    df["AQI_t_minus_6"] = np.where(is_lag6_consecutive, prev_aqi_6, np.nan)

    return df


def build_pollutant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts consolidated chemical precursor measurements available on or before date t.
    Also computes the fine-to-coarse PM ratio (PM2.5 / PM10) clipped to [0, 1].
    """
    df = df.copy()
    df["PM25_t"] = df["PM2.5"]
    df["PM10_t"] = df["PM10"]
    df["NO2_t"] = df["NO2"]
    df["SO2_t"] = df["SO2"]
    df["CO_t"] = df["CO"]
    df["O3_t"] = df["O3"]

    # Safe ratio calculation
    valid_pm = (df["PM25_t"].notna()) & (df["PM10_t"].notna()) & (df["PM10_t"] > 0)
    ratio = np.where(valid_pm, df["PM25_t"] / (df["PM10_t"] + 1e-5), np.nan)
    df["PM_ratio_t"] = np.clip(ratio, 0.0, 1.0)
    return df


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs backward-looking rolling statistics closed on the right at date t.
    Never extends into t+1.
    - AQI_roll_mean_3d: window=3, min_periods=2
    - AQI_roll_mean_7d: window=7, min_periods=4
    - AQI_roll_std_7d: window=7, min_periods=3
    """
    df = df.copy()
    df["AQI_roll_mean_3d"] = df.groupby("City")["AQI"].transform(
        lambda s: s.rolling(window=3, min_periods=2).mean()
    )
    df["AQI_roll_mean_7d"] = df.groupby("City")["AQI"].transform(
        lambda s: s.rolling(window=7, min_periods=4).mean()
    )
    df["AQI_roll_std_7d"] = df.groupby("City")["AQI"].transform(
        lambda s: s.rolling(window=7, min_periods=3).std()
    )
    return df


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs deterministic calendar features for the target date t+1.
    Because calendar time is deterministic, day-of-week, month, and circular harmonics
    for tomorrow are mathematically known today without leakage.
    """
    df = df.copy()
    valid_target_date = df["Target_Date"].notna()
    target_dt = pd.to_datetime(df["Target_Date"])

    target_doy = target_dt.dt.dayofyear
    df["target_sin_doy"] = np.where(
        valid_target_date, np.sin(2 * np.pi * target_doy / 365.25), np.nan
    )
    df["target_cos_doy"] = np.where(
        valid_target_date, np.cos(2 * np.pi * target_doy / 365.25), np.nan
    )
    df["target_dow"] = np.where(valid_target_date, target_dt.dt.dayofweek, np.nan)
    df["target_is_weekend"] = np.where(
        valid_target_date, df["target_dow"].isin([5, 6]).astype(float), np.nan
    )
    df["target_month"] = np.where(valid_target_date, target_dt.dt.month, np.nan)
    return df


def assign_chronological_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns chronological evaluation partition labels based strictly on Target_Date (t+1):
    - train:  2015-01-02 to 2018-06-30
    - val:    2018-07-01 to 2018-12-31
    - test:   2019-01-01 to 2019-12-31 (untouched held-out benchmark)
    - stress: 2020-01-01 to 2020-07-01 (COVID-19 shock evaluation)
    """
    df = df.copy()
    target_dt = pd.to_datetime(df["Target_Date"])

    conditions = [
        (target_dt >= SPLIT_BOUNDS["train"][0]) & (target_dt <= SPLIT_BOUNDS["train"][1]),
        (target_dt >= SPLIT_BOUNDS["val"][0]) & (target_dt <= SPLIT_BOUNDS["val"][1]),
        (target_dt >= SPLIT_BOUNDS["test"][0]) & (target_dt <= SPLIT_BOUNDS["test"][1]),
        (target_dt >= SPLIT_BOUNDS["stress"][0]) & (target_dt <= SPLIT_BOUNDS["stress"][1]),
    ]
    choices = ["train", "val", "test", "stress"]
    df["split"] = np.select(conditions, choices, default=None)
    return df


def build_cohort_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds boolean cohort flags for fair benchmarking:
    - is_target_valid: Target_AQI is observed (not NaN).
    - is_shared_cohort: Both AQI_t and Target_AQI are observed (Persistence-eligible).
    """
    df = df.copy()
    df["is_target_valid"] = df["Target_AQI"].notna()
    df["is_shared_cohort"] = df["AQI_t"].notna() & df["Target_AQI"].notna()
    return df


def build_feature_pipeline(
    df: pd.DataFrame,
    tier1_only: bool = True,
    eligibility_path: Optional[Path] = None,
    drop_terminal_unlabeled: bool = False,
) -> pd.DataFrame:
    """
    Executes the complete, end-to-end feature engineering pipeline.

    Parameters:
        df: Input DataFrame (e.g. from city_day_preprocessed_ffill.parquet or native).
        tier1_only: If True, filters strictly to the 18 Tier-1 cities.
        eligibility_path: Optional custom path to city_eligibility_matrix.csv.
        drop_terminal_unlabeled: If True, removes rows where Target_Date is NaT (e.g. terminal rows).

    Returns:
        pd.DataFrame with target, lags, pollutants, rolling moments, calendar features,
        splits, and cohort flags.
    """
    df = df.copy()
    if tier1_only:
        tier1 = load_tier1_cities(eligibility_path)
        df = df[df["City"].isin(tier1)].copy()

    df = build_target(df)
    df = build_lag_features(df)
    df = build_pollutant_features(df)
    df = build_rolling_features(df)
    df = build_calendar_features(df)
    df = assign_chronological_splits(df)
    df = build_cohort_flags(df)

    if drop_terminal_unlabeled:
        df = df[df["Target_Date"].notna()].copy()

    return df


def get_feature_columns() -> Tuple[List[str], List[str], str]:
    """Returns numerical feature column names, categorical column names, and target column."""
    return list(NUMERICAL_FEATURES), list(CATEGORICAL_FEATURES), TARGET_COL


def create_ridge_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for Ridge Regression:
    - One-Hot encodes City (handle_unknown='ignore')
    - SimpleImputer(median) + StandardScaler on all numerical features
    Fitted strictly on train_df.
    """
    num_cols, cat_cols, _ = get_feature_columns()
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
        ]
    )
    preprocessor.fit(train_df[cat_cols + num_cols])
    return preprocessor


def create_tree_preprocessor(train_df: pd.DataFrame) -> OrdinalEncoder:
    """
    Constructs an OrdinalEncoder for tree-based models (HistGBM, Random Forest):
    - Ordinal encodes City with unseen category handling (unknown_value=np.nan).
    Fitted strictly on train_df.
    """
    cat_cols = CATEGORICAL_FEATURES
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    encoder.fit(train_df[cat_cols])
    return encoder


def save_feature_datasets(
    processed_dir: Path = PROCESSED_DIR,
    interim_dir: Path = INTERIM_DIR,
) -> Dict[str, Path]:
    """
    Builds and persists the feature datasets for both Variant A (ffill) and Variant B (native).
    """
    saved = {}
    variants = [
        ("city_day_preprocessed_ffill.parquet", "city_day_features_ffill.parquet"),
        ("city_day_preprocessed_native.parquet", "city_day_features_native.parquet"),
    ]
    for in_file, out_file in variants:
        in_path = processed_dir / in_file
        if not in_path.exists():
            logger.warning(f"Input file not found: {in_path}")
            continue
        raw_df = pd.read_parquet(in_path)
        feat_df = build_feature_pipeline(
            raw_df, tier1_only=True, eligibility_path=interim_dir / "city_eligibility_matrix.csv"
        )
        out_path = processed_dir / out_file
        feat_df.to_parquet(out_path, index=False)
        saved[out_file] = out_path
        logger.info(f"Saved {out_file} with shape {feat_df.shape}")

    # Standard alias for primary feature dataset
    primary_out = processed_dir / "city_day_features.parquet"
    if "city_day_features_ffill.parquet" in saved:
        primary_df = pd.read_parquet(saved["city_day_features_ffill.parquet"])
        primary_df.to_parquet(primary_out, index=False)
        saved["city_day_features.parquet"] = primary_out

    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Building and saving Phase 6 feature datasets...")
    out_files = save_feature_datasets()
    for name, path in out_files.items():
        print(f"Created: {name} -> {path}")
