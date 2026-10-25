"""
Automated Unit and Integration Tests for Phase 6 Feature Engineering.

Verifies:
1. City boundary isolation (no cross-city spillover in lags or rolling statistics).
2. Next-calendar-day target alignment (target is strictly date + 1 day).
3. Calendar continuity and missing-date handling (multi-day gaps never bridge).
4. Future perturbation invariance (perturbing tau > t leaves features at t unchanged).
5. Target label integrity (missing AQI targets are never imputed).
6. Rolling-window endpoint correctness (windows end at t; exclude t+1).
7. Chronological partition disjointness (Train, Val, Test, Stress do not overlap).
8. Persistence baseline eligibility (predictions generated iff AQI_t is observed).
9. Training-only fitting of learned preprocessing (zero test leakage).
10. Full integration validation on actual preprocessed Tier-1 datasets.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.feature_builder import (
    build_target,
    build_lag_features,
    build_pollutant_features,
    build_rolling_features,
    build_calendar_features,
    assign_chronological_splits,
    build_cohort_flags,
    build_feature_pipeline,
    create_ridge_preprocessor,
    create_tree_preprocessor,
    get_feature_columns,
    load_tier1_cities,
    SPLIT_BOUNDS,
    TIER1_CITIES,
)

PROCESSED_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\data\processed")


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def synthetic_multicity_df() -> pd.DataFrame:
    """Creates a small synthetic dataset for two cities with known values."""
    dates_a = pd.date_range("2018-01-01", "2018-01-10", freq="D")
    dates_b = pd.date_range("2018-01-01", "2018-01-10", freq="D")

    df_a = pd.DataFrame({
        "City": "CityA",
        "Date": dates_a,
        "AQI": [100.0, 110.0, 120.0, 130.0, 140.0, np.nan, 160.0, 170.0, 180.0, 190.0],
        "PM2.5": [50.0] * 10,
        "PM10": [100.0] * 10,
        "NO2": [20.0] * 10,
        "SO2": [10.0] * 10,
        "CO": [1.0] * 10,
        "O3": [30.0] * 10,
    })

    df_b = pd.DataFrame({
        "City": "CityB",
        "Date": dates_b,
        "AQI": [500.0, 510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0],
        "PM2.5": [250.0] * 10,
        "PM10": [400.0] * 10,
        "NO2": [40.0] * 10,
        "SO2": [20.0] * 10,
        "CO": [2.0] * 10,
        "O3": [50.0] * 10,
    })

    return pd.concat([df_a, df_b], ignore_index=True)


@pytest.fixture
def synthetic_gapped_df() -> pd.DataFrame:
    """Creates a synthetic dataset with an intentional calendar gap (Jan 1, Jan 2, Jan 5)."""
    return pd.DataFrame({
        "City": ["CityA", "CityA", "CityA"],
        "Date": pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-05"]),
        "AQI": [100.0, 110.0, 200.0],
        "PM2.5": [50.0, 55.0, 100.0],
        "PM10": [100.0, 110.0, 200.0],
        "NO2": [20.0, 22.0, 40.0],
        "SO2": [10.0, 11.0, 20.0],
        "CO": [1.0, 1.1, 2.0],
        "O3": [30.0, 32.0, 60.0],
    })


# ==============================================================================
# UNIT TESTS
# ==============================================================================

def test_city_boundary_isolation(synthetic_multicity_df):
    """
    Test 1: City boundary isolation.
    Asserts that features and targets for CityB never use observations from CityA.
    """
    df = synthetic_multicity_df.copy()
    pipe = build_feature_pipeline(df, tier1_only=False)

    # CityA last row should have null target
    city_a_last = pipe[pipe["City"] == "CityA"].iloc[-1]
    assert pd.isna(city_a_last["Target_Date"]), "CityA last row must not have Target_Date"
    assert np.isnan(city_a_last["Target_AQI"]), "CityA last row Target_AQI must be NaN (not CityB first row)"

    # CityB first row lag features must not contain CityA observations
    city_b_first = pipe[pipe["City"] == "CityB"].iloc[0]
    assert np.isnan(city_b_first["AQI_t_minus_1"]), "CityB first row AQI_t_minus_1 must be NaN (not CityA last row)"
    assert np.isnan(city_b_first["AQI_t_minus_6"]), "CityB first row AQI_t_minus_6 must be NaN"
    assert np.isnan(city_b_first["AQI_roll_mean_3d"]), "CityB first row 3d rolling mean must be NaN"


def test_next_calendar_day_target_alignment(synthetic_multicity_df):
    """
    Test 2: Next-day target alignment.
    Asserts that Target_Date == Date + 1 day for all internal records.
    """
    df = synthetic_multicity_df.copy()
    pipe = build_feature_pipeline(df, tier1_only=False)

    for city, grp in pipe.groupby("City"):
        # For all except last row
        internal = grp.iloc[:-1]
        diffs = (pd.to_datetime(internal["Target_Date"]) - pd.to_datetime(internal["Date"])).dt.days
        assert (diffs == 1).all(), f"All internal records in {city} must have Target_Date exactly 1 day ahead"


def test_calendar_continuity_and_missing_date_handling(synthetic_gapped_df):
    """
    Test 3: Calendar continuity and missing-date handling.
    Asserts that when a multi-day gap exists (Jan 2 to Jan 5), Jan 5 is NOT used as next-day target for Jan 2.
    """
    df = synthetic_gapped_df.copy()
    pipe = build_feature_pipeline(df, tier1_only=False)

    row_jan1 = pipe[pipe["Date"] == "2018-01-01"].iloc[0]
    row_jan2 = pipe[pipe["Date"] == "2018-01-02"].iloc[0]
    row_jan5 = pipe[pipe["Date"] == "2018-01-05"].iloc[0]

    # Jan 1 has consecutive Jan 2 target
    assert row_jan1["Target_Date"] == pd.Timestamp("2018-01-02")
    assert row_jan1["Target_AQI"] == 110.0

    # Jan 2 is followed by Jan 5 (gap = 3 days). Target MUST BE NaN / NaT
    assert pd.isna(row_jan2["Target_Date"]), "Jan 2 target date must be NaT across multi-day gap"
    assert np.isnan(row_jan2["Target_AQI"]), "Jan 2 target AQI must be NaN across multi-day gap"

    # Jan 5 previous row is Jan 2 (gap = 3 days). AQI_t_minus_1 MUST BE NaN
    assert np.isnan(row_jan5["AQI_t_minus_1"]), "Jan 5 AQI_t_minus_1 must be NaN across multi-day gap"


def test_future_perturbation_invariance(synthetic_multicity_df):
    """
    Test 4: Future perturbation invariance.
    Asserts that perturbing raw observations at tau > t does not alter features at t.
    """
    df_clean = synthetic_multicity_df.copy()
    pipe_clean = build_feature_pipeline(df_clean, tier1_only=False)

    # Pick origin date t = 2018-01-05
    t_date = pd.Timestamp("2018-01-05")
    num_cols, _, _ = get_feature_columns()

    clean_features_t = pipe_clean[pipe_clean["Date"] == t_date][num_cols].copy()

    # Corrupt future data (dates > 2018-01-05) heavily
    df_perturbed = synthetic_multicity_df.copy()
    future_mask = df_perturbed["Date"] > t_date
    df_perturbed.loc[future_mask, "AQI"] = 9999.0
    df_perturbed.loc[future_mask, "PM2.5"] = 8888.0
    df_perturbed.loc[future_mask, "PM10"] = 7777.0

    pipe_perturbed = build_feature_pipeline(df_perturbed, tier1_only=False)
    perturbed_features_t = pipe_perturbed[pipe_perturbed["Date"] == t_date][num_cols].copy()

    # Features at t must be strictly identical
    np.testing.assert_allclose(
        clean_features_t.values,
        perturbed_features_t.values,
        equal_nan=True,
        err_msg="Features at date t were altered by future perturbations (leakage detected!)"
    )


def test_target_label_integrity(synthetic_multicity_df):
    """
    Test 5: Preservation of missing target values.
    Asserts that missing AQI is never imputed in the target column.
    """
    df = synthetic_multicity_df.copy()
    # In CityA, AQI on 2018-01-06 is NaN
    pipe = build_feature_pipeline(df, tier1_only=False)

    row_prior_to_nan = pipe[(pipe["City"] == "CityA") & (pipe["Date"] == "2018-01-05")].iloc[0]
    assert np.isnan(row_prior_to_nan["Target_AQI"]), "Target_AQI for 2018-01-06 must remain NaN (never imputed)"


def test_rolling_window_endpoint_correctness(synthetic_multicity_df):
    """
    Test 6: Rolling-window endpoint correctness.
    Asserts that rolling window at t uses observations [t-W+1, t] and strictly excludes t+1.
    """
    df = synthetic_multicity_df.copy()
    pipe = build_feature_pipeline(df, tier1_only=False)

    # CityB has AQI: [500, 510, 520, 530, ...]
    # For date 2018-01-03 (index 2):
    # 3d window is [Jan 1, Jan 2, Jan 3] -> [500, 510, 520] -> mean = 510.0
    row_jan3 = pipe[(pipe["City"] == "CityB") & (pipe["Date"] == "2018-01-03")].iloc[0]
    assert np.isclose(row_jan3["AQI_roll_mean_3d"], 510.0), "3d rolling mean must be mean of [t-2, t-1, t]"

    # If it had leaked Jan 4 (530), the mean would be 520.0
    assert not np.isclose(row_jan3["AQI_roll_mean_3d"], 520.0), "3d rolling mean leaked t+1!"


def test_chronological_split_disjointness():
    """
    Test 7: Chronological split disjointness.
    Asserts that Train, Val, Test, and Stress date intervals are mutually disjoint.
    """
    train_start, train_end = pd.to_datetime(SPLIT_BOUNDS["train"])
    val_start, val_end = pd.to_datetime(SPLIT_BOUNDS["val"])
    test_start, test_end = pd.to_datetime(SPLIT_BOUNDS["test"])
    stress_start, stress_end = pd.to_datetime(SPLIT_BOUNDS["stress"])

    assert train_end < val_start, "Train and Val intervals must not overlap"
    assert val_end < test_start, "Val and Test intervals must not overlap"
    assert test_end < stress_start, "Test and Stress intervals must not overlap"


def test_persistence_baseline_eligibility(synthetic_multicity_df):
    """
    Test 8: Persistence baseline eligibility.
    Asserts that persistence can only generate predictions when AQI_t is observed.
    """
    df = synthetic_multicity_df.copy()
    pipe = build_feature_pipeline(df, tier1_only=False)

    # CityA on 2018-01-06 has AQI_t = NaN, but target (2018-01-07) = 160.0
    row_nan_origin = pipe[(pipe["City"] == "CityA") & (pipe["Date"] == "2018-01-06")].iloc[0]
    assert np.isnan(row_nan_origin["AQI_t"])
    assert not np.isnan(row_nan_origin["Target_AQI"])
    assert bool(row_nan_origin["is_target_valid"]) is True
    assert bool(row_nan_origin["is_shared_cohort"]) is False, "Row with missing AQI_t must not be shared-cohort eligible"


def test_training_only_preprocessing_isolation(synthetic_multicity_df):
    """
    Test 9: Training-only fitting of learned preprocessing.
    Asserts that Ridge and Tree preprocessors fit strictly on training split.
    """
    df = synthetic_multicity_df.copy()
    pipe = build_feature_pipeline(df, tier1_only=False)
    # Assign dummy split
    pipe["split"] = ["train"] * 10 + ["test"] * 10
    train_df = pipe[pipe["split"] == "train"]

    ridge_prep = create_ridge_preprocessor(train_df)
    tree_prep = create_tree_preprocessor(train_df)

    num_cols, cat_cols, _ = get_feature_columns()

    # Preprocessor must transform without error
    X_train_trans = ridge_prep.transform(train_df[cat_cols + num_cols])
    assert X_train_trans.shape[0] == 10

    # Unseen category in test must be handled without error
    test_df = pipe[pipe["split"] == "test"].copy()
    X_test_trans = ridge_prep.transform(test_df[cat_cols + num_cols])
    assert X_test_trans.shape[0] == 10

    tree_cat_trans = tree_prep.transform(test_df[cat_cols])
    assert tree_cat_trans.shape[0] == 10


# ==============================================================================
# INTEGRATION TESTS ON REAL PROJECT DATASETS
# ==============================================================================

@pytest.mark.skipif(
    not (PROCESSED_DIR / "city_day_features.parquet").exists(),
    reason="city_day_features.parquet does not exist yet"
)
def test_real_feature_dataset_integrity():
    """
    Test 10: Integration test verifying the real Tier-1 feature dataset.
    Verifies row counts, column completeness, and cohort counts.
    """
    feat_df = pd.read_parquet(PROCESSED_DIR / "city_day_features.parquet")

    # Tier-1 cities
    cities = feat_df["City"].unique()
    assert len(cities) == 18, f"Expected 18 Tier-1 cities, found {len(cities)}"
    for c in TIER1_CITIES:
        assert c in cities, f"City {c} missing from feature dataset"

    # Total grid rows
    assert len(feat_df) == 27303, f"Expected 27,303 continuous grid rows, got {len(feat_df)}"

    # Required columns
    num_cols, cat_cols, target_col = get_feature_columns()
    for col in num_cols + cat_cols + [target_col, "split", "is_target_valid", "is_shared_cohort"]:
        assert col in feat_df.columns, f"Required column {col} missing from feature dataset"

    # Target-valid count overall (22,806)
    target_valid_count = feat_df["is_target_valid"].sum()
    assert target_valid_count == 22806, f"Expected 22,806 target-valid rows, got {target_valid_count}"

    # 2019 Primary Held-Out Test Cohort counts
    test_mask = feat_df["split"] == "test"
    assert test_mask.sum() == 6570, f"Expected 6,570 continuous days in 2019 test, got {test_mask.sum()}"

    test_target_valid = (test_mask & feat_df["is_target_valid"]).sum()
    assert test_target_valid == 6286, f"Expected 6,286 target-valid rows in 2019 test, got {test_target_valid}"

    test_shared_cohort = (test_mask & feat_df["is_shared_cohort"]).sum()
    assert test_shared_cohort == 6201, f"Expected 6,201 shared persistence rows in 2019 test, got {test_shared_cohort}"
