"""
Phase 9.2 Automated Unit and Integration Tests for Prediction Uncertainty Quantification.

Verifies:
1. Frozen model predictions remain strictly unchanged.
2. Calibration uses strictly pre-2019 validation observations (<= 2018-12-31).
3. Neither 2019 test nor 2020 stress targets are ever accessed for calibration.
4. Calibration quantiles are deterministic and adhere to the finite-sample order-statistic rank.
5. 95% intervals are at least as wide as 90% intervals for symmetric construction.
6. Lower prediction interval bounds are non-negative (truncated at 0.0).
7. Upper prediction interval bounds are not artificially clipped at 500.
8. Missing numerical inputs follow imputer conventions without producing NaNs or invalid bounds.
9. All 18 Tier-1 qualified cities are evaluated with positive sample counts.
10. Model artifact SHA-256 hash is identical to baseline.
11. Phase 8 artifact SHA-256 hashes are identical to baseline.
12. Phase 9.1 SHAP artifact SHA-256 hashes are identical to baseline.
13. Coverage and interval width calculations agree with exact hand-calculated ground truth.
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.feature_builder import PROCESSED_DIR, get_feature_columns
from src.models.interpretability import load_frozen_pipeline
from src.models.uncertainty import (
    compute_calibration_quantiles,
    compute_conformal_quantile,
    compute_interval_metrics,
    construct_asymmetric_intervals,
    construct_symmetric_intervals,
)

MODELS_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\models")
REPORTS_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\reports")
DOCS_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\docs\reports")

# Recorded baseline hashes prior to Phase 9.2 execution
BASELINE_HASHES = {
    "models/random_forest_val_model.joblib": "ab7246042b6cb4473c99be214c1d5494c0ade7a6d6309a2a34d724312759b19f",
    "reports/phase8_test_metrics.csv": "16235c23925206e3df260ded0b92c137914170f7bf080cabe8259188055b5125",
    "reports/phase8_stress_metrics.csv": "7e799719f3630ecb55d0a2c1ee189221c2f61748fb51b37b9c312455e929f6c0",
    "reports/phase8_test_predictions.csv": "6d2e27fb81e033aa5d03d0905b03c29b2ae88a8c35558d391485f08c9cbb9889",
    "reports/phase8_stress_predictions.csv": "1622f1090db5fc27bac3efea38d6e85b78664a5e4448cb3d58446e0e7e488b10",
    "reports/phase8_ahmedabad_sensitivity.csv": "b098e3e14f676df7a68da70938036266238b9ca142aabbfe1285c97453282ae3",
    "reports/phase8_classification_metrics.csv": "254e49c76e4ebdd7f40575716a1b5c000e5cccc4e93b227146a04899bf0c4727",
    "docs/reports/PHASE_8_EVALUATION_REPORT.md": "759b00044baaf690bc7108efd423cfc8540f9acddaa0fd66a8aae2e13f81d4eb",
    "src/models/interpretability.py": "0dcfa8993c5b4517693d98351dc900767103b9e93fb75ce176ea593645e58b36",
    "tests/test_interpretability.py": "50803d53c450df228ba7000434c2cf9049ad6c965f5db67f689d14fe47ec8d9a",
    "reports/shap_feature_importance.csv": "6c7e3077a08d4a0bd45c01e43d6b5ad5a560aa016fd764d585bda5833912a183",
    "reports/figures/shap_summary.png": "c1df663f7bb6fa58054e923f69798a7c9fa9e9f4e7ef6f04f3e7dc5dbba538f5",
    "reports/figures/shap_importance.png": "709c6a1b5ef892c1b62e15665f6df456a4aefb8720c86296218e88e20b455305",
    "reports/figures/shap_waterfall_typical.png": "33d4dea3b0a7bb53a5d573c19fa5800b81f37b30835a9af17d6c2ae91f391c4c",
    "reports/figures/shap_waterfall_extreme.png": "a28496f3e03e59c2ccddd035cfc7a97b836b461ff5a04065f9655884e38cfaa1",
    "docs/reports/PHASE_9_INTERPRETABILITY_REPORT.md": "51ffae4835213b5bacc2b9b6a0f450f280075e6bf26f50510e3e409e2d0af4de",
}


def _get_file_hash(rel_path: str) -> str:
    path = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT") / rel_path
    assert path.exists(), f"File does not exist: {path}"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def feature_df() -> pd.DataFrame:
    path = PROCESSED_DIR / "city_day_features.parquet"
    assert path.exists(), f"Missing dataset: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def frozen_pipeline() -> Pipeline:
    return load_frozen_pipeline()


def test_frozen_model_predictions_unchanged(frozen_pipeline, feature_df):
    """Test 1: Frozen model point predictions match ground-truth reference values."""
    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols

    val_sub = feature_df[
        (feature_df["split"] == "val") & feature_df["is_target_valid"] & feature_df["is_shared_cohort"]
    ].sort_values(["Target_Date", "City"]).reset_index(drop=True)

    typ_row = val_sub[(val_sub["City"] == "Gurugram") & (val_sub["Target_Date"] == pd.Timestamp("2018-07-31"))]
    pred_typ = frozen_pipeline.predict(typ_row[fcols])[0]
    assert abs(pred_typ - 115.45455813457224) < 1e-5

    ext_row = val_sub[(val_sub["City"] == "Ahmedabad") & (val_sub["Target_Date"] == pd.Timestamp("2018-11-18"))]
    pred_ext = frozen_pipeline.predict(ext_row[fcols])[0]
    assert abs(pred_ext - 1062.6817804309221) < 1e-5


def test_calibration_partition_barrier(frozen_pipeline, feature_df):
    """Test 2: Calibration uses strictly pre-2019 validation data."""
    cal = compute_calibration_quantiles(feature_df, frozen_pipeline, alpha_levels=(0.10, 0.05))
    assert cal["n_calibration"] == 2982
    assert cal["calibration_date_min"] == "2018-07-01"
    assert cal["calibration_date_max"] == "2018-12-31"


def test_test_and_stress_targets_quarantined(frozen_pipeline, feature_df):
    """Test 3: Neither 2019 test nor 2020 stress rows are used in calibration."""
    cal = compute_calibration_quantiles(feature_df, frozen_pipeline, alpha_levels=(0.10,))
    # Target dates in calibration strictly <= 2018-12-31
    assert pd.Timestamp(cal["calibration_date_max"]) < pd.Timestamp("2019-01-01")


def test_deterministic_finite_sample_quantiles():
    """Test 4: Quantile rank follows k = ceil((n+1)(1-alpha)) deterministically."""
    scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    # n = 10, alpha = 0.10 -> (10+1)*0.9 = 9.9 -> ceil(9.9) = 10 -> index 9 -> 100.0
    q, k, idx = compute_conformal_quantile(scores, alpha=0.10)
    assert k == 10
    assert idx == 9
    assert q == 100.0

    # Repeat produces identical result
    q2, k2, idx2 = compute_conformal_quantile(scores, alpha=0.10)
    assert q == q2 and k == k2 and idx == idx2


def test_95_intervals_at_least_as_wide_as_90(frozen_pipeline, feature_df):
    """Test 5: 95% symmetric intervals are at least as wide as 90% intervals."""
    cal = compute_calibration_quantiles(feature_df, frozen_pipeline, alpha_levels=(0.10, 0.05))
    q_90 = cal["quantiles"]["symmetric_90"]["quantile_value"]
    q_95 = cal["quantiles"]["symmetric_95"]["quantile_value"]
    assert q_95 >= q_90, f"Expected q_95 ({q_95}) >= q_90 ({q_90})"

    preds = np.array([25.0, 100.0, 350.0])
    l_90, u_90 = construct_symmetric_intervals(preds, q_90, truncate_lower=True)
    l_95, u_95 = construct_symmetric_intervals(preds, q_95, truncate_lower=True)

    w_90 = u_90 - l_90
    w_95 = u_95 - l_95
    assert np.all(w_95 >= w_90), "95% intervals must be at least as wide as 90% intervals"


def test_lower_bounds_non_negative():
    """Test 6: Truncated lower bounds never drop below zero."""
    preds = np.array([10.0, 5.0, 0.0])
    q = 50.0
    lower, upper = construct_symmetric_intervals(preds, q, truncate_lower=True)
    assert np.all(lower >= 0.0), f"Found negative lower bounds: {lower}"


def test_upper_bounds_not_clipped_at_500():
    """Test 7: Continuous upper bounds are not artificially clipped at 500."""
    preds = np.array([480.0, 1062.68])
    q = 57.0621
    _, upper = construct_symmetric_intervals(preds, q, truncate_lower=True)
    assert upper[0] > 500.0, "Prediction near 500 with buffer must exceed 500"
    assert upper[1] > 1000.0, "Extreme prediction must not be clipped at 500"


def test_missing_numerical_inputs_handled(frozen_pipeline):
    """Test 8: Inputs with missing numerical predictors produce finite interval bounds."""
    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols

    synth_dict = {"City": "Delhi"}
    for c in num_cols:
        synth_dict[c] = np.nan
    synth_dict["AQI_t"] = 120.0
    synth_df = pd.DataFrame([synth_dict])

    preds = frozen_pipeline.predict(synth_df)
    lower, upper = construct_symmetric_intervals(preds, q_value=57.0621, truncate_lower=True)

    assert np.isfinite(lower[0]), "Lower bound must be finite"
    assert np.isfinite(upper[0]), "Upper bound must be finite"
    assert lower[0] >= 0.0


def test_all_18_cities_handled(feature_df):
    """Test 9: 18 Tier-1 qualified cities are all evaluated."""
    test_sub = feature_df[(feature_df["split"] == "test") & feature_df["is_target_valid"] & feature_df["is_shared_cohort"]]
    cities = test_sub["City"].unique()
    assert len(cities) == 18, f"Expected 18 Tier-1 cities, got {len(cities)}"


def test_model_artifact_hash_unchanged():
    """Test 10: Model artifact SHA-256 hash matches baseline."""
    curr = _get_file_hash("models/random_forest_val_model.joblib")
    assert curr == BASELINE_HASHES["models/random_forest_val_model.joblib"]


def test_phase8_artifact_hashes_unchanged():
    """Test 11: Phase 8 metrics and report SHA-256 hashes match baseline."""
    p8_files = [
        "reports/phase8_test_metrics.csv",
        "reports/phase8_stress_metrics.csv",
        "reports/phase8_test_predictions.csv",
        "reports/phase8_stress_predictions.csv",
        "reports/phase8_ahmedabad_sensitivity.csv",
        "reports/phase8_classification_metrics.csv",
        "docs/reports/PHASE_8_EVALUATION_REPORT.md",
    ]
    for rel in p8_files:
        assert _get_file_hash(rel) == BASELINE_HASHES[rel], f"Hash mismatch for {rel}"


def test_phase91_artifact_hashes_unchanged():
    """Test 12: Phase 9.1 SHAP artifact SHA-256 hashes match baseline."""
    p91_files = [
        "src/models/interpretability.py",
        "tests/test_interpretability.py",
        "reports/shap_feature_importance.csv",
        "reports/figures/shap_summary.png",
        "reports/figures/shap_importance.png",
        "reports/figures/shap_waterfall_typical.png",
        "reports/figures/shap_waterfall_extreme.png",
        "docs/reports/PHASE_9_INTERPRETABILITY_REPORT.md",
    ]
    for rel in p91_files:
        assert _get_file_hash(rel) == BASELINE_HASHES[rel], f"Hash mismatch for {rel}"


def test_coverage_and_width_hand_calculation():
    """Test 13: Coverage and width metrics match hand-calculated ground truth."""
    y_true = np.array([50.0, 100.0, 150.0, 200.0])
    low = np.array([40.0, 90.0, 160.0, 180.0])  # y_true[2]=150 is NOT in [160, 210]
    high = np.array([60.0, 110.0, 210.0, 220.0])

    m = compute_interval_metrics(y_true, low, high)
    # Expected: 3 covered out of 4 -> 0.75
    assert m["n_evaluated"] == 4
    assert abs(m["empirical_coverage"] - 0.75) < 1e-9
    # Widths: [20, 20, 50, 40] -> mean = 32.5, median = 30.0, min = 20.0, max = 50.0
    assert abs(m["mean_width"] - 32.5) < 1e-9
    assert abs(m["median_width"] - 30.0) < 1e-9
    assert abs(m["min_width"] - 20.0) < 1e-9
    assert abs(m["max_width"] - 50.0) < 1e-9
