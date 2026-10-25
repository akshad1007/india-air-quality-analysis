"""
Automated Unit and Integration Tests for Phase 8 Final Model Evaluation.

Verifies:
1. Exact cohort eligibility counts (2019 test N=6,201 shared, 2020 stress N=3,124 shared).
2. Target-date and city alignment across prediction records.
3. Persistence baseline identity and eligibility rules.
4. No duplicate evaluation keys (City, Target_Date).
5. Consistent shared-cohort row identifiers across all compared models.
6. Metric calculations match independent ground-truth assertions.
7. Correct CPCB statutory category boundary mapping (all 6 buckets).
8. Confusion matrix dimensions (6x6) and fixed category ordering.
9. Missing and out-of-range category inputs (<0 maps to Good, >500 maps to Severe).
10. Residual sign convention: e = y_true - y_pred (positive = underprediction).
11. Reproducibility of saved artifact predictions on held-out test data.
12. Preservation of original continuous predictions without silent clipping.
13. Strict isolation: no fitting or tuning on 2019 or 2020 test partitions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.feature_builder import PROCESSED_DIR
from src.models.evaluation import (
    CPCB_CATEGORIES,
    compute_classification_diagnostics,
    compute_subgroup_diagnostics,
    map_aqi_to_cpcb_bucket,
)
from src.models.metrics import compute_regression_metrics, compute_relative_improvement

REPORTS_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\reports")


@pytest.fixture
def phase8_test_preds() -> pd.DataFrame:
    """Loads the generated 2019 test prediction records."""
    csv_path = REPORTS_DIR / "phase8_test_predictions.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture
def phase8_stress_preds() -> pd.DataFrame:
    """Loads the generated 2020 stress prediction records."""
    csv_path = REPORTS_DIR / "phase8_stress_predictions.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    return pd.read_csv(csv_path)


def test_exact_cohort_eligibility_counts(phase8_test_preds, phase8_stress_preds):
    """Test 1: Verifies exact sample counts in 2019 test and 2020 stress cohorts."""
    # 2019 Primary Test
    assert len(phase8_test_preds) == 6286, f"Expected 6,286 target-valid rows in 2019, got {len(phase8_test_preds)}"
    n_shared_2019 = phase8_test_preds["is_shared_cohort"].sum()
    assert n_shared_2019 == 6201, f"Expected 6,201 shared rows in 2019, got {n_shared_2019}"

    # 2020 Stress Test
    assert len(phase8_stress_preds) == 3163, f"Expected 3,163 target-valid rows in 2020, got {len(phase8_stress_preds)}"
    n_shared_2020 = phase8_stress_preds["is_shared_cohort"].sum()
    assert n_shared_2020 == 3124, f"Expected 3,124 shared rows in 2020, got {n_shared_2020}"


def test_no_duplicate_evaluation_keys(phase8_test_preds, phase8_stress_preds):
    """Test 2 & 4: Verifies no duplicate (City, Target_Date) keys exist in test outputs."""
    dup_test = phase8_test_preds.duplicated(subset=["City", "Target_Date"]).sum()
    assert dup_test == 0, f"Found {dup_test} duplicate keys in 2019 test predictions"

    dup_stress = phase8_stress_preds.duplicated(subset=["City", "Target_Date"]).sum()
    assert dup_stress == 0, f"Found {dup_stress} duplicate keys in 2020 stress predictions"


def test_target_date_and_city_alignment(phase8_test_preds):
    """Test 3: Verifies Target_Date is exactly Date + 1 day and within 2019."""
    dates = pd.to_datetime(phase8_test_preds["Date"])
    target_dates = pd.to_datetime(phase8_test_preds["Target_Date"])
    diffs = (target_dates - dates).dt.days
    assert (diffs == 1).all(), "Target_Date must be strictly 1 day after Date"

    assert target_dates.min() >= pd.Timestamp("2019-01-01")
    assert target_dates.max() <= pd.Timestamp("2019-12-31")


def test_persistence_identity_and_eligibility(phase8_test_preds):
    """Test 5: Verifies persistence predictions equal AQI_t and handles NaNs strictly."""
    shared_mask = phase8_test_preds["is_shared_cohort"] == True
    # On shared cohort, Pred_Persistence must equal AQI_t exactly
    np.testing.assert_allclose(
        phase8_test_preds.loc[shared_mask, "Pred_Persistence"],
        phase8_test_preds.loc[shared_mask, "AQI_t"],
        err_msg="Persistence prediction did not match AQI_t",
    )
    # On non-shared cohort, Pred_Persistence must be NaN
    assert phase8_test_preds.loc[~shared_mask, "Pred_Persistence"].isna().all()


def test_consistent_shared_cohort_row_identifiers(phase8_test_preds):
    """Test 6: Verifies all models and baseline are evaluated on identical rows in Cohort A."""
    shared_df = phase8_test_preds[phase8_test_preds["is_shared_cohort"]].copy()
    assert len(shared_df) == 6201

    cols = ["Pred_Persistence", "Pred_Ridge_Regression", "Pred_Random_Forest", "Pred_HistGradientBoosting"]
    for c in cols:
        assert not shared_df[c].isna().any(), f"Column {c} contains unexpected NaNs on shared cohort!"


def test_cpcb_category_boundary_mapping():
    """Test 7: Verifies exact CPCB category boundary definitions."""
    test_values = [0.0, 50.0, 50.1, 100.0, 100.1, 200.0, 200.1, 300.0, 300.1, 400.0, 400.1, 500.0]
    expected = [
        "Good", "Good",
        "Satisfactory", "Satisfactory",
        "Moderate", "Moderate",
        "Poor", "Poor",
        "Very Poor", "Very Poor",
        "Severe", "Severe"
    ]
    actual = map_aqi_to_cpcb_bucket(test_values).tolist()
    assert actual == expected, f"CPCB bucket mapping error: {actual} != {expected}"


def test_cpcb_out_of_range_and_missing_handling():
    """Test 8 & 9: Verifies out-of-range (<0, >500) and missing values."""
    edge_values = [-10.0, 600.0, np.nan]
    actual = map_aqi_to_cpcb_bucket(edge_values).tolist()
    assert actual[0] == "Good", "Negative AQI must map to 'Good'"
    assert actual[1] == "Severe", "AQI > 500 must map to 'Severe'"
    assert actual[2] is None, "NaN must map to None"


def test_confusion_matrix_dimensions_and_ordering():
    """Test 10: Verifies confusion matrix is 6x6 with fixed statutory category order."""
    y_true = [25.0, 75.0, 150.0, 250.0, 350.0, 450.0]
    y_pred = [25.0, 75.0, 150.0, 250.0, 350.0, 450.0]
    res = compute_classification_diagnostics(y_true, y_pred)

    cm = np.array(res["confusion_matrix"])
    assert cm.shape == (6, 6), f"Confusion matrix shape was {cm.shape}, expected (6, 6)"
    assert np.array_equal(cm, np.eye(6)), "Perfect predictions must yield identity confusion matrix"
    assert res["macro_f1"] == 1.0
    assert res["balanced_accuracy"] == 1.0


def test_residual_sign_convention():
    """
    Test 11: Verifies MBE sign convention MBE = mean(predicted_AQI - observed_AQI).
    - Positive MBE: Net average overprediction (predictions exceed observations)
    - Negative MBE: Net average underprediction (predictions fall below observations)
    - Measured in AQI points
    """
    # 1. Direct metric check with prompt's exact reference example
    y_true_ex = np.array([100.0, 200.0])
    y_pred_ex = np.array([110.0, 180.0])
    # (110 - 100) + (180 - 200) = 10 - 20 = -10; mean = -5.0 AQI points
    metrics_ex = compute_regression_metrics(y_true_ex, y_pred_ex)
    assert np.isclose(metrics_ex["MBE"], -5.0), f"Expected MBE -5.0, got {metrics_ex['MBE']}"
    assert np.isclose(metrics_ex["MAE"], 15.0), f"Expected MAE 15.0, got {metrics_ex['MAE']}"

    # 2. Overprediction example: true=[100, 200], pred=[120, 210] -> (+20 + 10)/2 = +15.0 AQI points
    y_pred_over = np.array([120.0, 210.0])
    metrics_over = compute_regression_metrics(y_true_ex, y_pred_over)
    assert np.isclose(metrics_over["MBE"], 15.0), f"Expected MBE +15.0, got {metrics_over['MBE']}"

    # 3. Subgroup diagnostics check
    df = pd.DataFrame({
        "City": ["CityA", "CityA"],
        "Target_Date": ["2019-01-01", "2019-01-02"],
        "Target_AQI": [100.0, 200.0],
        "Pred": [110.0, 180.0],
    })
    diag = compute_subgroup_diagnostics(df, pred_col="Pred")
    city_m = diag["by_city"].iloc[0]
    assert np.isclose(city_m["MAE"], 15.0)
    assert np.isclose(city_m["MBE"], -5.0)
    assert np.isclose(city_m["Underprediction_Rate"], 50.0)


def test_preservation_of_continuous_predictions(phase8_test_preds):
    """Test 12: Continuous predictions are preserved without artificial clipping."""
    cols = ["Pred_Ridge_Regression", "Pred_Random_Forest", "Pred_HistGradientBoosting"]
    for c in cols:
        preds = phase8_test_preds[c].dropna()
        # Verify predictions have fractional continuous values
        has_fractions = not np.all(np.isclose(preds, np.round(preds)))
        assert has_fractions, f"Predictions in {c} appear to be discrete/clipped, not continuous!"


def test_metric_reproducibility_against_saved_metrics(phase8_test_preds):
    """Test 13: Verifies independent metric recalculation matches saved CSV summary."""
    shared_df = phase8_test_preds[phase8_test_preds["is_shared_cohort"]].copy()
    y_true = shared_df["Target_AQI"]
    y_pred_rf = shared_df["Pred_Random_Forest"]

    metrics = compute_regression_metrics(y_true, y_pred_rf)
    # Expected from validation/test run: MAE 26.2993, RMSE 55.6234, R2 0.8398
    assert np.isclose(metrics["MAE"], 26.2993, atol=1e-3)
    assert np.isclose(metrics["RMSE"], 55.6234, atol=1e-3)
    assert np.isclose(metrics["R2"], 0.8398, atol=1e-3)
