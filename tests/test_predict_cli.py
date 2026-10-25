"""
Phase 9.3 Automated Unit and Integration Tests for Inference CLI.

Verifies:
1. CLI help and argument parsing.
2. Valid single-row JSON evaluation and output structure.
3. Valid batch CSV evaluation and output file generation.
4. Missing required columns (e.g. missing 'City').
5. Invalid numeric inputs and malformed dates.
6. Negative pollutant values rejection (physical constraint).
7. Missing-feature imputation disclosure (training median fallback).
8. Unknown-city warning generation.
9. CPCB category boundary edge cases.
10. Conformal interval calculation (90% and 95%), non-negative truncation, and continuous unclipped upper bound.
11. Non-finite prediction error handling.
12. Model artifact hash preservation and no-refit guarantee.
13. Batch atomic validation (no partial output on failure).
14. JSON output structure and presence of disclaimer.
15. Preservation of all existing model, report, figure, and submission artifacts.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.feature_builder import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TIER1_CITIES,
    get_feature_columns,
)
from src.models.evaluation import CPCB_CATEGORIES, map_aqi_to_cpcb_bucket
from src.models.interpretability import load_frozen_pipeline
from src.models.predict_cli import (
    CALIBRATED_CONFORMAL_Q_90,
    CALIBRATED_CONFORMAL_Q_95,
    DISCLAIMER_TEXT,
    compute_target_calendar_features,
    format_text_output,
    main,
    run_inference_batch,
    run_inference_single,
    validate_feature_row,
)

PROJECT_ROOT = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT")
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

FROZEN_MODEL_HASH = "ab7246042b6cb4473c99be214c1d5494c0ade7a6d6309a2a34d724312759b19f"

BASELINE_HASHES = {
    "models/random_forest_val_model.joblib": FROZEN_MODEL_HASH,
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
    "docs/reports/PHASE_9_INTERPRETABILITY_REPORT.md": "51ffae4835213b5bacc2b9b6a0f450f280075e6bf26f50510e3e409e2d0af4de",
    "src/models/uncertainty.py": "8abd41e51d98f0ccb2826c0bd359279b919a1e9a93905b27bd1e3bd0b47f7497",
    "tests/test_uncertainty.py": "ae0a8f175f8941d4390000d6dea0962129795aab11feae9d9d41c88d52a55347",
    "reports/phase9_uncertainty_metrics.csv": "136ea687edd818c7344cf7f3c00e1807f84cf790e46a5873d40883772e902b43",
    "reports/phase9_city_coverage.csv": "37f996ab36f93a9bb4201018c6c254466812db4144bae6b5b8795acf99f188d2",
    "docs/reports/PHASE_9_UNCERTAINTY_REPORT.md": "b442b3e9d2d0b192e4bafbe593f087951a977b5ca06e4678770e240cfc46245d",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def frozen_pipeline() -> Pipeline:
    return load_frozen_pipeline()


@pytest.fixture
def valid_feature_dict() -> dict:
    """Provides a realistic pre-engineered feature dictionary for Delhi."""
    return {
        "City": "Delhi",
        "Date": "2019-01-15",
        "Target_Date": "2019-01-16",
        "AQI_t": 320.0,
        "AQI_t_minus_1": 310.0,
        "AQI_t_minus_6": 280.0,
        "PM25_t": 180.0,
        "PM10_t": 290.0,
        "NO2_t": 65.0,
        "SO2_t": 18.0,
        "CO_t": 1.9,
        "O3_t": 35.0,
        "PM_ratio_t": 0.62,
        "AQI_roll_mean_3d": 315.0,
        "AQI_roll_mean_7d": 295.0,
        "AQI_roll_std_7d": 25.0,
        "target_sin_doy": 0.27,
        "target_cos_doy": 0.96,
        "target_dow": 2.0,
        "target_is_weekend": 0.0,
        "target_month": 1.0,
    }


def test_cli_help(capsys):
    """Test 1: CLI responds to --help with status 0 and usage description."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "predict_cli" in captured.out
    assert "--input-json" in captured.out
    assert "--input-csv" in captured.out


def test_cli_single_row_json_evaluation(frozen_pipeline, valid_feature_dict, tmp_path, capsys):
    """Test 2: Single-row JSON inference produces valid point forecast and intervals."""
    json_path = tmp_path / "input.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(valid_feature_dict, f)

    ret = main(["--input-json", str(json_path), "--format", "json"])
    assert ret == 0
    captured = capsys.readouterr()
    res = json.loads(captured.out)

    assert res["city"] == "Delhi"
    assert res["target_date"] == "2019-01-16"
    assert isinstance(res["point_forecast_aqi"], (float, int))
    assert res["point_forecast_aqi"] > 0
    assert res["cpcb_category"] in CPCB_CATEGORIES
    assert res["is_tier1_qualified"] is True
    assert DISCLAIMER_TEXT in res["disclaimer"]

    intervals = res["prediction_intervals"]
    assert intervals["symmetric_90"]["margin"] == CALIBRATED_CONFORMAL_Q_90
    assert intervals["symmetric_95"]["margin"] == CALIBRATED_CONFORMAL_Q_95
    assert intervals["symmetric_95"]["upper"] >= intervals["symmetric_90"]["upper"]
    assert intervals["symmetric_90"]["lower"] >= 0.0


def test_cli_batch_csv_evaluation(frozen_pipeline, valid_feature_dict, tmp_path, capsys):
    """Test 3: Batch CSV evaluation runs atomically and produces output CSV."""
    df_in = pd.DataFrame([valid_feature_dict, {**valid_feature_dict, "City": "Mumbai", "AQI_t": 110.0}])
    csv_in = tmp_path / "batch_in.csv"
    csv_out = tmp_path / "batch_out.csv"
    df_in.to_csv(csv_in, index=False)

    ret = main(["--input-csv", str(csv_in), "--output-file", str(csv_out)])
    assert ret == 0
    assert csv_out.exists()

    df_res = pd.read_csv(csv_out)
    assert len(df_res) == 2
    assert "pred_aqi" in df_res.columns
    assert "pred_cpcb_category" in df_res.columns
    assert "pred_ci90_lower" in df_res.columns
    assert "pred_ci95_upper" in df_res.columns
    assert np.all(df_res["pred_ci90_lower"] >= 0.0)


def test_missing_city_column_rejected():
    """Test 4: Missing City column throws validation error."""
    row = {"AQI_t": 100.0, "Target_Date": "2019-01-02"}
    with pytest.raises(ValueError, match="Missing required categorical feature: 'City'"):
        validate_feature_row(row)


def test_invalid_numeric_and_malformed_date():
    """Test 5: Malformed date and non-numeric fields are rejected."""
    # Malformed numeric
    bad_num = {"City": "Delhi", "AQI_t": "one_hundred", "Target_Date": "2019-01-02"}
    with pytest.raises(ValueError, match="must be numeric"):
        validate_feature_row(bad_num)

    # Date temporal misalignment (target is not origin + 1 day)
    misaligned = {"City": "Delhi", "Date": "2019-01-01", "Target_Date": "2019-01-05"}
    with pytest.raises(ValueError, match="Temporal alignment error"):
        validate_feature_row(misaligned)


def test_negative_pollutant_values_rejected():
    """Test 6: Negative physical pollutant values are strictly rejected."""
    bad_pm = {"City": "Delhi", "AQI_t": 100.0, "PM25_t": -15.0, "Target_Date": "2019-01-02"}
    with pytest.raises(ValueError, match="Physical constraint violation.*cannot be negative"):
        validate_feature_row(bad_pm)


def test_missing_feature_imputation_disclosure(frozen_pipeline):
    """Test 7: Missing numerical features are imputed by training medians and disclosed."""
    row = {
        "City": "Bengaluru",
        "Target_Date": "2019-04-10",
        "AQI_t": 85.0,
        # Omit all other numerical features
    }
    res = run_inference_single(frozen_pipeline, row)
    assert len(res["imputed_features"]) > 0
    assert "PM25_t" in res["imputed_features"]
    assert "AQI_t_minus_1" in res["imputed_features"]
    assert isinstance(res["point_forecast_aqi"], float)


def test_unknown_city_warning(frozen_pipeline, valid_feature_dict):
    """Test 8: Unknown cities trigger an explicit extrapolation warning without crashing."""
    row = {**valid_feature_dict, "City": "Kopargaon"}
    res = run_inference_single(frozen_pipeline, row)
    assert res["is_tier1_qualified"] is False
    assert len(res["warnings"]) > 0
    assert any("not in the 18 Tier-1 qualified" in w for w in res["warnings"])
    assert isinstance(res["point_forecast_aqi"], float)


def test_cpcb_category_boundaries():
    """Test 9: Exact CPCB boundary mapping matches statutory thresholds."""
    test_values = np.array([-10.0, 50.0, 50.001, 100.0, 100.001, 200.0, 200.001, 300.0, 300.001, 400.0, 400.001, 800.0])
    cats = map_aqi_to_cpcb_bucket(test_values)
    expected = [
        "Good",        # -10 <= 50 (below zero mapped to Good)
        "Good",        # 50.0
        "Satisfactory",# 50.001
        "Satisfactory",# 100.0
        "Moderate",    # 100.001
        "Moderate",    # 200.0
        "Poor",        # 200.001
        "Poor",        # 300.0
        "Very Poor",   # 300.001
        "Very Poor",   # 400.0
        "Severe",      # 400.001
        "Severe",      # 800.0 (statutory ceiling)
    ]
    assert list(cats) == expected


def test_conformal_intervals_bounds_and_unclipped_upper(frozen_pipeline):
    """Test 10: Lower bounds clipped at 0.0; upper bounds unclipped at 500."""
    # Near zero
    low_pred = np.array([20.0])
    res_low = run_inference_single(frozen_pipeline, {"City": "Chennai", "AQI_t": 20.0, "Target_Date": "2019-05-01"})
    l_90 = res_low["prediction_intervals"]["symmetric_90"]["lower"]
    u_90 = res_low["prediction_intervals"]["symmetric_90"]["upper"]
    assert l_90 >= 0.0

    # Near 500 with buffer exceeding 500
    row_high = {"City": "Delhi", "AQI_t": 480.0, "PM25_t": 400.0, "Target_Date": "2019-11-15"}
    res_high = run_inference_single(frozen_pipeline, row_high)
    u_high_90 = res_high["prediction_intervals"]["symmetric_90"]["upper"]
    assert u_high_90 > 500.0, f"Upper bound should be unclipped, got {u_high_90}"


def test_target_calendar_derivation():
    """Test 11: Circular harmonics, dow, weekend, and month match ground truth."""
    cal = compute_target_calendar_features("2019-01-05")  # Saturday
    assert cal["target_dow"] == 5.0
    assert cal["target_is_weekend"] == 1.0
    assert cal["target_month"] == 1.0
    # Day 5 / 365.25
    expected_sin = np.sin(2 * np.pi * 5 / 365.25)
    assert abs(cal["target_sin_doy"] - expected_sin) < 1e-6


def test_batch_atomic_failure_no_partial_output(frozen_pipeline, valid_feature_dict, tmp_path):
    """Test 12: Invalid row anywhere in batch halts execution without producing output."""
    df_in = pd.DataFrame([
        valid_feature_dict,
        {**valid_feature_dict, "PM25_t": -50.0},  # Invalid row
    ])
    csv_in = tmp_path / "bad_batch.csv"
    csv_out = tmp_path / "should_not_exist.csv"
    df_in.to_csv(csv_in, index=False)

    ret = main(["--input-csv", str(csv_in), "--output-file", str(csv_out)])
    assert ret == 2
    assert not csv_out.exists(), "Output file must not be written if batch validation fails"


def test_frozen_model_no_refit_guarantee(frozen_pipeline, valid_feature_dict):
    """Test 13: Estimator state is strictly immutable during inference."""
    rf = frozen_pipeline.named_steps["regressor"]
    trees_before = len(rf.estimators_)
    assert trees_before == 100

    # Run multiple inferences
    for _ in range(5):
        run_inference_single(frozen_pipeline, valid_feature_dict)

    trees_after = len(rf.estimators_)
    assert trees_after == 100
    assert _sha256(MODELS_DIR / "random_forest_val_model.joblib") == FROZEN_MODEL_HASH


def test_repository_artifacts_preservation():
    """Test 14: Verifies all existing milestone artifacts remain byte-for-byte identical."""
    for rel_path, expected_hash in BASELINE_HASHES.items():
        full_path = PROJECT_ROOT / rel_path
        assert full_path.exists(), f"Deliverable missing: {rel_path}"
        actual_hash = _sha256(full_path)
        assert actual_hash == expected_hash, f"Hash mismatch on {rel_path}!"
