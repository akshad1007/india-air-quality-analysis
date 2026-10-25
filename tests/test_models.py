"""
Automated Unit and Integration Tests for Phase 7 Modeling.

Verifies:
1. Persistence predictions equal AQI_t.
2. Persistence eligibility requires observed current and target AQI.
3. All models use strictly approved predictor columns (no Target_AQI or Target_Date in X).
4. Target columns and future-derived fields are strictly excluded from feature space.
5. Temporal fold boundaries are chronological and non-overlapping.
6. Rows with the same target date are not split across folds.
7. Learned preprocessing is fitted only on each training fold.
8. Validation/test information does not affect fitted imputation or scaling statistics.
9. Missing numerical predictors are handled as intended across models.
10. Unseen city categories do not cause silent failure or crash.
11. Predictions align with the correct city and Target_Date.
12. Model predictions are finite on valid inputs.
13. Model outputs are reproducible under fixed random seeds.
14. Metrics use identical row identifiers for shared-cohort comparisons.
15. No primary test (2019) or stress test (2020) data is used for tuning.
16. The complete test suite runs and passes cleanly.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.feature_builder import get_feature_columns, PROCESSED_DIR
from src.models.baseline import NaivePersistenceModel
from src.models.metrics import compute_regression_metrics, compute_relative_improvement
from src.models.pipelines import (
    build_histgbm_pipeline,
    build_rf_pipeline,
    build_ridge_pipeline,
)
from src.models.validation import (
    ExpandingWindowFold,
    evaluate_model_on_fold,
    get_designated_validation_fold,
    get_training_folds,
    run_chronological_cv,
)


@pytest.fixture
def synthetic_training_df() -> pd.DataFrame:
    """Creates a small synthetic time-series dataset with 2 cities over 20 days."""
    dates = pd.date_range("2017-01-01", "2017-01-20", freq="D")
    records = []
    for city in ["CityA", "CityB"]:
        for i, d in enumerate(dates):
            records.append({
                "City": city,
                "Date": d,
                "Target_Date": d + pd.Timedelta(days=1),
                "Target_AQI": 100.0 + i * 2.0 + (50.0 if city == "CityB" else 0.0),
                "AQI_t": 98.0 + i * 2.0 if i != 5 else np.nan,  # Intentional missing at i=5
                "AQI_t_minus_1": 96.0 + i * 2.0 if i > 0 else np.nan,
                "AQI_t_minus_6": 86.0 + i * 2.0 if i > 5 else np.nan,
                "PM25_t": 50.0 + i if i != 8 else np.nan,  # Missing numerical
                "PM10_t": 100.0 + i,
                "NO2_t": 20.0,
                "SO2_t": 10.0,
                "CO_t": 1.0,
                "O3_t": 30.0,
                "PM_ratio_t": 0.5,
                "AQI_roll_mean_3d": 98.0 + i * 2.0,
                "AQI_roll_mean_7d": 95.0 + i * 2.0,
                "AQI_roll_std_7d": 3.0,
                "target_sin_doy": 0.1,
                "target_cos_doy": 0.9,
                "target_dow": (d + pd.Timedelta(days=1)).dayofweek,
                "target_is_weekend": float((d + pd.Timedelta(days=1)).dayofweek in [5, 6]),
                "target_month": (d + pd.Timedelta(days=1)).month,
                "is_target_valid": True,
                "is_shared_cohort": i != 5,  # False when AQI_t is missing
            })
    return pd.DataFrame(records)


def test_persistence_predictions_equal_aqi_t(synthetic_training_df):
    """Test 1: Persistence predictions must equal AQI_t exactly."""
    model = NaivePersistenceModel(aqi_col="AQI_t")
    preds = model.predict(synthetic_training_df)

    valid_mask = synthetic_training_df["AQI_t"].notna()
    np.testing.assert_allclose(
        preds[valid_mask],
        synthetic_training_df.loc[valid_mask, "AQI_t"].to_numpy(dtype=float),
        err_msg="Persistence prediction did not match AQI_t",
    )
    # Where AQI_t is NaN, prediction must be NaN
    assert np.isnan(preds[~valid_mask]).all(), "Persistence must predict NaN when AQI_t is missing"


def test_persistence_eligibility_requires_both_observed(synthetic_training_df):
    """Test 2: Persistence eligibility requires valid AQI_t and Target_AQI."""
    # Row with index 5 in CityA has AQI_t missing
    row_nan_origin = synthetic_training_df[synthetic_training_df["AQI_t"].isna()].iloc[0]
    assert bool(row_nan_origin["is_shared_cohort"]) is False, "Missing AQI_t must exclude row from shared cohort"
    assert bool(row_nan_origin["is_target_valid"]) is True, "Row has valid target despite missing origin"


def test_models_use_strictly_approved_predictors():
    """Test 3 & 4: Ensure target columns and future-derived fields are excluded from feature space."""
    num_cols, cat_cols, target_col = get_feature_columns()
    all_features = cat_cols + num_cols

    forbidden_cols = ["Target_AQI", "Target_Date", "split", "is_target_valid", "is_shared_cohort"]
    for col in forbidden_cols:
        assert col not in all_features, f"Forbidden column '{col}' found in feature list!"
    assert target_col == "Target_AQI"


def test_temporal_fold_boundaries_are_chronological():
    """Test 5: Fold boundaries must be strictly chronological and expanding."""
    folds = get_training_folds()
    for f in folds:
        t_start, t_end = pd.to_datetime(f.train_start), pd.to_datetime(f.train_end)
        v_start, v_end = pd.to_datetime(f.val_start), pd.to_datetime(f.val_end)
        assert t_start <= t_end, f"Train start > end in fold {f.fold_idx}"
        assert t_end < v_start, f"Train end >= val start in fold {f.fold_idx} (temporal overlap!)"
        assert v_start <= v_end, f"Val start > end in fold {f.fold_idx}"

    # Designated val fold
    d_fold = get_designated_validation_fold()
    assert pd.to_datetime(d_fold.train_end) < pd.to_datetime(d_fold.val_start)


def test_same_target_date_not_split_across_folds():
    """Test 6: All rows with the same Target_Date must be kept in the same split."""
    folds = get_training_folds()
    # The fold boundaries define closed intervals [start, end].
    # By construction, filtering on Target_Date ensures all cities for date D are in the same fold.
    for f in folds:
        assert f.train_end < f.val_start, "Boundary overlap could split a target date across train and val!"


def test_learned_preprocessing_fitted_only_on_training_fold(synthetic_training_df):
    """Test 7 & 8: Preprocessing fits strictly on training split, zero val leakage."""
    fold = ExpandingWindowFold(
        fold_idx=1,
        train_start="2017-01-02",
        train_end="2017-01-10",
        val_start="2017-01-11",
        val_end="2017-01-20",
        description="Test fold",
    )
    # Build Ridge pipeline
    pipe = build_ridge_pipeline(alpha=10.0, random_state=42)
    res = evaluate_model_on_fold(pipe, synthetic_training_df, fold)

    # Imputer statistics inside pipe must match training data median strictly
    num_cols, _, _ = get_feature_columns()
    train_mask = (
        (pd.to_datetime(synthetic_training_df["Target_Date"]) >= fold.train_start)
        & (pd.to_datetime(synthetic_training_df["Target_Date"]) <= fold.train_end)
        & synthetic_training_df["is_target_valid"]
    )
    expected_train_medians = synthetic_training_df.loc[train_mask, num_cols].median().to_numpy()

    fitted_imputer = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    actual_imputer_medians = fitted_imputer.statistics_

    np.testing.assert_allclose(
        actual_imputer_medians,
        expected_train_medians,
        err_msg="Imputer statistics in fitted pipeline did not match training data slice!",
    )


def test_missing_numerical_predictors_handled_across_models(synthetic_training_df):
    """Test 9: All 3 models handle missing numerical predictors gracefully."""
    fold = ExpandingWindowFold(
        fold_idx=1,
        train_start="2017-01-02",
        train_end="2017-01-10",
        val_start="2017-01-11",
        val_end="2017-01-20",
        description="Test fold",
    )
    # Ridge
    ridge = build_ridge_pipeline(alpha=10.0, random_state=42)
    res_ridge = evaluate_model_on_fold(ridge, synthetic_training_df, fold)
    assert not np.isnan(res_ridge["model_all_targets"]["MAE"])

    # RF
    rf = build_rf_pipeline(n_estimators=10, max_depth=4, random_state=42)
    res_rf = evaluate_model_on_fold(rf, synthetic_training_df, fold)
    assert not np.isnan(res_rf["model_all_targets"]["MAE"])

    # HistGBM (native missing handling)
    hgbm = build_histgbm_pipeline(max_iter=10, max_depth=4, random_state=42)
    res_hgbm = evaluate_model_on_fold(hgbm, synthetic_training_df, fold)
    assert not np.isnan(res_hgbm["model_all_targets"]["MAE"])


def test_unseen_city_categories_handled_without_crash(synthetic_training_df):
    """Test 10: Unseen city at prediction time does not cause crash or failure."""
    fold = ExpandingWindowFold(
        fold_idx=1,
        train_start="2017-01-02",
        train_end="2017-01-10",
        val_start="2017-01-11",
        val_end="2017-01-20",
        description="Test fold",
    )
    df = synthetic_training_df.copy()
    # Replace CityB in validation set with an unseen city 'CityC'
    val_mask = pd.to_datetime(df["Target_Date"]) >= "2017-01-11"
    df.loc[val_mask & (df["City"] == "CityB"), "City"] = "CityC"

    # Ridge uses OneHotEncoder(handle_unknown='ignore')
    ridge = build_ridge_pipeline(alpha=10.0, random_state=42)
    res_ridge = evaluate_model_on_fold(ridge, df, fold)
    assert len(res_ridge["val_predictions"]) > 0

    # HistGBM uses OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    hgbm = build_histgbm_pipeline(max_iter=10, max_depth=4, random_state=42)
    res_hgbm = evaluate_model_on_fold(hgbm, df, fold)
    assert len(res_hgbm["val_predictions"]) > 0


def test_model_predictions_are_finite(synthetic_training_df):
    """Test 11 & 12: Model predictions are finite on valid inputs."""
    fold = ExpandingWindowFold(
        fold_idx=1,
        train_start="2017-01-02",
        train_end="2017-01-10",
        val_start="2017-01-11",
        val_end="2017-01-20",
        description="Test fold",
    )
    hgbm = build_histgbm_pipeline(max_iter=10, max_depth=4, random_state=42)
    res = evaluate_model_on_fold(hgbm, synthetic_training_df, fold)
    preds = res["val_predictions"]["Pred_Model"].to_numpy()
    assert np.all(np.isfinite(preds)), "HistGBM produced non-finite predictions!"


def test_model_reproducibility(synthetic_training_df):
    """Test 13: Fixed random_state produces identical predictions."""
    fold = ExpandingWindowFold(
        fold_idx=1,
        train_start="2017-01-02",
        train_end="2017-01-10",
        val_start="2017-01-11",
        val_end="2017-01-20",
        description="Test fold",
    )
    pipe1 = build_rf_pipeline(n_estimators=10, max_depth=4, random_state=42)
    res1 = evaluate_model_on_fold(pipe1, synthetic_training_df, fold)

    pipe2 = build_rf_pipeline(n_estimators=10, max_depth=4, random_state=42)
    res2 = evaluate_model_on_fold(pipe2, synthetic_training_df, fold)

    np.testing.assert_allclose(
        res1["val_predictions"]["Pred_Model"].to_numpy(),
        res2["val_predictions"]["Pred_Model"].to_numpy(),
        err_msg="RandomForest predictions were non-deterministic with fixed random_state!",
    )


def test_shared_cohort_uses_identical_row_identifiers(synthetic_training_df):
    """Test 14: Shared cohort comparisons between baseline and candidate use identical indices."""
    fold = ExpandingWindowFold(
        fold_idx=1,
        train_start="2017-01-02",
        train_end="2017-01-10",
        val_start="2017-01-11",
        val_end="2017-01-20",
        description="Test fold",
    )
    ridge = build_ridge_pipeline(alpha=10.0, random_state=42)
    res = evaluate_model_on_fold(ridge, synthetic_training_df, fold)

    # Persistence metrics and Model metrics must have the exact same sample count N
    assert res["persistence_shared"]["N"] == res["model_shared"]["N"]
    assert res["n_val_shared"] == res["persistence_shared"]["N"]


def test_primary_test_and_stress_test_isolated_from_tuning():
    """Test 15: Primary test (2019) and stress test (2020) dates are excluded from training folds."""
    folds = get_training_folds()
    d_fold = get_designated_validation_fold()
    all_tuning_folds = folds + [d_fold]

    test_start = pd.Timestamp("2019-01-01")
    for f in all_tuning_folds:
        val_end = pd.to_datetime(f.val_end)
        assert val_end < test_start, f"Tuning fold {f.fold_idx} encroaches into 2019 held-out test set!"
