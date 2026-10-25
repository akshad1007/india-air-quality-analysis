"""
Phase 9.1 Automated Unit and Integration Tests for SHAP Interpretability.

Verifies:
1. Correct loading of the frozen Random Forest model artifact.
2. Correct feature names and feature ordering (19 raw inputs -> 36 transformed features).
3. Reproducible background sampling strictly from the historical training period (<= 2018-06-30).
4. Reproducible selection of Typical (Moderate AQI) and Extreme (Severe AQI) cases.
5. Strict exclusion of target and future-derived columns from model inputs (no leakage).
6. Correct SHAP explanation output shape (N, 36).
7. SHAP additive reconstruction within the declared tolerance (f(x) = phi_0 + sum(phi_i)).
8. Consistency between SHAP reconstruction and direct pipeline predictions.
9. Handling of missing numerical input predictors according to project conventions.
10. Confirmation that model predictions and fitted estimators remain strictly unchanged.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from src.features.feature_builder import PROCESSED_DIR, get_feature_columns
from src.models.interpretability import (
    DEFAULT_ADDITIVITY_TOLERANCE,
    build_tree_explainer,
    compute_feature_importance_table,
    compute_shap_explanation,
    get_transformed_feature_names,
    load_frozen_pipeline,
    sample_background_dataset,
    select_extreme_case,
    select_typical_case,
    verify_shap_additivity,
)

MODELS_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\models")
REPORTS_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\reports")


@pytest.fixture(scope="module")
def feature_df() -> pd.DataFrame:
    """Loads processed feature dataset."""
    path = PROCESSED_DIR / "city_day_features.parquet"
    assert path.exists(), f"Missing feature dataset: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def frozen_pipeline() -> Pipeline:
    """Loads the frozen Random Forest pipeline."""
    model_path = MODELS_DIR / "random_forest_val_model.joblib"
    assert model_path.exists(), f"Missing frozen model: {model_path}"
    return load_frozen_pipeline(model_path)


def test_load_frozen_pipeline(frozen_pipeline):
    """Test 1: Frozen model loads correctly and preserves fitted architecture."""
    assert isinstance(frozen_pipeline, Pipeline)
    assert "preprocessor" in frozen_pipeline.named_steps
    assert "regressor" in frozen_pipeline.named_steps

    preprocessor = frozen_pipeline.named_steps["preprocessor"]
    regressor = frozen_pipeline.named_steps["regressor"]

    assert isinstance(preprocessor, ColumnTransformer)
    assert isinstance(regressor, RandomForestRegressor)
    assert regressor.n_estimators == 100
    assert regressor.max_depth == 10
    assert regressor.random_state == 42


def test_feature_names_and_ordering(frozen_pipeline):
    """Test 2: Authoritative feature schema matches regressor input dimension (36 features)."""
    preprocessor = frozen_pipeline.named_steps["preprocessor"]
    regressor = frozen_pipeline.named_steps["regressor"]

    num_cols, cat_cols, target_col = get_feature_columns()
    assert cat_cols == ["City"]
    assert len(num_cols) == 18
    assert target_col == "Target_AQI"

    raw_features = get_transformed_feature_names(preprocessor, clean=False)
    clean_features = get_transformed_feature_names(preprocessor, clean=True)

    assert len(raw_features) == 36
    assert len(clean_features) == 36
    assert regressor.n_features_in_ == 36

    # Verify first 18 features are OneHot City indicators
    onehot_cities = [f for f in raw_features if f.startswith("cat__City_")]
    assert len(onehot_cities) == 18

    # Verify remaining 18 features are numerical predictors
    numeric_features = [f for f in raw_features if f.startswith("num__")]
    assert len(numeric_features) == 18


def test_reproducible_background_sampling(feature_df):
    """Test 3: Background dataset is sampled deterministically and exclusively from training split."""
    sample1, raw1 = sample_background_dataset(feature_df, n_samples=100, random_state=42)
    sample2, raw2 = sample_background_dataset(feature_df, n_samples=100, random_state=42)

    pd.testing.assert_frame_equal(sample1, sample2)
    pd.testing.assert_frame_equal(raw1, raw2)

    assert len(sample1) == 100
    # Every sampled row must be from historical training period (<= 2018-06-30)
    assert (sample1["split"] == "train").all()
    assert (sample1["Target_Date"] <= pd.Timestamp("2018-06-30")).all()

    # Quarantined partitions must never appear in background sample
    assert not (sample1["split"] == "test").any()
    assert not (sample1["split"] == "stress").any()


def test_reproducible_selection_typical_and_extreme_cases(feature_df):
    """Test 4: Typical and Extreme cases are selected reproducibly by documented rules."""
    typ_row1, typ_idx1 = select_typical_case(feature_df)
    typ_row2, typ_idx2 = select_typical_case(feature_df)
    assert typ_idx1 == typ_idx2
    assert typ_row1["City"] == "Gurugram"
    assert typ_row1["Target_Date"] == pd.Timestamp("2018-07-31")
    assert typ_row1["Target_AQI"] == 112.0
    # Moderate category check (100 < y <= 200)
    assert 100.0 < typ_row1["Target_AQI"] <= 200.0

    ext_row1, ext_idx1 = select_extreme_case(feature_df)
    ext_row2, ext_idx2 = select_extreme_case(feature_df)
    assert ext_idx1 == ext_idx2
    assert ext_row1["City"] == "Ahmedabad"
    assert ext_row1["Target_Date"] == pd.Timestamp("2018-11-18")
    assert ext_row1["Target_AQI"] == 1672.0
    # Severe category check (> 400)
    assert ext_row1["Target_AQI"] > 400.0


def test_no_target_or_future_leakage_in_inputs(feature_df):
    """Test 5: Model explanation inputs strictly exclude target and non-predictive columns."""
    num_cols, cat_cols, target_col = get_feature_columns()
    fcols = cat_cols + num_cols

    forbidden = [
        "Target_AQI",
        "Target_Date",
        "split",
        "is_shared_cohort",
        "is_target_valid",
        "is_training_eligible",
        "is_extreme_event",
    ]

    for col in forbidden:
        assert col not in fcols, f"Forbidden column {col} found in feature columns"


def test_shap_output_shape(frozen_pipeline, feature_df):
    """Test 6: SHAP explanation values matrix has exact dimensions (N, 36)."""
    preprocessor = frozen_pipeline.named_steps["preprocessor"]
    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols

    val_sub = feature_df[
        (feature_df["split"] == "val") & feature_df["is_target_valid"] & feature_df["is_shared_cohort"]
    ].head(15)

    X_trans = preprocessor.transform(val_sub[fcols])
    clean_names = get_transformed_feature_names(preprocessor, clean=True)

    explainer = build_tree_explainer(frozen_pipeline, feature_perturbation="tree_path_dependent")
    sv = compute_shap_explanation(explainer, X_trans, clean_names, check_additivity=True)

    assert sv.values.shape == (15, 36)
    assert len(sv.feature_names) == 36


def test_shap_additive_reconstruction_within_tolerance(frozen_pipeline, feature_df):
    """Test 7: SHAP additive equation f(x) = phi_0 + sum(phi_i) holds within numerical tolerance."""
    preprocessor = frozen_pipeline.named_steps["preprocessor"]
    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols

    val_sub = feature_df[
        (feature_df["split"] == "val") & feature_df["is_target_valid"] & feature_df["is_shared_cohort"]
    ].sample(50, random_state=42)

    X_trans = preprocessor.transform(val_sub[fcols])
    clean_names = get_transformed_feature_names(preprocessor, clean=True)

    explainer = build_tree_explainer(frozen_pipeline, feature_perturbation="tree_path_dependent")
    sv = compute_shap_explanation(explainer, X_trans, clean_names, check_additivity=True)

    preds = frozen_pipeline.predict(val_sub[fcols])
    audit = verify_shap_additivity(explainer, sv, preds, tolerance=DEFAULT_ADDITIVITY_TOLERANCE)

    assert audit["passed"] is True
    assert audit["max_reconstruction_error"] < 1e-6, (
        f"Path-dependent TreeSHAP error {audit['max_reconstruction_error']} exceeded 1e-6"
    )


def test_consistency_between_shap_reconstruction_and_direct_predictions(frozen_pipeline, feature_df):
    """Test 8: Reconstructed SHAP predictions match direct pipeline predictions identically."""
    preprocessor = frozen_pipeline.named_steps["preprocessor"]
    regressor = frozen_pipeline.named_steps["regressor"]
    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols

    typ_row, _ = select_typical_case(feature_df)
    ext_row, _ = select_extreme_case(feature_df)
    test_cases = pd.DataFrame([typ_row[fcols], ext_row[fcols]])

    pipeline_preds = frozen_pipeline.predict(test_cases)
    direct_preds = regressor.predict(preprocessor.transform(test_cases))
    np.testing.assert_allclose(pipeline_preds, direct_preds, rtol=1e-7, atol=1e-7)

    explainer = build_tree_explainer(frozen_pipeline, feature_perturbation="tree_path_dependent")
    sv = compute_shap_explanation(
        explainer,
        preprocessor.transform(test_cases),
        get_transformed_feature_names(preprocessor, clean=True),
    )

    base = explainer.expected_value
    base_val = float(base[0]) if isinstance(base, (list, np.ndarray)) else float(base)
    reconstructed = base_val + np.sum(sv.values, axis=1)

    np.testing.assert_allclose(reconstructed, pipeline_preds, rtol=1e-7, atol=1e-7)


def test_handling_of_missing_numerical_inputs(frozen_pipeline):
    """Test 9: Inputs with missing numerical predictors are imputed without crash."""
    preprocessor = frozen_pipeline.named_steps["preprocessor"]
    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols

    # Synthesize row with intentional NaNs in chemical predictors
    synth_dict = {"City": "Delhi"}
    for c in num_cols:
        synth_dict[c] = np.nan
    # Give at least AQI_t a value
    synth_dict["AQI_t"] = 150.0
    synth_df = pd.DataFrame([synth_dict])

    pred = frozen_pipeline.predict(synth_df)
    assert np.isfinite(pred[0]), "Prediction must be finite even with missing numerical features"

    explainer = build_tree_explainer(frozen_pipeline, feature_perturbation="tree_path_dependent")
    sv = compute_shap_explanation(
        explainer,
        preprocessor.transform(synth_df),
        get_transformed_feature_names(preprocessor, clean=True),
    )
    assert np.all(np.isfinite(sv.values)), "SHAP values must be finite for imputed inputs"


def test_model_predictions_remain_strictly_unchanged(frozen_pipeline, feature_df):
    """Test 10: Model predictions remain strictly identical to verified Phase 8 values."""
    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols

    typ_row, _ = select_typical_case(feature_df)
    ext_row, _ = select_extreme_case(feature_df)

    pred_typ = frozen_pipeline.predict(pd.DataFrame([typ_row[fcols]]))[0]
    pred_ext = frozen_pipeline.predict(pd.DataFrame([ext_row[fcols]]))[0]

    # Ground truth values established from frozen model
    expected_typ = 115.45455813457224
    expected_ext = 1062.6817804309221

    assert abs(pred_typ - expected_typ) < 1e-5, f"Typical pred changed: {pred_typ} vs {expected_typ}"
    assert abs(pred_ext - expected_ext) < 1e-5, f"Extreme pred changed: {pred_ext} vs {expected_ext}"
