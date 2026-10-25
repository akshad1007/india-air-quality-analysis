"""
Phase 9.1 — SHAP Explainability Engine.

Provides reproducible, scientifically grounded local and global interpretability
for the frozen Random Forest next-day AQI forecasting champion model.

Scientific Governance & Constraints:
1. Model Immutability: The frozen Random Forest pipeline (models/random_forest_val_model.joblib)
   is loaded in strictly read-only mode. No retraining, parameter modification, or refitting occurs.
2. Chronological Data Barrier: Background reference distributions are sampled strictly from
   historical training observations (Target_Date <= 2018-06-30). Held-out test (2019) and
   stress (2020) data are strictly quarantined and never used to construct explanations.
3. Feature Lineage & Leakage Prevention: Inputs X consist strictly of the 19 approved predictors
   (City + 18 chemical, lag, and calendar features). Target_AQI, Target_Date, and future fields
   are strictly excluded.
4. Non-Causal Semantics: SHAP attributions quantify model prediction mechanics (how features
   shift the model's conditional prediction from baseline), NOT physical pollutant causation
   or atmospheric chemical reactions.
5. Mathematical Additivity: Verified via f(x) = phi_0 + sum(phi_i) within justified numerical tolerance.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from src.features.feature_builder import PROCESSED_DIR, get_feature_columns

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Default tolerance for SHAP additive reconstruction: 0.5 AQI points (< 0.1% of 500 AQI scale)
DEFAULT_ADDITIVITY_TOLERANCE = 0.5


def load_frozen_pipeline(model_path: Optional[Path] = None) -> Pipeline:
    """
    Loads and validates the frozen Random Forest forecasting pipeline.

    Args:
        model_path: Path to the serialized joblib artifact. Defaults to models/random_forest_val_model.joblib.

    Returns:
        Fitted sklearn.pipeline.Pipeline containing 'preprocessor' and 'regressor'.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
        TypeError: If the loaded object is not a Pipeline with expected named steps.
    """
    path = model_path or (MODELS_DIR / "random_forest_val_model.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Frozen Random Forest artifact not found: {path}")

    pipeline = joblib.load(path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected sklearn.pipeline.Pipeline, got {type(pipeline)}")

    if "preprocessor" not in pipeline.named_steps or "regressor" not in pipeline.named_steps:
        raise TypeError("Pipeline must contain named steps 'preprocessor' and 'regressor'")

    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]

    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(f"Pipeline preprocessor must be ColumnTransformer, got {type(preprocessor)}")
    if not isinstance(regressor, RandomForestRegressor):
        raise TypeError(f"Pipeline regressor must be RandomForestRegressor, got {type(regressor)}")

    return pipeline


def get_transformed_feature_names(preprocessor: ColumnTransformer, clean: bool = True) -> List[str]:
    """
    Extracts authoritative output feature names from the fitted preprocessor.

    Args:
        preprocessor: Fitted ColumnTransformer.
        clean: If True, strips 'cat__' and 'num__' prefixes for publication readability.

    Returns:
        List of feature names matching the regressor input dimensions (36 features).
    """
    raw_names = list(preprocessor.get_feature_names_out())
    if clean:
        return [name.replace("num__", "").replace("cat__", "") for name in raw_names]
    return raw_names


def sample_background_dataset(
    df: pd.DataFrame,
    n_samples: int = 100,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extracts a reproducible background sample strictly from the historical training partition.

    Constraints:
    - Partition: Training period only (split == 'train', Target_Date <= 2018-06-30).
    - Eligibility: is_target_valid == True, is_shared_cohort == True.
    - Zero Leakage: Strictly extracts predictor columns (cat_cols + num_cols);
      Target_AQI and target dates are excluded.

    Args:
        df: Processed feature DataFrame (e.g. from city_day_features.parquet).
        n_samples: Number of background rows to sample (default: 100).
        random_state: Fixed random seed for deterministic sampling.

    Returns:
        Tuple of (raw_sample_df, raw_sample_features_df).
    """
    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols

    train_mask = (df["split"] == "train") & df["is_target_valid"] & df["is_shared_cohort"]
    train_pool = df[train_mask].sort_values(["Target_Date", "City"]).reset_index(drop=True)

    if len(train_pool) == 0:
        raise ValueError("No eligible training records found for background sampling.")

    sample_size = min(n_samples, len(train_pool))
    sampled_df = train_pool.sample(n=sample_size, random_state=random_state).sort_values(
        ["Target_Date", "City"]
    ).reset_index(drop=True)

    X_bg_raw = sampled_df[fcols].copy()
    return sampled_df, X_bg_raw


def build_tree_explainer(
    pipeline: Pipeline,
    background_transformed: Optional[np.ndarray] = None,
    feature_perturbation: str = "tree_path_dependent",
) -> shap.TreeExplainer:
    """
    Initializes a SHAP TreeExplainer for the Random Forest regressor.

    Args:
        pipeline: Loaded frozen pipeline.
        background_transformed: Optional transformed background matrix.
        feature_perturbation: 'tree_path_dependent' (exact, uses internal tree weights)
                             or 'interventional' (requires background data).

    Returns:
        shap.TreeExplainer instance.
    """
    regressor = pipeline.named_steps["regressor"]
    if feature_perturbation == "tree_path_dependent":
        return shap.TreeExplainer(regressor, feature_perturbation="tree_path_dependent")
    elif feature_perturbation == "interventional":
        if background_transformed is None:
            raise ValueError("background_transformed matrix must be provided for interventional TreeExplainer")
        return shap.TreeExplainer(regressor, data=background_transformed, feature_perturbation="interventional")
    else:
        raise ValueError(f"Unknown feature_perturbation: {feature_perturbation}")


def compute_shap_explanation(
    explainer: shap.TreeExplainer,
    transformed_matrix: np.ndarray,
    feature_names: List[str],
    check_additivity: bool = True,
) -> shap.Explanation:
    """
    Computes SHAP values and wraps them in a shap.Explanation object with authoritative feature names.

    Args:
        explainer: Fitted TreeExplainer.
        transformed_matrix: Array of shape (N, 36) output by pipeline.preprocessor.
        feature_names: List of 36 feature names.
        check_additivity: Whether to enforce SHAP internal additivity check.

    Returns:
        shap.Explanation object.
    """
    sv = explainer(transformed_matrix, check_additivity=check_additivity)
    sv.feature_names = list(feature_names)
    return sv


def verify_shap_additivity(
    explainer: shap.TreeExplainer,
    shap_explanation: shap.Explanation,
    predictions: np.ndarray,
    tolerance: float = DEFAULT_ADDITIVITY_TOLERANCE,
) -> Dict[str, Any]:
    """
    Mathematically verifies the additive attribution relationship:
        f(x) ≈ phi_0 + sum_{i=1}^p phi_i

    Args:
        explainer: TreeExplainer providing expected_value (baseline).
        shap_explanation: Computed Explanation object.
        predictions: Model predictions from pipeline.predict(X_raw).
        tolerance: Allowed maximum absolute difference in AQI points.

    Returns:
        Dictionary containing additivity verification metrics.
    """
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[0])
    else:
        base_val = float(base_val)

    shap_sums = np.sum(shap_explanation.values, axis=1)
    reconstructed = base_val + shap_sums
    abs_errors = np.abs(reconstructed - np.asarray(predictions, dtype=float))

    max_error = float(np.max(abs_errors))
    mean_error = float(np.mean(abs_errors))
    passed = bool(max_error <= tolerance)

    return {
        "passed": passed,
        "max_reconstruction_error": max_error,
        "mean_reconstruction_error": mean_error,
        "declared_tolerance": float(tolerance),
        "baseline_expected_value": base_val,
        "n_evaluated": int(len(predictions)),
    }


def compute_feature_importance_table(
    shap_explanation: shap.Explanation,
    feature_names: List[str],
    raw_feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes global feature importance as mean absolute SHAP value:
        I_j = (1 / N) * sum_{k=1}^N |phi_{k, j}|

    Args:
        shap_explanation: Computed Explanation.
        feature_names: Clean feature names.
        raw_feature_names: Optional raw preprocessor feature names.

    Returns:
        pd.DataFrame sorted descending by mean_abs_shap with 1-based rank.
    """
    mean_abs = np.mean(np.abs(shap_explanation.values), axis=0)
    raw_names = raw_feature_names or feature_names

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "raw_feature_name": raw_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    imp_df["rank"] = imp_df.index + 1
    return imp_df[["rank", "feature", "raw_feature_name", "mean_abs_shap"]]


def select_typical_case(df: pd.DataFrame) -> Tuple[pd.Series, int]:
    """
    Selects a reproducible Typical Case from the pre-2019 validation partition.

    Rule:
    1. Filter: split == 'val', is_target_valid == True, is_shared_cohort == True.
    2. CPCB Category: Moderate hazard band (100 < Target_AQI <= 200).
    3. Representative target: Minimizes absolute distance to the median observed validation AQI (112.0 AQI points).
    4. Deterministic tie-breaking: Earliest Target_Date, then alphabetical City.

    Returns:
        Tuple of (selected_row_series, row_index_in_sorted_val).
    """
    val = df[(df["split"] == "val") & df["is_target_valid"] & df["is_shared_cohort"]].sort_values(
        ["Target_Date", "City"]
    ).reset_index(drop=True)

    if len(val) == 0:
        raise ValueError("No eligible validation rows available.")

    med_aqi = float(val["Target_AQI"].median())
    mod_mask = (val["Target_AQI"] > 100.0) & (val["Target_AQI"] <= 200.0)
    mod_df = val[mod_mask].copy()

    if len(mod_df) == 0:
        raise ValueError("No Moderate category observations found in validation set.")

    diffs = (mod_df["Target_AQI"] - med_aqi).abs()
    min_diff = diffs.min()
    best_candidates = mod_df[diffs == min_diff].sort_values(["Target_Date", "City"])
    typical_idx = int(best_candidates.index[0])
    return val.iloc[typical_idx], typical_idx


def select_extreme_case(df: pd.DataFrame) -> Tuple[pd.Series, int]:
    """
    Selects a reproducible Extreme Case from the pre-2019 validation partition.

    Rule:
    1. Filter: split == 'val', is_target_valid == True, is_shared_cohort == True.
    2. Representative target: Maximizes observed Target_AQI.
    3. Deterministic tie-breaking: Earliest Target_Date, then alphabetical City.

    Returns:
        Tuple of (selected_row_series, row_index_in_sorted_val).
    """
    val = df[(df["split"] == "val") & df["is_target_valid"] & df["is_shared_cohort"]].sort_values(
        ["Target_Date", "City"]
    ).reset_index(drop=True)

    if len(val) == 0:
        raise ValueError("No eligible validation rows available.")

    max_aqi = val["Target_AQI"].max()
    best_candidates = val[val["Target_AQI"] == max_aqi].sort_values(["Target_Date", "City"])
    extreme_idx = int(best_candidates.index[0])
    return val.iloc[extreme_idx], extreme_idx


def generate_interpretability_plots(
    shap_explanation: shap.Explanation,
    imp_df: pd.DataFrame,
    typical_idx: int,
    extreme_idx: int,
    typical_meta: Dict[str, Any],
    extreme_meta: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Generates and saves the 4 publication-quality SHAP visualization figures.

    Figures generated:
    1. shap_summary.png: SHAP beeswarm summary plot.
    2. shap_importance.png: Horizontal bar chart of Top 20 features by mean |SHAP|.
    3. shap_waterfall_typical.png: Waterfall plot for the Typical case.
    4. shap_waterfall_extreme.png: Waterfall plot for the Extreme case.

    Args:
        shap_explanation: Evaluated Explanation object on validation set.
        imp_df: Feature importance DataFrame.
        typical_idx: Row index of Typical case.
        extreme_idx: Row index of Extreme case.
        typical_meta: Metadata dict for Typical case (City, dates, AQI, pred).
        extreme_meta: Metadata dict for Extreme case (City, dates, AQI, pred).
        output_dir: Directory to save figures. Defaults to reports/figures/.

    Returns:
        Dictionary mapping figure names to saved Paths.
    """
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    # 1. Beeswarm Summary Plot
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
    plt.title(
        "SHAP Feature Attribution Summary (Validation Set, N=2,982)\nRandom Forest Next-Day AQI Forecaster",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("SHAP Value (Impact on Model AQI Prediction, continuous AQI points)", fontsize=10)
    plt.tight_layout()
    p_summary = out_dir / "shap_summary.png"
    plt.savefig(p_summary, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths["shap_summary"] = p_summary

    # 2. Bar Importance Plot
    plt.figure(figsize=(10, 7))
    top20 = imp_df.head(20).iloc[::-1]
    plt.barh(top20["feature"], top20["mean_abs_shap"], color="#1f77b4", edgecolor="#0d47a1", alpha=0.85)
    plt.title(
        "Global Feature Importance: Mean |SHAP| Value\nTop 20 Features (Validation Set, N=2,982)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Mean |SHAP Value| (AQI Points)", fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p_imp = out_dir / "shap_importance.png"
    plt.savefig(p_imp, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths["shap_importance"] = p_imp

    # 3. Waterfall Typical Case
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_explanation[typical_idx], max_display=12, show=False)
    plt.title(
        f"SHAP Local Explanation: Typical Case (Moderate AQI)\n"
        f"City: {typical_meta['city']} | Target Date: {typical_meta['target_date']} | "
        f"Observed AQI: {typical_meta['observed_aqi']:.2f} | Pred: {typical_meta['predicted_aqi']:.2f}",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    p_water_typ = out_dir / "shap_waterfall_typical.png"
    plt.savefig(p_water_typ, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths["shap_waterfall_typical"] = p_water_typ

    # 4. Waterfall Extreme Case
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_explanation[extreme_idx], max_display=12, show=False)
    plt.title(
        f"SHAP Local Explanation: Extreme Case (Severe AQI)\n"
        f"City: {extreme_meta['city']} | Target Date: {extreme_meta['target_date']} | "
        f"Observed AQI: {extreme_meta['observed_aqi']:.2f} | Pred: {extreme_meta['predicted_aqi']:.2f}",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    p_water_ext = out_dir / "shap_waterfall_extreme.png"
    plt.savefig(p_water_ext, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths["shap_waterfall_extreme"] = p_water_ext

    return saved_paths


def run_shap_interpretability(
    dataset_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Executes the complete Milestone 9.1 SHAP Explainability Engine.

    Workflow:
    1. Load frozen Random Forest pipeline.
    2. Load feature dataset and slice validation cohort (N=2,982).
    3. Sample reproducible background dataset from training period (N=100, seed=42).
    4. Compute TreeSHAP attributions using path-dependent explainer.
    5. Verify mathematical additivity within 0.5 AQI points tolerance.
    6. Extract Top 5 global features and save shap_feature_importance.csv.
    7. Select typical and extreme cases and compute local waterfall metrics.
    8. Generate 4 publication figures.

    Returns:
        Structured audit dictionary summarizing execution results.
    """
    feat_path = dataset_path or (PROCESSED_DIR / "city_day_features.parquet")
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {feat_path}")

    logger.info("Loading feature dataset...")
    df = pd.read_parquet(feat_path)

    logger.info("Loading frozen Random Forest pipeline...")
    pipeline = load_frozen_pipeline(model_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]

    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols
    feature_names = get_transformed_feature_names(preprocessor, clean=True)
    raw_feature_names = get_transformed_feature_names(preprocessor, clean=False)

    # Background sample from training partition (Target_Date <= 2018-06-30)
    logger.info("Sampling reproducible background dataset from training period...")
    bg_df, bg_X_raw = sample_background_dataset(df, n_samples=100, random_state=42)
    bg_transformed = preprocessor.transform(bg_X_raw)

    # Slice validation cohort
    val_df = df[(df["split"] == "val") & df["is_target_valid"] & df["is_shared_cohort"]].sort_values(
        ["Target_Date", "City"]
    ).reset_index(drop=True)

    val_X_raw = val_df[fcols].copy()
    val_transformed = preprocessor.transform(val_X_raw)
    val_preds = pipeline.predict(val_X_raw)

    # Build TreeExplainer
    logger.info("Computing SHAP values for validation cohort...")
    explainer = build_tree_explainer(pipeline, feature_perturbation="tree_path_dependent")
    shap_explanation = compute_shap_explanation(
        explainer, val_transformed, feature_names, check_additivity=True
    )

    # Verify additivity
    additivity_audit = verify_shap_additivity(
        explainer, shap_explanation, val_preds, tolerance=DEFAULT_ADDITIVITY_TOLERANCE
    )
    logger.info(
        f"Additivity verification: passed={additivity_audit['passed']}, "
        f"max_error={additivity_audit['max_reconstruction_error']:.2e}"
    )

    # Compute global feature importance
    imp_df = compute_feature_importance_table(
        shap_explanation, feature_names, raw_feature_names
    )

    # Select local cases
    typ_row, typ_idx = select_typical_case(df)
    ext_row, ext_idx = select_extreme_case(df)

    typ_pred = float(pipeline.predict(pd.DataFrame([typ_row[fcols]]))[0])
    ext_pred = float(pipeline.predict(pd.DataFrame([ext_row[fcols]]))[0])

    typ_meta = {
        "city": typ_row["City"],
        "date": typ_row["Date"].strftime("%Y-%m-%d"),
        "target_date": typ_row["Target_Date"].strftime("%Y-%m-%d"),
        "observed_aqi": float(typ_row["Target_AQI"]),
        "predicted_aqi": typ_pred,
        "residual": float(typ_row["Target_AQI"] - typ_pred),
    }

    ext_meta = {
        "city": ext_row["City"],
        "date": ext_row["Date"].strftime("%Y-%m-%d"),
        "target_date": ext_row["Target_Date"].strftime("%Y-%m-%d"),
        "observed_aqi": float(ext_row["Target_AQI"]),
        "predicted_aqi": ext_pred,
        "residual": float(ext_row["Target_AQI"] - ext_pred),
    }

    # Save artifacts if requested
    saved_files = {}
    if save_artifacts:
        rep_dir = output_dir or REPORTS_DIR
        rep_dir.mkdir(parents=True, exist_ok=True)

        csv_path = rep_dir / "shap_feature_importance.csv"
        imp_df.to_csv(csv_path, index=False)
        saved_files["feature_importance_csv"] = csv_path

        plot_paths = generate_interpretability_plots(
            shap_explanation,
            imp_df,
            typ_idx,
            ext_idx,
            typ_meta,
            ext_meta,
            output_dir=rep_dir / "figures",
        )
        saved_files.update(plot_paths)

    return {
        "model_artifact": str(model_path or (MODELS_DIR / "random_forest_val_model.joblib")),
        "background_sample_size": len(bg_df),
        "background_date_range": (
            bg_df["Target_Date"].min().strftime("%Y-%m-%d"),
            bg_df["Target_Date"].max().strftime("%Y-%m-%d"),
        ),
        "validation_sample_size": len(val_df),
        "top_5_features": imp_df.head(5)[["rank", "feature", "mean_abs_shap"]].to_dict(orient="records"),
        "additivity_audit": additivity_audit,
        "typical_case": typ_meta,
        "extreme_case": ext_meta,
        "saved_artifacts": {k: str(v) for k, v in saved_files.items()},
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Executing Phase 9.1 SHAP Explainability Engine...")
    results = run_shap_interpretability(save_artifacts=True)
    print("\nExecution Summary:")
    print(f"Model Artifact:          {results['model_artifact']}")
    print(f"Background Sample:       N={results['background_sample_size']} ({results['background_date_range'][0]} to {results['background_date_range'][1]})")
    print(f"Validation Sample:       N={results['validation_sample_size']}")
    print(f"Additivity Check Passed: {results['additivity_audit']['passed']} (Max Error: {results['additivity_audit']['max_reconstruction_error']:.2e} AQI points)")
    print("\nTop 5 Global Features:")
    for f in results["top_5_features"]:
        print(f"  Rank {f['rank']}: {f['feature']:<20} (Mean |SHAP|: {f['mean_abs_shap']:.4f} AQI points)")
    print(f"\nTypical Case: {results['typical_case']['city']} on {results['typical_case']['target_date']} | Obs: {results['typical_case']['observed_aqi']} | Pred: {results['typical_case']['predicted_aqi']:.2f}")
    print(f"Extreme Case: {results['extreme_case']['city']} on {results['extreme_case']['target_date']} | Obs: {results['extreme_case']['observed_aqi']} | Pred: {results['extreme_case']['predicted_aqi']:.2f}")
    print("\nSaved Artifacts:")
    for k, v in results["saved_artifacts"].items():
        print(f"  {k}: {v}")
