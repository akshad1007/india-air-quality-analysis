"""
Phase 8 Final Model Evaluation, CPCB Diagnostics & Residual Analysis.

Enforces:
1. Frozen Phase 7 model configurations.
2. Strict separation of 2019 Primary Test and 2020 Stress Test.
3. Two distinct cohorts: Cohort A (Shared Persistence) vs Cohort B (Model-Only).
4. Official CPCB statutory AQI category mapping:
   - Good: [0, 50]
   - Satisfactory: (50, 100]
   - Moderate: (100, 200]
   - Poor: (200, 300]
   - Very Poor: (300, 400]
   - Severe: > 400
5. Residual sign convention: e_i = y_i - y_hat_i (positive = underprediction, negative = overprediction).
6. Comprehensive sensitivity analysis with and without the Ahmedabad CO anomaly.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.features.feature_builder import PROCESSED_DIR, get_feature_columns
from src.models.baseline import NaivePersistenceModel
from src.models.metrics import compute_regression_metrics, compute_relative_improvement

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\reports")
MODELS_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\models")

CPCB_CATEGORIES = [
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe",
]


def map_aqi_to_cpcb_bucket(aqi_values: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
    """
    Maps continuous AQI values into the 6 statutory CPCB hazard categories.
    Boundary Rules:
    - y <= 50: Good
    - 50 < y <= 100: Satisfactory
    - 100 < y <= 200: Moderate
    - 200 < y <= 300: Poor
    - 300 < y <= 400: Very Poor
    - y > 400: Severe

    Out-of-range Handling:
    - y < 0 is mapped to 'Good' (represents clean air prediction below zero threshold).
    - y > 500 is mapped to 'Severe' (statutory hazardous ceiling).
    - NaN is preserved as None.
    """
    arr = np.asarray(aqi_values, dtype=float)
    result = np.empty(arr.shape, dtype=object)

    nan_mask = np.isnan(arr)
    result[nan_mask] = None

    valid_mask = ~nan_mask
    val = arr[valid_mask]

    categories = np.empty(val.shape, dtype=object)
    categories[val <= 50.0] = "Good"
    categories[(val > 50.0) & (val <= 100.0)] = "Satisfactory"
    categories[(val > 100.0) & (val <= 200.0)] = "Moderate"
    categories[(val > 200.0) & (val <= 300.0)] = "Poor"
    categories[(val > 300.0) & (val <= 400.0)] = "Very Poor"
    categories[val > 400.0] = "Severe"

    result[valid_mask] = categories
    return result


def compute_classification_diagnostics(
    y_true_cont: Union[np.ndarray, pd.Series],
    y_pred_cont: Union[np.ndarray, pd.Series],
) -> Dict[str, Any]:
    """
    Maps continuous true and predicted AQI values to CPCB hazard categories and
    computes multi-class classification diagnostic metrics.
    """
    y_t_arr = np.asarray(y_true_cont, dtype=float)
    y_p_arr = np.asarray(y_pred_cont, dtype=float)

    valid_mask = (~np.isnan(y_t_arr)) & (~np.isnan(y_p_arr))
    if np.sum(valid_mask) == 0:
        return {
            "N": 0,
            "macro_f1": np.nan,
            "balanced_accuracy": np.nan,
            "confusion_matrix": [],
            "per_category": {},
        }

    y_true_cat = map_aqi_to_cpcb_bucket(y_t_arr[valid_mask])
    y_pred_cat = map_aqi_to_cpcb_bucket(y_p_arr[valid_mask])

    # Confusion matrix with fixed 6x6 labels
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=CPCB_CATEGORIES)

    # Per-category metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_cat, y_pred_cat, labels=CPCB_CATEGORIES, zero_division=0
    )

    macro_f1 = float(f1_score(y_true_cat, y_pred_cat, labels=CPCB_CATEGORIES, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true_cat, y_pred_cat))

    per_category = {}
    for idx, cat_name in enumerate(CPCB_CATEGORIES):
        per_category[cat_name] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }

    return {
        "N": int(np.sum(valid_mask)),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "confusion_matrix": cm.tolist(),
        "per_category": per_category,
    }


def compute_subgroup_diagnostics(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str = "Target_AQI",
) -> Dict[str, pd.DataFrame]:
    """
    Computes residual diagnostics across City, Month, Season, and AQI Severity Tiers.
    Residual convention: e_i = y_true - y_pred
    - Positive residual (e > 0): Underprediction
    - Negative residual (e < 0): Overprediction
    """
    df = df.copy()
    valid_mask = df[target_col].notna() & df[pred_col].notna()
    sub_df = df[valid_mask].copy()

    # Residual calculation: e = y - y_hat
    sub_df["residual"] = sub_df[target_col] - sub_df[pred_col]
    sub_df["abs_error"] = np.abs(sub_df["residual"])
    sub_df["sq_error"] = sub_df["residual"] ** 2

    # Map target CPCB bucket
    sub_df["cpcb_bucket"] = map_aqi_to_cpcb_bucket(sub_df[target_col])

    # Define meteorological seasons
    month = pd.to_datetime(sub_df["Target_Date"]).dt.month
    sub_df["season"] = np.select(
        [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8, 9]),
            month.isin([10, 11]),
        ],
        ["Winter", "Pre-Monsoon (Summer)", "Monsoon", "Post-Monsoon (Autumn)"],
        default="Unknown",
    )

    def _agg_group(group_col: str) -> pd.DataFrame:
        agg = (
            sub_df.groupby(group_col)
            .agg(
                N=("residual", "count"),
                MAE=("abs_error", "mean"),
                RMSE=("sq_error", lambda s: np.sqrt(np.mean(s))),
                MBE=("residual", lambda s: -np.mean(s)),  # MBE = mean(y_hat - y) = -mean(residual)
                Underprediction_Rate=("residual", lambda s: np.mean(s > 0) * 100),
            )
            .reset_index()
        )
        agg["MAE"] = agg["MAE"].round(4)
        agg["RMSE"] = agg["RMSE"].round(4)
        agg["MBE"] = agg["MBE"].round(4)
        agg["Underprediction_Rate"] = agg["Underprediction_Rate"].round(2)
        return agg

    city_diag = _agg_group("City")
    season_diag = _agg_group("season")
    bucket_diag = _agg_group("cpcb_bucket")

    return {
        "by_city": city_diag,
        "by_season": season_diag,
        "by_bucket": bucket_diag,
    }


def evaluate_split_predictions(
    split_name: str,
    df_split: pd.DataFrame,
    models: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates predictions and evaluates regression and classification diagnostics
    on a designated held-out split (e.g. 'test' or 'stress').
    """
    num_cols, cat_cols, _ = get_feature_columns()
    f_cols = cat_cols + num_cols

    pred_records = df_split[
        ["City", "Date", "Target_Date", "Target_AQI", "AQI_t", "is_shared_cohort", "is_target_valid"]
    ].copy()

    # 1. Persistence Baseline
    persistence = NaivePersistenceModel(aqi_col="AQI_t")
    pred_records["Pred_Persistence"] = persistence.predict(df_split)

    # 2. Supervised Models
    for model_name, pipe in models.items():
        slug = model_name.replace(" ", "_")
        pred_records[f"Pred_{slug}"] = pipe.predict(df_split[f_cols])

    # Shared cohort mask
    shared_mask = pred_records["is_shared_cohort"].to_numpy(dtype=bool)

    # Calculate regression metrics
    metrics_list = []
    classif_list = []

    # Persistence metrics (Cohort A only)
    persist_m = compute_regression_metrics(
        pred_records.loc[shared_mask, "Target_AQI"],
        pred_records.loc[shared_mask, "Pred_Persistence"],
    )
    metrics_list.append({
        "Split": split_name,
        "Model": "Naive Persistence",
        "Cohort": "Cohort A (Shared)",
        **persist_m,
        "MAE_Improvement_%": 0.0,
        "RMSE_Improvement_%": 0.0,
    })

    # Persistence classification
    persist_cls = compute_classification_diagnostics(
        pred_records.loc[shared_mask, "Target_AQI"],
        pred_records.loc[shared_mask, "Pred_Persistence"],
    )
    classif_list.append({
        "Split": split_name,
        "Model": "Naive Persistence",
        "Cohort": "Cohort A (Shared)",
        "N": persist_cls["N"],
        "Macro_F1": persist_cls["macro_f1"],
        "Balanced_Accuracy": persist_cls["balanced_accuracy"],
    })

    base_mae = persist_m["MAE"]
    base_rmse = persist_m["RMSE"]

    for model_name in models.keys():
        slug = model_name.replace(" ", "_")
        col_pred = f"Pred_{slug}"

        # Cohort A (Shared)
        m_shared = compute_regression_metrics(
            pred_records.loc[shared_mask, "Target_AQI"],
            pred_records.loc[shared_mask, col_pred],
        )
        mae_imp = compute_relative_improvement(base_mae, m_shared["MAE"], lower_is_better=True)
        rmse_imp = compute_relative_improvement(base_rmse, m_shared["RMSE"], lower_is_better=True)

        metrics_list.append({
            "Split": split_name,
            "Model": model_name,
            "Cohort": "Cohort A (Shared)",
            **m_shared,
            "MAE_Improvement_%": mae_imp,
            "RMSE_Improvement_%": rmse_imp,
        })

        # Cohort A Classification
        cls_shared = compute_classification_diagnostics(
            pred_records.loc[shared_mask, "Target_AQI"],
            pred_records.loc[shared_mask, col_pred],
        )
        classif_list.append({
            "Split": split_name,
            "Model": model_name,
            "Cohort": "Cohort A (Shared)",
            "N": cls_shared["N"],
            "Macro_F1": cls_shared["macro_f1"],
            "Balanced_Accuracy": cls_shared["balanced_accuracy"],
        })

        # Cohort B (Model-Only, All Targets)
        m_all = compute_regression_metrics(
            pred_records["Target_AQI"],
            pred_records[col_pred],
        )
        metrics_list.append({
            "Split": split_name,
            "Model": f"{model_name} (All Targets)",
            "Cohort": "Cohort B (Model-Only)",
            **m_all,
            "MAE_Improvement_%": np.nan,
            "RMSE_Improvement_%": np.nan,
        })

        # Cohort B Classification
        cls_all = compute_classification_diagnostics(
            pred_records["Target_AQI"],
            pred_records[col_pred],
        )
        classif_list.append({
            "Split": split_name,
            "Model": f"{model_name} (All Targets)",
            "Cohort": "Cohort B (Model-Only)",
            "N": cls_all["N"],
            "Macro_F1": cls_all["macro_f1"],
            "Balanced_Accuracy": cls_all["balanced_accuracy"],
        })

    metrics_df = pd.DataFrame(metrics_list)
    classif_df = pd.DataFrame(classif_list)

    # Ahmedabad Sensitivity Analysis (Secondary Diagnostic on Cohort A)
    ahmedabad_mask = pred_records["City"] == "Ahmedabad"
    non_ahmedabad_shared = shared_mask & (~ahmedabad_mask)

    sensitivity_list = []
    persist_no_ahm = compute_regression_metrics(
        pred_records.loc[non_ahmedabad_shared, "Target_AQI"],
        pred_records.loc[non_ahmedabad_shared, "Pred_Persistence"],
    )
    sensitivity_list.append({
        "Split": split_name,
        "Subset": "Without Ahmedabad (17 Cities)",
        "Model": "Naive Persistence",
        **persist_no_ahm,
        "MAE_Improvement_%": 0.0,
        "RMSE_Improvement_%": 0.0,
    })

    base_mae_no_ahm = persist_no_ahm["MAE"]
    base_rmse_no_ahm = persist_no_ahm["RMSE"]

    for model_name in models.keys():
        slug = model_name.replace(" ", "_")
        col_pred = f"Pred_{slug}"
        m_no_ahm = compute_regression_metrics(
            pred_records.loc[non_ahmedabad_shared, "Target_AQI"],
            pred_records.loc[non_ahmedabad_shared, col_pred],
        )
        sensitivity_list.append({
            "Split": split_name,
            "Subset": "Without Ahmedabad (17 Cities)",
            "Model": model_name,
            **m_no_ahm,
            "MAE_Improvement_%": compute_relative_improvement(base_mae_no_ahm, m_no_ahm["MAE"], lower_is_better=True),
            "RMSE_Improvement_%": compute_relative_improvement(base_rmse_no_ahm, m_no_ahm["RMSE"], lower_is_better=True),
        })

    sensitivity_df = pd.DataFrame(sensitivity_list)

    return pred_records, metrics_df, classif_df, sensitivity_df


def run_phase8_evaluation(save_outputs: bool = True) -> Dict[str, Any]:
    """
    Executes the full Phase 8 evaluation protocol across:
    1. Primary Held-Out Test Set (2019-01-01 to 2019-12-31)
    2. Exogenous Stress-Test Set (2020-01-01 to 2020-07-01)
    """
    feat_path = PROCESSED_DIR / "city_day_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {feat_path}")

    logger.info(f"Loading feature dataset from {feat_path}")
    df = pd.read_parquet(feat_path)

    # Load frozen Phase 7 models
    models = {
        "Ridge Regression": joblib.load(MODELS_DIR / "ridge_regression_val_model.joblib"),
        "Random Forest": joblib.load(MODELS_DIR / "random_forest_val_model.joblib"),
        "HistGradientBoosting": joblib.load(MODELS_DIR / "histgradientboosting_val_model.joblib"),
    }

    # Slice 2019 test set (target-valid only)
    test_df = df[(df["split"] == "test") & df["is_target_valid"]].copy()
    stress_df = df[(df["split"] == "stress") & df["is_target_valid"]].copy()

    logger.info(f"Evaluating Primary 2019 Test Set: {len(test_df)} target-valid rows")
    test_preds, test_metrics, test_classif, test_sensitivity = evaluate_split_predictions(
        "2019 Primary Test", test_df, models
    )

    logger.info(f"Evaluating 2020 Stress Set: {len(stress_df)} target-valid rows")
    stress_preds, stress_metrics, stress_classif, stress_sensitivity = evaluate_split_predictions(
        "2020 Stress Test", stress_df, models
    )

    # Residual diagnostics on Champion Model (Random Forest) for 2019 Primary Test
    rf_col = "Pred_Random_Forest"
    test_residuals_shared = compute_subgroup_diagnostics(
        test_preds[test_preds["is_shared_cohort"]], rf_col
    )

    if save_outputs:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Save predictions
        test_preds.to_csv(REPORTS_DIR / "phase8_test_predictions.csv", index=False)
        stress_preds.to_csv(REPORTS_DIR / "phase8_stress_predictions.csv", index=False)

        # Save metrics
        test_metrics.to_csv(REPORTS_DIR / "phase8_test_metrics.csv", index=False)
        stress_metrics.to_csv(REPORTS_DIR / "phase8_stress_metrics.csv", index=False)
        all_classif = pd.concat([test_classif, stress_classif], ignore_index=True)
        all_classif.to_csv(REPORTS_DIR / "phase8_classification_metrics.csv", index=False)

        # Save residual diagnostics
        test_residuals_shared["by_city"].to_csv(REPORTS_DIR / "phase8_residual_by_city.csv", index=False)
        test_residuals_shared["by_season"].to_csv(REPORTS_DIR / "phase8_residual_by_season.csv", index=False)
        test_residuals_shared["by_bucket"].to_csv(REPORTS_DIR / "phase8_residual_by_bucket.csv", index=False)

        # Save sensitivity
        all_sensitivity = pd.concat([test_sensitivity, stress_sensitivity], ignore_index=True)
        all_sensitivity.to_csv(REPORTS_DIR / "phase8_ahmedabad_sensitivity.csv", index=False)

        logger.info(f"Successfully generated all Phase 8 evaluation outputs in {REPORTS_DIR}")

    return {
        "test_metrics": test_metrics,
        "stress_metrics": stress_metrics,
        "classification_metrics": pd.concat([test_classif, stress_classif], ignore_index=True),
        "sensitivity_metrics": pd.concat([test_sensitivity, stress_sensitivity], ignore_index=True),
        "residuals_by_city": test_residuals_shared["by_city"],
        "residuals_by_season": test_residuals_shared["by_season"],
        "residuals_by_bucket": test_residuals_shared["by_bucket"],
        "test_predictions": test_preds,
        "stress_predictions": stress_preds,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Executing Phase 8 Final Model Evaluation...")
    out = run_phase8_evaluation(save_outputs=True)
    print("\n=== 2019 PRIMARY TEST EVALUATION METRICS ===")
    print(out["test_metrics"].to_string(index=False))
    print("\n=== 2020 STRESS TEST EVALUATION METRICS ===")
    print(out["stress_metrics"].to_string(index=False))
    print("\n=== CPCB CLASSIFICATION DIAGNOSTICS ===")
    print(out["classification_metrics"].to_string(index=False))
    print("\n=== AHMEDABAD SENSITIVITY ANALYSIS ===")
    print(out["sensitivity_metrics"].to_string(index=False))
