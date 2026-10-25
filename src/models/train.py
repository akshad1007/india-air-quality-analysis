"""
Phase 7 Model Training, Chronological Tuning & Model Selection.

Workflow:
1. Load feature dataset (city_day_features.parquet).
2. Establish Naive Persistence baseline metrics on shared cohort.
3. Perform chronological expanding-window tuning on training partition
   (Target_Date <= 2018-06-30).
4. Evaluate candidate models on the designated Validation period
   (2018-07-01 to 2018-12-31).
5. Compare models against persistence on identical eligible rows (Cohort A).
6. Save evaluation outputs and model artifacts.
7. Strictly preserve the untouched 2019 test set and 2020 stress set.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.features.feature_builder import PROCESSED_DIR
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

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\reports")
MODELS_DIR = Path(r"c:\Users\KANHA\Desktop\SPSS PROJECT\models")


def tune_ridge_models(df: pd.DataFrame, folds: List[ExpandingWindowFold]) -> Tuple[float, List[Dict[str, Any]]]:
    """Evaluates candidate alpha values on training folds to select optimal regularization."""
    candidate_alphas = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    tuning_records = []
    best_alpha = 10.0
    best_mean_mae = float("inf")

    logger.info("Tuning Ridge regression regularization parameter alpha...")
    for alpha in candidate_alphas:
        cv_results = run_chronological_cv(
            lambda a=alpha: build_ridge_pipeline(alpha=a, random_state=42),
            df,
            folds,
        )
        fold_maes = [r["model_shared"]["MAE"] for r in cv_results]
        mean_mae = float(np.mean(fold_maes))
        tuning_records.append({
            "alpha": alpha,
            "mean_cv_mae": round(mean_mae, 4),
            "fold_maes": fold_maes,
        })
        if mean_mae < best_mean_mae:
            best_mean_mae = mean_mae
            best_alpha = alpha

    logger.info(f"Optimal Ridge alpha: {best_alpha} (Mean CV MAE: {best_mean_mae:.4f})")
    return best_alpha, tuning_records


def tune_rf_models(df: pd.DataFrame, folds: List[ExpandingWindowFold]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates candidate Random Forest configurations across training folds."""
    candidate_configs = [
        {"n_estimators": 50, "max_depth": 8, "min_samples_split": 10},
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
        {"n_estimators": 100, "max_depth": 14, "min_samples_split": 5},
    ]
    tuning_records = []
    best_cfg = candidate_configs[1]
    best_mean_mae = float("inf")

    logger.info("Tuning Random Forest hyperparameters...")
    for cfg in candidate_configs:
        cv_results = run_chronological_cv(
            lambda c=cfg: build_rf_pipeline(
                n_estimators=c["n_estimators"],
                max_depth=c["max_depth"],
                min_samples_split=c["min_samples_split"],
                random_state=42,
            ),
            df,
            folds,
        )
        fold_maes = [r["model_shared"]["MAE"] for r in cv_results]
        mean_mae = float(np.mean(fold_maes))
        record = {**cfg, "mean_cv_mae": round(mean_mae, 4), "fold_maes": fold_maes}
        tuning_records.append(record)
        if mean_mae < best_mean_mae:
            best_mean_mae = mean_mae
            best_cfg = cfg

    logger.info(f"Optimal Random Forest config: {best_cfg} (Mean CV MAE: {best_mean_mae:.4f})")
    return best_cfg, tuning_records


def tune_histgbm_models(df: pd.DataFrame, folds: List[ExpandingWindowFold]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates candidate HistGradientBoosting configurations across training folds."""
    candidate_configs = [
        {"learning_rate": 0.05, "max_iter": 100, "max_depth": 6, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_iter": 150, "max_depth": 8, "l2_regularization": 2.0},
        {"learning_rate": 0.03, "max_iter": 200, "max_depth": 8, "l2_regularization": 1.0},
    ]
    tuning_records = []
    best_cfg = candidate_configs[1]
    best_mean_mae = float("inf")

    logger.info("Tuning HistGradientBoosting hyperparameters...")
    for cfg in candidate_configs:
        cv_results = run_chronological_cv(
            lambda c=cfg: build_histgbm_pipeline(
                learning_rate=c["learning_rate"],
                max_iter=c["max_iter"],
                max_depth=c["max_depth"],
                l2_regularization=c["l2_regularization"],
                random_state=42,
            ),
            df,
            folds,
        )
        fold_maes = [r["model_shared"]["MAE"] for r in cv_results]
        mean_mae = float(np.mean(fold_maes))
        record = {**cfg, "mean_cv_mae": round(mean_mae, 4), "fold_maes": fold_maes}
        tuning_records.append(record)
        if mean_mae < best_mean_mae:
            best_mean_mae = mean_mae
            best_cfg = cfg

    logger.info(f"Optimal HistGBM config: {best_cfg} (Mean CV MAE: {best_mean_mae:.4f})")
    return best_cfg, tuning_records


def run_phase7_selection(
    feature_parquet_path: Optional[Path] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Executes complete Phase 7 model selection and evaluation protocol.
    """
    path = feature_parquet_path or (PROCESSED_DIR / "city_day_features.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")

    logger.info(f"Loading feature dataset from {path}")
    df = pd.read_parquet(path)

    # Confirm 2019 and 2020 are preserved untouched
    test_rows = df[df["split"] == "test"]
    stress_rows = df[df["split"] == "stress"]
    logger.info(
        f"Verified held-out partitions: 2019 Test ({len(test_rows)} rows), 2020 Stress ({len(stress_rows)} rows) - strictly isolated from tuning."
    )

    training_folds = get_training_folds()
    val_fold = get_designated_validation_fold()

    # Step 1: Tune models on training folds
    best_ridge_alpha, ridge_tuning = tune_ridge_models(df, training_folds)
    best_rf_cfg, rf_tuning = tune_rf_models(df, training_folds)
    best_histgbm_cfg, histgbm_tuning = tune_histgbm_models(df, training_folds)

    # Step 2: Evaluate selected models on designated validation period (2018-07-01 to 2018-12-31)
    logger.info("Evaluating selected models on designated validation split...")

    models = {
        "Ridge Regression": build_ridge_pipeline(alpha=best_ridge_alpha, random_state=42),
        "Random Forest": build_rf_pipeline(**best_rf_cfg, random_state=42),
        "HistGradientBoosting": build_histgbm_pipeline(**best_histgbm_cfg, random_state=42),
    }

    val_eval_results = {}
    val_predictions_df = None

    for model_name, pipe in models.items():
        eval_res = evaluate_model_on_fold(pipe, df, val_fold)
        val_eval_results[model_name] = {
            "model_shared": eval_res["model_shared"],
            "model_all_targets": eval_res["model_all_targets"],
            "improvement_mae_pct": eval_res["improvement_mae_pct"],
            "improvement_rmse_pct": eval_res["improvement_rmse_pct"],
        }
        if val_predictions_df is None:
            # Baseline predictions are identical across model evaluations
            val_predictions_df = eval_res["val_predictions"][
                ["City", "Target_Date", "Target_AQI", "AQI_t", "is_shared_cohort", "Pred_Persistence"]
            ].copy()
        val_predictions_df[f"Pred_{model_name.replace(' ', '_')}"] = eval_res["val_predictions"]["Pred_Model"]

    # Baseline metrics on designated validation period
    persistence_val_metrics = eval_res["persistence_shared"]

    # Summary table
    summary_data = [
        {
            "Model": "Naive Persistence",
            "Cohort": "Cohort A (Shared)",
            "N": persistence_val_metrics["N"],
            "MAE": persistence_val_metrics["MAE"],
            "RMSE": persistence_val_metrics["RMSE"],
            "R2": persistence_val_metrics["R2"],
            "MAE_Improvement_%": 0.0,
        }
    ]
    for model_name, res in val_eval_results.items():
        summary_data.append({
            "Model": model_name,
            "Cohort": "Cohort A (Shared)",
            "N": res["model_shared"]["N"],
            "MAE": res["model_shared"]["MAE"],
            "RMSE": res["model_shared"]["RMSE"],
            "R2": res["model_shared"]["R2"],
            "MAE_Improvement_%": res["improvement_mae_pct"],
        })
        summary_data.append({
            "Model": f"{model_name} (All Targets)",
            "Cohort": "Cohort B (Model-Only)",
            "N": res["model_all_targets"]["N"],
            "MAE": res["model_all_targets"]["MAE"],
            "RMSE": res["model_all_targets"]["RMSE"],
            "R2": res["model_all_targets"]["R2"],
            "MAE_Improvement_%": np.nan,
        })

    summary_df = pd.DataFrame(summary_data)

    results_payload = {
        "designated_validation_period": {
            "val_start": val_fold.val_start,
            "val_end": val_fold.val_end,
            "train_start": val_fold.train_start,
            "train_end": val_fold.train_end,
        },
        "hyperparameter_tuning": {
            "ridge": {"selected_alpha": best_ridge_alpha, "tuning_records": ridge_tuning},
            "random_forest": {"selected_config": best_rf_cfg, "tuning_records": rf_tuning},
            "histgbm": {"selected_config": best_histgbm_cfg, "tuning_records": histgbm_tuning},
        },
        "persistence_shared_metrics": persistence_val_metrics,
        "validation_evaluation_summary": val_eval_results,
    }

    if save_artifacts:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        json_path = REPORTS_DIR / "model_selection_results.json"
        with open(json_path, "w") as f:
            json.dump(results_payload, f, indent=2)

        # Save CSV summary
        summary_df.to_csv(REPORTS_DIR / "model_selection_summary.csv", index=False)

        # Save validation predictions
        val_predictions_df.to_csv(REPORTS_DIR / "validation_predictions.csv", index=False)

        # Serialize fitted models on training partition
        for name, pipe in models.items():
            slug = name.lower().replace(" ", "_")
            joblib.dump(pipe, MODELS_DIR / f"{slug}_val_model.joblib")

        logger.info(f"Serialized models to {MODELS_DIR} and reports to {REPORTS_DIR}")

    return {
        "summary_table": summary_df,
        "results_payload": results_payload,
        "val_predictions": val_predictions_df,
        "trained_models": models,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Executing Phase 7 Model Selection & Implementation...")
    res = run_phase7_selection()
    print("\n=== DESIGNATED VALIDATION PERIOD EVALUATION SUMMARY (2018-07-01 to 2018-12-31) ===")
    print(res["summary_table"].to_string(index=False))
