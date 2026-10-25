"""
Chronological Expanding-Window Validation and Model Evaluation.

Enforces zero-leakage temporal cross-validation:
1. Training data precedes validation data in all folds (tau_train < tau_val).
2. Rows sharing the same Target_Date remain in the same temporal split.
3. Learned preprocessing (scaling, imputation, encoding) fits strictly on the training fold.
4. Primary test (2019) and stress test (2020) are strictly excluded from tuning folds.
5. Comparisons against Naive Persistence are evaluated on identical row indices (Cohort A).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.features.feature_builder import get_feature_columns
from src.models.baseline import NaivePersistenceModel
from src.models.metrics import compute_regression_metrics, compute_relative_improvement


@dataclass
class ExpandingWindowFold:
    fold_idx: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    description: str


def get_training_folds() -> List[ExpandingWindowFold]:
    """
    Returns the approved chronological expanding-window folds on the training partition
    (Target_Date 2015-01-02 to 2018-06-30).
    """
    return [
        ExpandingWindowFold(
            fold_idx=1,
            train_start="2015-01-02",
            train_end="2016-12-31",
            val_start="2017-01-01",
            val_end="2017-06-30",
            description="Fold 1: Train 2015-16 (2.0y) -> Val early 2017 (6m)",
        ),
        ExpandingWindowFold(
            fold_idx=2,
            train_start="2015-01-02",
            train_end="2017-06-30",
            val_start="2017-07-01",
            val_end="2017-12-31",
            description="Fold 2: Train 2015-mid17 (2.5y) -> Val late 2017 (6m)",
        ),
        ExpandingWindowFold(
            fold_idx=3,
            train_start="2015-01-02",
            train_end="2017-12-31",
            val_start="2018-01-01",
            val_end="2018-06-30",
            description="Fold 3: Train 2015-17 (3.0y) -> Val early 2018 (6m)",
        ),
    ]


def get_designated_validation_fold() -> ExpandingWindowFold:
    """
    Returns the designated confirmation fold:
    Trained on full training split (2015-01-02 to 2018-06-30)
    Validated on designated validation split (2018-07-01 to 2018-12-31).
    """
    return ExpandingWindowFold(
        fold_idx=0,
        train_start="2015-01-02",
        train_end="2018-06-30",
        val_start="2018-07-01",
        val_end="2018-12-31",
        description="Designated Val: Train 2015-mid18 (~3.5y) -> Val late 2018 (6m)",
    )


def evaluate_model_on_fold(
    model: Any,
    df: pd.DataFrame,
    fold: ExpandingWindowFold,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "Target_AQI",
) -> Dict[str, Any]:
    """
    Trains model on fold's training slice and evaluates on validation slice.
    Reports:
    - Shared Cohort (Cohort A): Model vs Persistence on identical rows.
    - Model-Only Cohort (Cohort B): All target-valid rows.
    """
    num_cols, cat_cols, _ = get_feature_columns()
    f_cols = feature_cols or (cat_cols + num_cols)

    target_dt = pd.to_datetime(df["Target_Date"])

    # Training split: strictly between train_start and train_end, target notna
    train_mask = (
        (target_dt >= fold.train_start)
        & (target_dt <= fold.train_end)
        & df["is_target_valid"]
    )
    # Validation split: strictly between val_start and val_end, target notna
    val_mask = (
        (target_dt >= fold.val_start)
        & (target_dt <= fold.val_end)
        & df["is_target_valid"]
    )

    train_df = df[train_mask]
    val_df = df[val_mask]

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            f"Fold {fold.fold_idx} has insufficient data: train={len(train_df)}, val={len(val_df)}"
        )

    # Fit model strictly on training slice
    X_train = train_df[f_cols]
    y_train = train_df[target_col].to_numpy(dtype=float)
    model.fit(X_train, y_train)

    # Predict on validation slice
    X_val = val_df[f_cols]
    y_val = val_df[target_col].to_numpy(dtype=float)
    y_pred_model = model.predict(X_val)

    # Persistence predictions
    persistence = NaivePersistenceModel(aqi_col="AQI_t")
    y_pred_persist = persistence.predict(val_df)

    # 1. Primary Shared-Cohort Comparison (both AQI_t and Target_AQI valid)
    shared_mask = val_df["is_shared_cohort"].to_numpy(dtype=bool)
    shared_indices = val_df.index[shared_mask]

    metrics_persist_shared = compute_regression_metrics(
        y_val[shared_mask], y_pred_persist[shared_mask]
    )
    metrics_model_shared = compute_regression_metrics(
        y_val[shared_mask], y_pred_model[shared_mask]
    )

    # Relative improvement on shared cohort
    mae_improvement = compute_relative_improvement(
        metrics_persist_shared["MAE"], metrics_model_shared["MAE"], lower_is_better=True
    )
    rmse_improvement = compute_relative_improvement(
        metrics_persist_shared["RMSE"], metrics_model_shared["RMSE"], lower_is_better=True
    )

    # 2. Model-Only Cohort (all target-valid rows)
    metrics_model_all = compute_regression_metrics(y_val, y_pred_model)

    return {
        "fold": fold,
        "n_train": len(train_df),
        "n_val_grid": len(df[(target_dt >= fold.val_start) & (target_dt <= fold.val_end)]),
        "n_val_target": len(val_df),
        "n_val_shared": int(np.sum(shared_mask)),
        "shared_indices": shared_indices,
        "persistence_shared": metrics_persist_shared,
        "model_shared": metrics_model_shared,
        "model_all_targets": metrics_model_all,
        "improvement_mae_pct": mae_improvement,
        "improvement_rmse_pct": rmse_improvement,
        "val_predictions": pd.DataFrame({
            "City": val_df["City"],
            "Target_Date": val_df["Target_Date"],
            "Target_AQI": y_val,
            "AQI_t": val_df["AQI_t"],
            "Pred_Persistence": y_pred_persist,
            "Pred_Model": y_pred_model,
            "is_shared_cohort": shared_mask,
        }, index=val_df.index),
    }


def run_chronological_cv(
    model_builder: Callable[[], Any],
    df: pd.DataFrame,
    folds: Optional[List[ExpandingWindowFold]] = None,
) -> List[Dict[str, Any]]:
    """Runs chronological expanding-window cross validation across multiple folds."""
    eval_folds = folds or get_training_folds()
    results = []
    for fold in eval_folds:
        model = model_builder()
        res = evaluate_model_on_fold(model, df, fold)
        results.append(res)
    return results
