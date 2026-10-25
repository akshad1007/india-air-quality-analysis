"""
Evaluation metrics for continuous AQI forecasting.

Provides:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R2 (Coefficient of Determination), handling edge cases (zero variance, small sample)
- MBE (Mean Bias Error)
- Relative improvement calculations vs. baseline
"""

from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """
    Computes regression evaluation metrics on valid, aligned predictions.

    Parameters:
        y_true: Ground truth observed AQI targets.
        y_pred: Model predicted AQI values.

    Returns:
        Dict with 'N', 'MAE', 'RMSE', 'R2', 'MBE'.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # Filter out NaNs simultaneously
    valid_mask = (~np.isnan(y_true_arr)) & (~np.isnan(y_pred_arr))
    y_t = y_true_arr[valid_mask]
    y_p = y_pred_arr[valid_mask]

    n_samples = len(y_t)
    if n_samples == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "MBE": np.nan,
        }

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mbe = float(np.mean(y_p - y_t))

    # R2 requires at least 2 samples and non-zero ground truth variance
    if n_samples < 2 or np.var(y_t) < 1e-12:
        r2 = np.nan
    else:
        r2 = float(r2_score(y_t, y_p))

    return {
        "N": n_samples,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4) if not np.isnan(r2) else np.nan,
        "MBE": round(mbe, 4),
    }


def compute_relative_improvement(
    baseline_val: float,
    model_val: float,
    lower_is_better: bool = True,
) -> float:
    """
    Computes the percentage improvement of a model relative to a baseline.
    Positive value indicates improvement.

    Parameters:
        baseline_val: Metric value of baseline (e.g. Persistence MAE).
        model_val: Metric value of candidate model.
        lower_is_better: True for loss metrics (MAE, RMSE), False for score metrics (R2).

    Returns:
        Percentage improvement.
    """
    if np.isnan(baseline_val) or np.isnan(model_val) or abs(baseline_val) < 1e-12:
        return np.nan

    if lower_is_better:
        # Error reduction: (base - model) / base * 100
        improvement = (baseline_val - model_val) / baseline_val * 100.0
    else:
        # Score increase: (model - base) / abs(base) * 100
        improvement = (model_val - baseline_val) / abs(baseline_val) * 100.0

    return round(float(improvement), 2)
