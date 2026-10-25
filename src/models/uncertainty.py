"""
Phase 9.2 — Prediction Uncertainty Quantification Engine.

Provides mathematically verified, distribution-free prediction intervals for the
frozen Random Forest next-day AQI forecasting model.

Methodology & Calibration Standards:
1. Model Immutability: Operates as an external inference uncertainty wrapper around the frozen
   pipeline (models/random_forest_val_model.joblib). Zero retraining, parameter tuning, or refitting.
2. Chronological Calibration Barrier: Calibration is performed strictly on pre-2019 observations
   from the Designated Validation partition (2018-07-01 to 2018-12-31, N=2,982 shared cohort).
   The 2019 primary test and 2020 stress partitions are quarantined and used exclusively for
   held-out evaluation.
3. Conformal Nonconformity Formulation:
   - Residual convention: e_i = y_i - y_hat_i (positive = underprediction, negative = overprediction).
   - Absolute nonconformity score: s_i = |y_i - y_hat_i|.
   - Finite-sample order-statistic rank: k = ceil((n + 1) * (1 - alpha)).
   - Interval construction: L_i = max(0.0, y_hat_i - q_{1-alpha}), U_i = y_hat_i + q_{1-alpha}.
4. Operational Domain Constraints: Lower bounds are truncated at zero (non-negative continuous AQI).
   Upper bounds are strictly NOT clipped at 500, preserving continuous numerical evaluation.
5. Epistemic Limitation: Conformal exchangeability does not hold unconditionally for chronological
   environmental time series with serial autocorrelation and seasonal drift. Evaluated intervals
   represent empirical calibration baselines, not asymptotic guarantees.
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
from sklearn.pipeline import Pipeline

from src.features.feature_builder import PROCESSED_DIR, get_feature_columns
from src.models.evaluation import CPCB_CATEGORIES, map_aqi_to_cpcb_bucket
from src.models.interpretability import load_frozen_pipeline

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def compute_conformal_quantile(
    calibration_scores: Union[np.ndarray, pd.Series],
    alpha: float = 0.10,
) -> Tuple[float, int, int]:
    """
    Computes the finite-sample conformal quantile from nonconformity scores:
        k = ceil((n + 1) * (1 - alpha))
        q = s_{(k)} (k-th smallest score)

    Args:
        calibration_scores: 1D array of calibration nonconformity scores (e.g. absolute residuals).
        alpha: Miscoverage rate (e.g. 0.10 for 90% nominal interval, 0.05 for 95%).

    Returns:
        Tuple of (conformal_quantile_value, order_statistic_rank_k, zero_indexed_array_index).
    """
    scores = np.sort(np.asarray(calibration_scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration score array cannot be empty.")

    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    idx = min(max(0, k - 1), n - 1)
    q_val = float(scores[idx])
    return q_val, k, idx


def compute_calibration_quantiles(
    df: pd.DataFrame,
    pipeline: Pipeline,
    alpha_levels: Tuple[float, ...] = (0.10, 0.05),
) -> Dict[str, Any]:
    """
    Computes symmetric conformal quantiles and asymmetric empirical residual quantiles
    strictly from the pre-2019 validation partition.

    Args:
        df: Feature DataFrame (must contain 'split', 'is_target_valid', 'is_shared_cohort').
        pipeline: Frozen forecasting pipeline.
        alpha_levels: Tuple of significance levels (default: 0.10 for 90%, 0.05 for 95%).

    Returns:
        Dictionary containing calibration metrics, sample size, date range, and quantiles.
    """
    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols

    val_mask = (df["split"] == "val") & df["is_target_valid"] & df["is_shared_cohort"]
    val_df = df[val_mask].sort_values(["Target_Date", "City"]).reset_index(drop=True)

    if len(val_df) == 0:
        raise ValueError("No eligible validation observations found for calibration.")

    # Quarantine verification
    assert (val_df["split"] == "val").all(), "Calibration data must strictly be from validation split."
    assert (val_df["Target_Date"] <= pd.Timestamp("2018-12-31")).all(), (
        "Calibration data must not extend beyond 2018-12-31."
    )

    preds = pipeline.predict(val_df[fcols])
    y_true = val_df["Target_AQI"].to_numpy(dtype=float)

    # Residual convention: e_i = y_i - y_hat_i
    residuals = y_true - preds
    abs_residuals = np.abs(residuals)
    n_cal = len(abs_residuals)

    quantiles_summary = {}
    for alpha in alpha_levels:
        level_pct = int(round((1.0 - alpha) * 100))
        # Symmetric conformal quantile
        q_sym, k, idx = compute_conformal_quantile(abs_residuals, alpha=alpha)

        # Asymmetric empirical residual quantiles
        delta_low = float(np.quantile(residuals, alpha / 2.0))
        delta_high = float(np.quantile(residuals, 1.0 - alpha / 2.0))

        quantiles_summary[f"symmetric_{level_pct}"] = {
            "alpha": float(alpha),
            "nominal_coverage": 1.0 - alpha,
            "rank_k": k,
            "array_idx": idx,
            "quantile_value": q_sym,
        }
        quantiles_summary[f"asymmetric_{level_pct}"] = {
            "alpha": float(alpha),
            "nominal_coverage": 1.0 - alpha,
            "delta_lower": delta_low,
            "delta_upper": delta_high,
        }

    return {
        "n_calibration": n_cal,
        "calibration_date_min": val_df["Target_Date"].min().strftime("%Y-%m-%d"),
        "calibration_date_max": val_df["Target_Date"].max().strftime("%Y-%m-%d"),
        "calibration_mae": float(np.mean(abs_residuals)),
        "calibration_mbe": float(np.mean(residuals)),
        "quantiles": quantiles_summary,
    }


def construct_symmetric_intervals(
    point_predictions: Union[np.ndarray, pd.Series],
    q_value: float,
    truncate_lower: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs symmetric conformal prediction intervals:
        L_i = max(0.0, y_hat_i - q) (if truncate_lower)
        U_i = y_hat_i + q (continuous, unclipped at 500)

    Args:
        point_predictions: Continuous model forecasts (AQI points).
        q_value: Conformal absolute-residual quantile.
        truncate_lower: Whether to enforce non-negative lower bound at 0.0.

    Returns:
        Tuple of (lower_bounds, upper_bounds).
    """
    y_hat = np.asarray(point_predictions, dtype=float)
    lower = y_hat - float(q_value)
    if truncate_lower:
        lower = np.maximum(0.0, lower)
    upper = y_hat + float(q_value)
    return lower, upper


def construct_asymmetric_intervals(
    point_predictions: Union[np.ndarray, pd.Series],
    delta_lower: float,
    delta_upper: float,
    truncate_lower: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs asymmetric empirical residual prediction intervals:
        L_i = max(0.0, y_hat_i + delta_lower)
        U_i = y_hat_i + delta_upper

    Args:
        point_predictions: Continuous model forecasts (AQI points).
        delta_lower: Lower residual quantile (e.g. 5th percentile, typically negative).
        delta_upper: Upper residual quantile (e.g. 95th percentile, positive).
        truncate_lower: Whether to enforce non-negative lower bound at 0.0.

    Returns:
        Tuple of (lower_bounds, upper_bounds).
    """
    y_hat = np.asarray(point_predictions, dtype=float)
    lower = y_hat + float(delta_lower)
    if truncate_lower:
        lower = np.maximum(0.0, lower)
    upper = y_hat + float(delta_upper)
    return lower, upper


def compute_interval_metrics(
    y_true: Union[np.ndarray, pd.Series],
    lower_bounds: Union[np.ndarray, pd.Series],
    upper_bounds: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """
    Calculates empirical coverage and interval width metrics.

    Args:
        y_true: True continuous observations.
        lower_bounds: Lower prediction interval bounds.
        upper_bounds: Upper prediction interval bounds.

    Returns:
        Dictionary with coverage, mean width, median width, min width, max width.
    """
    y = np.asarray(y_true, dtype=float)
    l = np.asarray(lower_bounds, dtype=float)
    u = np.asarray(upper_bounds, dtype=float)

    is_covered = (y >= l) & (y <= u)
    widths = u - l

    return {
        "n_evaluated": int(len(y)),
        "empirical_coverage": float(np.mean(is_covered)),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "min_width": float(np.min(widths)),
        "max_width": float(np.max(widths)),
    }


def compute_subgroup_coverage(
    df_eval: pd.DataFrame,
    y_true: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    group_col: str,
) -> pd.DataFrame:
    """
    Evaluates empirical coverage and interval widths partitioned by a subgroup column (e.g. City or bucket).

    Args:
        df_eval: Evaluation DataFrame.
        y_true: True continuous observations.
        lower_bounds: Lower prediction interval bounds.
        upper_bounds: Upper prediction interval bounds.
        group_col: Column to group by.

    Returns:
        pd.DataFrame with N, empirical coverage, mean width, and median width per subgroup.
    """
    is_covered = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    widths = upper_bounds - lower_bounds

    temp_df = pd.DataFrame({
        "group": df_eval[group_col].to_numpy(),
        "covered": is_covered,
        "width": widths,
    })

    records = []
    for grp_val, grp_data in temp_df.groupby("group"):
        records.append({
            group_col: grp_val,
            "N": len(grp_data),
            "Coverage": float(grp_data["covered"].mean()),
            "Mean_Width": float(grp_data["width"].mean()),
            "Median_Width": float(grp_data["width"].median()),
        })

    return pd.DataFrame(records)


def generate_prediction_intervals_figure(
    df_test: pd.DataFrame,
    pipeline: Pipeline,
    q_90: float,
    city: str = "Delhi",
    start_date: str = "2019-01-01",
    end_date: str = "2019-03-31",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generates a publication-grade time-series visualization displaying observed AQI,
    frozen Random Forest point predictions, and the 90% prediction interval ribbon.

    Args:
        df_test: Test partition DataFrame.
        pipeline: Frozen pipeline.
        q_90: Calibrated 90% conformal quantile.
        city: Representative city to plot.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        output_path: Path to save figure. Defaults to reports/figures/prediction_intervals.png.

    Returns:
        Path to saved PNG.
    """
    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols

    city_mask = (
        (df_test["City"] == city)
        & (df_test["Target_Date"] >= pd.Timestamp(start_date))
        & (df_test["Target_Date"] <= pd.Timestamp(end_date))
    )
    sub = df_test[city_mask].sort_values("Target_Date").reset_index(drop=True)

    if len(sub) == 0:
        raise ValueError(f"No records found for city {city} between {start_date} and {end_date}.")

    preds = pipeline.predict(sub[fcols])
    y_true = sub["Target_AQI"].to_numpy(dtype=float)
    dates = pd.to_datetime(sub["Target_Date"])

    lower_90, upper_90 = construct_symmetric_intervals(preds, q_90, truncate_lower=True)
    is_covered = (y_true >= lower_90) & (y_true <= upper_90)
    sub_cov = float(np.mean(is_covered))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prediction interval ribbon
    ax.fill_between(
        dates,
        lower_90,
        upper_90,
        color="#1f77b4",
        alpha=0.25,
        label=f"90% Prediction Interval (±{q_90:.1f} AQI pts)",
    )

    # Point predictions
    ax.plot(
        dates,
        preds,
        color="#1f77b4",
        linewidth=1.8,
        linestyle="-",
        label="Random Forest Forecast (y_hat)",
    )

    # Observed AQI points
    ax.scatter(
        dates[is_covered],
        y_true[is_covered],
        color="#2ca02c",
        s=30,
        alpha=0.9,
        edgecolors="black",
        linewidth=0.5,
        label="Observed AQI (Inside Band)",
        zorder=4,
    )

    # Observed AQI points outside band
    if not np.all(is_covered):
        ax.scatter(
            dates[~is_covered],
            y_true[~is_covered],
            color="#d62728",
            s=45,
            alpha=1.0,
            marker="^",
            edgecolors="black",
            linewidth=0.8,
            label="Observed AQI (Outside Band)",
            zorder=5,
        )

    # CPCB Category Reference Lines
    cpcb_bands = [
        (50, "Good", "#00e400"),
        (100, "Satisfactory", "#92d050"),
        (200, "Moderate", "#ffff00"),
        (300, "Poor", "#ff7e00"),
        (400, "Very Poor", "#ff0000"),
    ]
    for thresh, name, col in cpcb_bands:
        ax.axhline(thresh, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

    ax.set_title(
        f"Next-Day AQI 90% Prediction Intervals — {city} (Winter-Spring 2019)\n"
        f"Conformal Calibration on Pre-2019 Validation Split | Subgroup Coverage: {sub_cov * 100:.1f}% (N={len(sub)})",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Target Date (t+1)", fontsize=10)
    ax.set_ylabel("Air Quality Index (Continuous AQI Points)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()

    out_p = output_path or (FIGURES_DIR / "prediction_intervals.png")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    return out_p


def run_uncertainty_pipeline(
    dataset_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Executes the complete Milestone 9.2 Uncertainty Quantification Pipeline.

    Workflow:
    1. Load feature dataset and frozen Random Forest pipeline.
    2. Calibrate symmetric (90%, 95%) and asymmetric intervals on pre-2019 validation set (N=2,982).
    3. Evaluate empirical coverage and widths on 2019 test set (N=6,201) and 2020 stress set (N=3,124).
    4. Compute granular 18-city coverage metrics and severity category diagnostics.
    5. Evaluate extreme sensor anomaly cohort (>500 AQI).
    6. Generate time-series interval plot and persist summary CSVs.

    Returns:
        Structured audit dictionary summarizing uncertainty quantification results.
    """
    feat_path = dataset_path or (PROCESSED_DIR / "city_day_features.parquet")
    df = pd.read_parquet(feat_path)
    pipeline = load_frozen_pipeline(model_path)

    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols

    # 1. Calibration
    logger.info("Calibrating prediction intervals on pre-2019 validation partition...")
    cal_results = compute_calibration_quantiles(df, pipeline, alpha_levels=(0.10, 0.05))
    q_90 = cal_results["quantiles"]["symmetric_90"]["quantile_value"]
    q_95 = cal_results["quantiles"]["symmetric_95"]["quantile_value"]

    delta_low_90 = cal_results["quantiles"]["asymmetric_90"]["delta_lower"]
    delta_high_90 = cal_results["quantiles"]["asymmetric_90"]["delta_upper"]
    delta_low_95 = cal_results["quantiles"]["asymmetric_95"]["delta_lower"]
    delta_high_95 = cal_results["quantiles"]["asymmetric_95"]["delta_upper"]

    # 2. Evaluation on Test (2019) and Stress (2020)
    eval_metrics_list = []
    city_tables = {}
    bucket_tables = {}
    anomaly_tables = {}

    splits_to_eval = [
        ("2019 Primary Test", "test"),
        ("2020 Stress Test", "stress"),
    ]

    for split_label, split_name in splits_to_eval:
        sub = df[(df["split"] == split_name) & df["is_target_valid"] & df["is_shared_cohort"]].sort_values(
            ["Target_Date", "City"]
        ).reset_index(drop=True)

        preds = pipeline.predict(sub[fcols])
        y_true = sub["Target_AQI"].to_numpy(dtype=float)

        # Symmetric 90%
        l_sym_90, u_sym_90 = construct_symmetric_intervals(preds, q_90, truncate_lower=True)
        m_sym_90 = compute_interval_metrics(y_true, l_sym_90, u_sym_90)
        eval_metrics_list.append({
            "Split": split_label,
            "Method": "Symmetric Conformal",
            "Nominal_Coverage": 0.90,
            **m_sym_90,
        })

        # Symmetric 95%
        l_sym_95, u_sym_95 = construct_symmetric_intervals(preds, q_95, truncate_lower=True)
        m_sym_95 = compute_interval_metrics(y_true, l_sym_95, u_sym_95)
        eval_metrics_list.append({
            "Split": split_label,
            "Method": "Symmetric Conformal",
            "Nominal_Coverage": 0.95,
            **m_sym_95,
        })

        # Asymmetric 90%
        l_asym_90, u_asym_90 = construct_asymmetric_intervals(preds, delta_low_90, delta_high_90, truncate_lower=True)
        m_asym_90 = compute_interval_metrics(y_true, l_asym_90, u_asym_90)
        eval_metrics_list.append({
            "Split": split_label,
            "Method": "Asymmetric Empirical",
            "Nominal_Coverage": 0.90,
            **m_asym_90,
        })

        # Asymmetric 95%
        l_asym_95, u_asym_95 = construct_asymmetric_intervals(preds, delta_low_95, delta_high_95, truncate_lower=True)
        m_asym_95 = compute_interval_metrics(y_true, l_asym_95, u_asym_95)
        eval_metrics_list.append({
            "Split": split_label,
            "Method": "Asymmetric Empirical",
            "Nominal_Coverage": 0.95,
            **m_asym_95,
        })

        # Subgroups (evaluated on Symmetric 90% baseline)
        city_df = compute_subgroup_coverage(sub, y_true, l_sym_90, u_sym_90, "City")
        city_df["Split"] = split_label
        city_tables[split_name] = city_df

        sub["bucket"] = map_aqi_to_cpcb_bucket(y_true)
        bucket_df = compute_subgroup_coverage(sub, y_true, l_sym_90, u_sym_90, "bucket")
        bucket_df["Split"] = split_label
        bucket_tables[split_name] = bucket_df

        # Anomaly > 500
        above_500 = sub[sub["Target_AQI"] > 500.0]
        if len(above_500) > 0:
            y_500 = above_500["Target_AQI"].to_numpy(dtype=float)
            p_500 = pipeline.predict(above_500[fcols])
            l_500, u_500 = construct_symmetric_intervals(p_500, q_90, truncate_lower=True)
            m_500 = compute_interval_metrics(y_500, l_500, u_500)
            anomaly_tables[split_name] = {
                "Split": split_label,
                "N": len(above_500),
                "Coverage_90": m_500["empirical_coverage"],
                "Mean_Width": m_500["mean_width"],
            }
        else:
            anomaly_tables[split_name] = {"Split": split_label, "N": 0, "Coverage_90": np.nan, "Mean_Width": np.nan}

    metrics_df = pd.DataFrame(eval_metrics_list)
    all_city_df = pd.concat([city_tables["test"], city_tables["stress"]], ignore_index=True)

    saved_artifacts = {}
    if save_artifacts:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        p_metrics_csv = REPORTS_DIR / "phase9_uncertainty_metrics.csv"
        metrics_df.to_csv(p_metrics_csv, index=False)
        saved_artifacts["metrics_csv"] = str(p_metrics_csv)

        p_city_csv = REPORTS_DIR / "phase9_city_coverage.csv"
        all_city_df.to_csv(p_city_csv, index=False)
        saved_artifacts["city_csv"] = str(p_city_csv)

        # Plot 2019 Delhi winter-spring sequence
        test_sub = df[(df["split"] == "test") & df["is_target_valid"] & df["is_shared_cohort"]].copy()
        p_plot = generate_prediction_intervals_figure(
            test_sub,
            pipeline,
            q_90,
            city="Delhi",
            start_date="2019-01-01",
            end_date="2019-03-31",
            output_path=FIGURES_DIR / "prediction_intervals.png",
        )
        saved_artifacts["figure"] = str(p_plot)

    return {
        "calibration": cal_results,
        "metrics_summary": metrics_df,
        "city_breakdown": all_city_df,
        "anomaly_breakdown": anomaly_tables,
        "saved_artifacts": saved_artifacts,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Executing Phase 9.2 Prediction Uncertainty Quantification...")
    results = run_uncertainty_pipeline(save_artifacts=True)
    print("\nCalibration Results:")
    print(f"  Validation records: N={results['calibration']['n_calibration']} ({results['calibration']['calibration_date_min']} to {results['calibration']['calibration_date_max']})")
    print(f"  Symmetric 90% Conformal Quantile (q): {results['calibration']['quantiles']['symmetric_90']['quantile_value']:.4f} AQI points")
    print(f"  Symmetric 95% Conformal Quantile (q): {results['calibration']['quantiles']['symmetric_95']['quantile_value']:.4f} AQI points")
    print("\nSummary Metrics Table:")
    print(results["metrics_summary"].to_string(index=False))
    print("\nAnomaly (> 500 AQI) Coverage:")
    for split_k, v in results["anomaly_breakdown"].items():
        print(f"  {v['Split']}: N={v['N']}, Coverage_90={v['Coverage_90'] * 100:.2f}%")
    print("\nSaved Artifacts:")
    for k, v in results["saved_artifacts"].items():
        print(f"  {k}: {v}")
