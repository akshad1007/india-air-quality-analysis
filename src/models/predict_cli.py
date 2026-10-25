"""
Phase 9.3 — Next-Day AQI Forecasting Inference CLI.

Provides a reproducible, leakage-free command-line interface for the frozen
Random Forest forecasting pipeline and conformal uncertainty quantification engine.

Key Governance Invariants:
1. Model Immutability: Loads models/random_forest_val_model.joblib strictly in read-only mode.
   Never fits, retrains, or updates model weights or preprocessor parameters.
2. Input Contract: Requires pre-engineered feature rows conforming to the 19-feature schema
   (1 categorical City + 18 numerical features).
3. Leakage Prevention: Validates calendar alignment. Target calendar features represent day t+1
   using the project's exact 365.25-day harmonic formulation.
4. Conformal Calibration: Uses verified finite-sample quantiles from the pre-2019 validation
   split (q_90 = 57.0621, q_95 = 90.3847). Lower bounds truncated at 0.0; upper bounds unclipped.
5. Disclosure & Extrapolation Warnings:
   - Any feature imputed by the pipeline's training median is explicitly disclosed.
   - Non-Tier-1 cities receive an explicit extrapolation warning.
   - Extreme-event under-coverage disclaimer is always attached.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.feature_builder import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TIER1_CITIES,
    get_feature_columns,
)
from src.models.evaluation import CPCB_CATEGORIES, map_aqi_to_cpcb_bucket
from src.models.interpretability import load_frozen_pipeline
from src.models.uncertainty import construct_symmetric_intervals

logger = logging.getLogger(__name__)

# Calibrated conformal quantiles from pre-2019 validation split (N=2,982)
# Provenance: docs/reports/PHASE_9_UNCERTAINTY_REPORT.md and src/models/uncertainty.py
CALIBRATED_CONFORMAL_Q_90 = 57.0621
CALIBRATED_CONFORMAL_Q_95 = 90.3847

DISCLAIMER_TEXT = (
    "Prediction intervals represent finite-sample conformal calibration on the "
    "pre-2019 validation set (N=2,982). Intervals are not guaranteed reliable during "
    "extreme episodes (AQI > 500 coverage degrades to 29.33%) or severe distribution drift."
)


def compute_target_calendar_features(target_date: Union[str, pd.Timestamp]) -> Dict[str, float]:
    """
    Computes deterministic calendar features for target day t+1 without lookahead.
    Uses the project's exact 365.25-day circular harmonic convention.
    """
    dt = pd.to_datetime(target_date)
    if pd.isna(dt):
        raise ValueError("Target date cannot be NaT or invalid.")

    doy = dt.dayofyear
    sin_doy = float(np.sin(2 * np.pi * doy / 365.25))
    cos_doy = float(np.cos(2 * np.pi * doy / 365.25))
    dow = float(dt.dayofweek)
    is_weekend = 1.0 if dow in [5.0, 6.0] else 0.0
    month = float(dt.month)

    return {
        "target_sin_doy": sin_doy,
        "target_cos_doy": cos_doy,
        "target_dow": dow,
        "target_is_weekend": is_weekend,
        "target_month": month,
    }


def validate_feature_row(row_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Validates a single input row against the required 19-feature schema.

    Returns:
        Tuple of (clean_row_dict, imputed_features_list, warnings_list).

    Raises:
        ValueError: If required categorical columns are missing, values are non-numeric,
                    or physical quantities are negative.
    """
    clean_row = {}
    imputed_features = []
    warnings = []

    # 1. City Validation
    if "City" not in row_dict or pd.isna(row_dict["City"]) or not str(row_dict["City"]).strip():
        raise ValueError("Missing required categorical feature: 'City'.")
    city = str(row_dict["City"]).strip()
    clean_row["City"] = city

    if city not in TIER1_CITIES:
        warnings.append(
            f"City '{city}' is not in the 18 Tier-1 qualified training cities. "
            "One-hot preprocessor will encode it as an all-zero vector; "
            "predictions represent national baseline relationships without city-specific intercept adjustments."
        )

    # 2. Date and Calendar Validation
    target_date = row_dict.get("Target_Date")
    origin_date = row_dict.get("Date")

    if origin_date is not None and target_date is not None:
        try:
            dt_origin = pd.to_datetime(origin_date)
            dt_target = pd.to_datetime(target_date)
            if (dt_target - dt_origin) != pd.Timedelta(days=1):
                raise ValueError(
                    f"Temporal alignment error: Target_Date ({dt_target.strftime('%Y-%m-%d')}) "
                    f"must be exactly 1 calendar day ahead of prediction origin Date ({dt_origin.strftime('%Y-%m-%d')})."
                )
        except Exception as e:
            if "Temporal alignment error" in str(e):
                raise
            raise ValueError(f"Malformed date encountered: {e}")

    # Check if calendar features need derivation
    calendar_cols = ["target_sin_doy", "target_cos_doy", "target_dow", "target_is_weekend", "target_month"]
    missing_cal = [col for col in calendar_cols if col not in row_dict or pd.isna(row_dict[col])]

    if missing_cal:
        if target_date is not None and not pd.isna(target_date):
            derived_cal = compute_target_calendar_features(target_date)
            for c_col, c_val in derived_cal.items():
                clean_row[c_col] = c_val
        elif origin_date is not None and not pd.isna(origin_date):
            derived_target = pd.to_datetime(origin_date) + pd.Timedelta(days=1)
            derived_cal = compute_target_calendar_features(derived_target)
            for c_col, c_val in derived_cal.items():
                clean_row[c_col] = c_val
        else:
            raise ValueError(
                f"Missing calendar features {missing_cal}. "
                "Must provide either the pre-computed calendar features or a valid 'Target_Date'/'Date'."
            )
    else:
        for c_col in calendar_cols:
            clean_row[c_col] = float(row_dict[c_col])

    # 3. Numerical Predictors Validation
    num_cols, _, _ = get_feature_columns()
    non_calendar_num = [c for c in num_cols if c not in calendar_cols]

    for col in non_calendar_num:
        val = row_dict.get(col)
        if val is None or pd.isna(val) or val == "":
            imputed_features.append(col)
            clean_row[col] = np.nan
        else:
            try:
                num_val = float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{col}' must be numeric, got: {val}")

            # Physical validity check: concentrations and AQI cannot be negative
            if num_val < 0.0:
                raise ValueError(f"Physical constraint violation: '{col}' cannot be negative (got {num_val}).")

            clean_row[col] = num_val

    return clean_row, imputed_features, warnings


def run_inference_single(
    pipeline: Pipeline,
    row_dict: Dict[str, Any],
    q_90: float = CALIBRATED_CONFORMAL_Q_90,
    q_95: float = CALIBRATED_CONFORMAL_Q_95,
) -> Dict[str, Any]:
    """
    Executes inference for a single observation.
    """
    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols

    clean_row, imputed, warnings = validate_feature_row(row_dict)
    df_row = pd.DataFrame([clean_row])[fcols]

    # Predict via frozen pipeline
    point_pred = float(pipeline.predict(df_row)[0])
    if math.isnan(point_pred) or math.isinf(point_pred):
        raise RuntimeError(f"Model generated non-finite prediction: {point_pred}")

    # Map CPCB category
    cpcb_cat = str(map_aqi_to_cpcb_bucket(np.array([point_pred]))[0])

    # Compute prediction intervals
    l_90, u_90 = construct_symmetric_intervals(np.array([point_pred]), q_90, truncate_lower=True)
    l_95, u_95 = construct_symmetric_intervals(np.array([point_pred]), q_95, truncate_lower=True)

    target_date_str = str(row_dict.get("Target_Date", row_dict.get("Date", "Unknown")))
    if target_date_str != "Unknown":
        try:
            target_date_str = pd.to_datetime(target_date_str).strftime("%Y-%m-%d")
        except Exception:
            pass

    return {
        "city": clean_row["City"],
        "target_date": target_date_str,
        "point_forecast_aqi": round(point_pred, 2),
        "cpcb_category": cpcb_cat,
        "prediction_intervals": {
            "symmetric_90": {
                "lower": round(float(l_90[0]), 2),
                "upper": round(float(u_90[0]), 2),
                "margin": round(float(q_90), 4),
            },
            "symmetric_95": {
                "lower": round(float(l_95[0]), 2),
                "upper": round(float(u_95[0]), 2),
                "margin": round(float(q_95), 4),
            },
        },
        "imputed_features": imputed,
        "is_tier1_qualified": clean_row["City"] in TIER1_CITIES,
        "warnings": warnings,
        "disclaimer": DISCLAIMER_TEXT,
    }


def run_inference_batch(
    pipeline: Pipeline,
    df_input: pd.DataFrame,
    q_90: float = CALIBRATED_CONFORMAL_Q_90,
    q_95: float = CALIBRATED_CONFORMAL_Q_95,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Validates and executes inference on an entire batch DataFrame.
    Validates all rows prior to prediction to guarantee atomic execution.
    """
    num_cols, cat_cols, _ = get_feature_columns()
    fcols = cat_cols + num_cols

    clean_records = []
    all_imputed = []
    all_warnings = []

    # Atomic validation pass
    for idx, raw_row in df_input.iterrows():
        try:
            clean_r, imp, wrn = validate_feature_row(raw_row.to_dict())
            clean_records.append(clean_r)
            all_imputed.append(imp)
            all_warnings.append(wrn)
        except Exception as e:
            raise ValueError(f"Batch validation failed on row {idx}: {e}")

    df_clean = pd.DataFrame(clean_records)[fcols]
    preds = pipeline.predict(df_clean)

    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
        raise RuntimeError("Model generated non-finite predictions in batch.")

    cpcb_cats = map_aqi_to_cpcb_bucket(preds)
    l_90, u_90 = construct_symmetric_intervals(preds, q_90, truncate_lower=True)
    l_95, u_95 = construct_symmetric_intervals(preds, q_95, truncate_lower=True)

    df_out = df_input.copy()
    df_out["pred_aqi"] = np.round(preds, 2)
    df_out["pred_cpcb_category"] = cpcb_cats
    df_out["pred_ci90_lower"] = np.round(l_90, 2)
    df_out["pred_ci90_upper"] = np.round(u_90, 2)
    df_out["pred_ci95_lower"] = np.round(l_95, 2)
    df_out["pred_ci95_upper"] = np.round(u_95, 2)
    df_out["imputed_features"] = [",".join(imp) if imp else "None" for imp in all_imputed]
    df_out["tier1_qualified"] = [clean_r["City"] in TIER1_CITIES for clean_r in clean_records]

    # Convert to structured list of dicts for JSON reporting
    structured_records = []
    for i in range(len(preds)):
        structured_records.append({
            "row_index": i,
            "city": clean_records[i]["City"],
            "point_forecast_aqi": round(float(preds[i]), 2),
            "cpcb_category": str(cpcb_cats[i]),
            "prediction_intervals": {
                "symmetric_90": {"lower": round(float(l_90[i]), 2), "upper": round(float(u_90[i]), 2)},
                "symmetric_95": {"lower": round(float(l_95[i]), 2), "upper": round(float(u_95[i]), 2)},
            },
            "imputed_features": all_imputed[i],
            "warnings": all_warnings[i],
        })

    return df_out, structured_records


def format_text_output(res: Dict[str, Any]) -> str:
    """Formats a single prediction result as a readable CLI text display."""
    lines = [
        "=" * 80,
        "INDIA AIR QUALITY INTELLIGENCE — NEXT-DAY AQI FORECASTING CLI",
        "Model: Frozen Random Forest Pipeline (Phase 7 Champion)",
        "=" * 80,
        f"City:                 {res['city']}" + (" (Tier-1 Qualified)" if res['is_tier1_qualified'] else " [UNQUALIFIED - EXTRAPOLATION]"),
        f"Target Date:          {res['target_date']}",
        f"Point Forecast (AQI): {res['point_forecast_aqi']:.2f}",
        f"CPCB Category:        {res['cpcb_category']}",
        "-" * 80,
        "Uncertainty Quantification (Conformal Prediction Intervals):",
        f"  90% Prediction Interval: [{res['prediction_intervals']['symmetric_90']['lower']:.2f}, {res['prediction_intervals']['symmetric_90']['upper']:.2f}] (Margin: ±{res['prediction_intervals']['symmetric_90']['margin']:.2f})",
        f"  95% Prediction Interval: [{res['prediction_intervals']['symmetric_95']['lower']:.2f}, {res['prediction_intervals']['symmetric_95']['upper']:.2f}] (Margin: ±{res['prediction_intervals']['symmetric_95']['margin']:.2f})",
    ]

    if res.get("imputed_features"):
        lines.append("-" * 80)
        lines.append("Imputed Missing Features (Training Set Median):")
        for imp in res["imputed_features"]:
            lines.append(f"  - {imp}")

    if res.get("warnings"):
        lines.append("-" * 80)
        lines.append("WARNINGS:")
        for wrn in res["warnings"]:
            lines.append(f"  ! {wrn}")

    lines.extend([
        "-" * 80,
        f"DISCLAIMER: {res['disclaimer']}",
        "=" * 80,
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="predict_cli",
        description=(
            "Next-Day AQI Forecasting CLI for India Air Quality Intelligence.\n"
            "Evaluates pre-engineered feature rows using the frozen Random Forest pipeline "
            "and calculates 90% and 95% conformal prediction intervals."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-json",
        type=str,
        help="Path to a JSON file containing one pre-engineered feature record.",
    )
    group.add_argument(
        "--input-csv",
        type=str,
        help="Path to a batch CSV file containing pre-engineered feature rows.",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save batch predictions (CSV or JSON based on file extension).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format to stdout (default: text).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to model artifact (defaults to models/random_forest_val_model.joblib).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    model_p = Path(args.model_path) if args.model_path else None

    # Load frozen model
    try:
        pipeline = load_frozen_pipeline(model_p)
    except Exception as e:
        sys.stderr.write(f"Error loading model artifact: {e}\n")
        return 1

    # Single-Row JSON Evaluation
    if args.input_json:
        json_path = Path(args.input_json)
        if not json_path.exists():
            sys.stderr.write(f"Error: Input JSON file not found: {json_path}\n")
            return 2

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                row_data = json.load(f)
            if not isinstance(row_data, dict):
                sys.stderr.write("Error: Input JSON must contain a single JSON object/dictionary.\n")
                return 2

            res = run_inference_single(pipeline, row_data)
        except Exception as e:
            sys.stderr.write(f"Inference error: {e}\n")
            return 2

        if args.format == "json":
            print(json.dumps(res, indent=2))
        else:
            print(format_text_output(res))

        if args.output_file:
            out_p = Path(args.output_file)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

        return 0

    # Batch CSV Evaluation
    if args.input_csv:
        csv_path = Path(args.input_csv)
        if not csv_path.exists():
            sys.stderr.write(f"Error: Input CSV file not found: {csv_path}\n")
            return 2

        try:
            df_in = pd.read_csv(csv_path)
            if len(df_in) == 0:
                sys.stderr.write("Error: Input CSV contains zero rows.\n")
                return 2

            df_out, structured_list = run_inference_batch(pipeline, df_in)
        except Exception as e:
            sys.stderr.write(f"Batch inference error: {e}\n")
            return 2

        if args.format == "json":
            batch_summary = {
                "total_rows": len(df_out),
                "predictions": structured_list,
                "disclaimer": DISCLAIMER_TEXT,
            }
            print(json.dumps(batch_summary, indent=2))
        else:
            print(f"Batch prediction successful for {len(df_out)} rows.")
            print(f"Summary: Mean Predicted AQI = {df_out['pred_aqi'].mean():.2f}")
            print(f"CPCB Category Breakdown:\n{df_out['pred_cpcb_category'].value_counts().to_string()}")
            print("\n" + "=" * 80)
            print(f"DISCLAIMER: {DISCLAIMER_TEXT}")
            print("=" * 80)

        if args.output_file:
            out_p = Path(args.output_file)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.suffix.lower() == ".json":
                with open(out_p, "w", encoding="utf-8") as f:
                    json.dump({"total_rows": len(df_out), "predictions": structured_list}, f, indent=2)
            else:
                df_out.to_csv(out_p, index=False)
            print(f"Output saved to: {out_p}")

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
