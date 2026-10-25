"""
Feature engineering module for India Air Quality Intelligence.
Predicts next-day AQI (t+1) using strictly information available at or before prediction origin t.
"""

from src.features.feature_builder import (
    build_feature_pipeline,
    build_target,
    build_lag_features,
    build_pollutant_features,
    build_rolling_features,
    build_calendar_features,
    assign_chronological_splits,
    get_feature_columns,
    TIER1_CITIES,
)

__all__ = [
    "build_feature_pipeline",
    "build_target",
    "build_lag_features",
    "build_pollutant_features",
    "build_rolling_features",
    "build_calendar_features",
    "assign_chronological_splits",
    "get_feature_columns",
    "TIER1_CITIES",
]
