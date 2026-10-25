"""
Phase 7 Predictive Modeling Package for India Air Quality Intelligence.
Implements Naive Persistence baseline, Ridge Regression, Random Forest,
and HistGradientBoostingRegressor using chronological validation.
"""

from src.models.metrics import compute_regression_metrics, compute_relative_improvement
from src.models.baseline import NaivePersistenceModel
from src.models.pipelines import (
    build_ridge_pipeline,
    build_rf_pipeline,
    build_histgbm_pipeline,
)
from src.models.validation import (
    ExpandingWindowFold,
    get_training_folds,
    evaluate_model_on_fold,
    run_chronological_cv,
)

__all__ = [
    "compute_regression_metrics",
    "compute_relative_improvement",
    "NaivePersistenceModel",
    "build_ridge_pipeline",
    "build_rf_pipeline",
    "build_histgbm_pipeline",
    "ExpandingWindowFold",
    "get_training_folds",
    "evaluate_model_on_fold",
    "run_chronological_cv",
]
