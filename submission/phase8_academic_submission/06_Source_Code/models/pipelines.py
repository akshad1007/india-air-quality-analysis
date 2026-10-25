"""
Supervised machine learning pipelines for Next-Day AQI Forecasting.

Models:
1. Ridge Regression (One-Hot City + Median Imputer + StandardScaler + Ridge)
2. Random Forest Regressor (One-Hot City + Median Imputer + RandomForest)
3. HistGradientBoostingRegressor (Ordinal City + native missing handling + HistGBM)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.features.feature_builder import get_feature_columns


def build_ridge_pipeline(
    alpha: float = 10.0,
    random_state: int = 42,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> Pipeline:
    """
    Constructs a leak-free Ridge Regression pipeline:
    - OneHotEncoder on City (handle_unknown='ignore')
    - SimpleImputer(median) + StandardScaler on numerical features
    - Ridge(alpha=alpha)
    """
    default_num, default_cat, _ = get_feature_columns()
    n_cols = num_cols or default_num
    c_cols = cat_cols or default_cat

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), c_cols),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                n_cols,
            ),
        ]
    )

    model = Ridge(alpha=alpha, random_state=random_state)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )
    return pipeline


def build_rf_pipeline(
    n_estimators: int = 100,
    max_depth: Optional[int] = 12,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> Pipeline:
    """
    Constructs a Random Forest Regressor pipeline:
    - OneHotEncoder on City (handle_unknown='ignore')
    - SimpleImputer(median) on numerical features
    - RandomForestRegressor
    """
    default_num, default_cat, _ = get_feature_columns()
    n_cols = num_cols or default_num
    c_cols = cat_cols or default_cat

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), c_cols),
            ("num", SimpleImputer(strategy="median"), n_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )
    return pipeline


def build_histgbm_pipeline(
    learning_rate: float = 0.05,
    max_iter: int = 100,
    max_depth: Optional[int] = 8,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    random_state: int = 42,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> Pipeline:
    """
    Constructs a HistGradientBoostingRegressor pipeline:
    - OrdinalEncoder on City (handle_unknown='use_encoded_value', unknown_value=np.nan)
    - Native passthrough for numerical features (histograms handle NaNs natively)
    - HistGradientBoostingRegressor with categorical_features=[0]
    """
    default_num, default_cat, _ = get_feature_columns()
    n_cols = num_cols or default_num
    c_cols = cat_cols or default_cat

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                c_cols,
            ),
            ("num", "passthrough", n_cols),
        ]
    )

    # City is transformed into column index 0
    model = HistGradientBoostingRegressor(
        categorical_features=[0],
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )
    return pipeline
