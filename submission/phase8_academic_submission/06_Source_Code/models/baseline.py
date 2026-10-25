"""
Naive Persistence Baseline Model for Next-Day AQI Forecasting.

Mathematical Formulation:
    hat{AQI}_{c, t+1} = AQI_{c, t}

Rules & Constraints:
1. Persistence requires that current-day AQI_t is observed.
2. If AQI_t is NaN, the prediction is NaN (persistence cannot forecast).
3. The model is evaluated strictly on the shared cohort where both AQI_t
   and Target_AQI are observed. Missing AQI_t is never imputed.
"""

from typing import Union
import numpy as np
import pandas as pd


class NaivePersistenceModel:
    """
    Naive Persistence Forecaster:
    Predicts next-day AQI as current-day observed AQI.
    """

    def __init__(self, aqi_col: str = "AQI_t"):
        self.aqi_col = aqi_col
        self.is_fitted_ = True  # Baseline requires no parameter fitting

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """No-op fit method for scikit-learn compatibility."""
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generates persistence forecast: hat{AQI}_{t+1} = AQI_t.

        Parameters:
            X: DataFrame containing `self.aqi_col` (or column 0 if array).

        Returns:
            np.ndarray of predictions with NaNs preserved where AQI_t was missing.
        """
        if isinstance(X, pd.DataFrame):
            if self.aqi_col in X.columns:
                preds = X[self.aqi_col].to_numpy(dtype=float, copy=True)
            elif "AQI" in X.columns:
                preds = X["AQI"].to_numpy(dtype=float, copy=True)
            else:
                raise KeyError(f"Neither '{self.aqi_col}' nor 'AQI' found in DataFrame columns.")
        elif isinstance(X, np.ndarray):
            # Assumes first column is AQI_t if 2D array
            if X.ndim == 2:
                preds = X[:, 0].astype(float)
            else:
                preds = X.astype(float)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        return preds
