#!/usr/bin/env python3
"""A collection of simple helper functions for
building and using linear models."""
from sklearn import metrics
import numpy as np


def evaluation_metrics_for_regression(y_true, y_pred):
    """Calculate and return four common scores MSE, RMSE, MAE, and R²
    to see how well your model's predictions match the real values."""
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return (mse, rmse, mae, r2)
