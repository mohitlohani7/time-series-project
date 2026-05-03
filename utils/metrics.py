"""
Evaluation metrics for time series forecasting models.
"""
import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    actual, predicted = np.array(actual), np.array(predicted)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    actual, predicted = np.array(actual), np.array(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    actual, predicted = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    # Avoid division by zero
    mask = actual != 0
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    actual, predicted = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    denominator = (np.abs(actual) + np.abs(predicted))
    mask = denominator != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(2.0 * np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100)


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    actual, predicted = np.array(actual), np.array(predicted)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


def compute_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute all evaluation metrics and return as a dictionary."""
    return {
        "MAE": round(mae(actual, predicted), 4),
        "RMSE": round(rmse(actual, predicted), 4),
        "MAPE (%)": round(mape(actual, predicted), 2),
        "sMAPE (%)": round(smape(actual, predicted), 2),
        "R²": round(r_squared(actual, predicted), 4),
    }


def format_metrics_table(metrics_dict: dict) -> pd.DataFrame:
    """
    Format metrics from multiple models into a comparison DataFrame.
    metrics_dict: {model_name: {metric: value, ...}, ...}
    """
    df = pd.DataFrame(metrics_dict).T
    df.index.name = "Model"
    return df
