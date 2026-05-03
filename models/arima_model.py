"""
ARIMA / SARIMA forecasting model.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")


def auto_arima_order(series: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> tuple:
    """
    Simple auto-selection of ARIMA order (p,d,q) using AIC.
    For production, consider using pmdarima.auto_arima.
    """
    best_aic = float("inf")
    best_order = (1, 1, 1)
    
    series = series.dropna()
    
    for d in range(max_d + 1):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(series, order=(p, d, q))
                    result = model.fit()
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    
    return best_order


def fit_arima(
    series: pd.Series,
    order: tuple = None,
    forecast_steps: int = 30,
    auto_order: bool = True,
) -> dict:
    """
    Fit ARIMA model and generate forecast.
    
    Returns:
        dict with keys: forecast, lower_bound, upper_bound, model_summary, order, aic
    """
    series = series.dropna().astype(float)
    
    if order is None and auto_order:
        # Expanded auto-selection to capture highly complex, volatile market shifts
        order = auto_arima_order(series, max_p=5, max_d=2, max_q=5)
    elif order is None:
        order = (1, 1, 1)
    
    try:
        model = ARIMA(series, order=order)
        result = model.fit()
        
        # Forecast
        forecast_result = result.get_forecast(steps=forecast_steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)
        
        # Generate future dates
        last_date = series.index[-1]
        freq = pd.infer_freq(series.index)
        if freq is None:
            freq = "D"
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq=freq)
        
        forecast_series = pd.Series(forecast_mean.values, index=future_dates, name="Forecast")
        lower = pd.Series(conf_int.iloc[:, 0].values, index=future_dates, name="Lower")
        upper = pd.Series(conf_int.iloc[:, 1].values, index=future_dates, name="Upper")
        
        # In-sample fitted values
        fitted = result.fittedvalues
        
        return {
            "forecast": forecast_series,
            "lower_bound": lower,
            "upper_bound": upper,
            "fitted": fitted,
            "model_summary": str(result.summary()),
            "order": order,
            "aic": round(result.aic, 2),
            "bic": round(result.bic, 2),
        }
    except Exception as e:
        raise Exception(f"ARIMA fitting failed: {str(e)}")


def fit_sarima(
    series: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12),
    forecast_steps: int = 30,
) -> dict:
    """
    Fit SARIMA model with seasonal component.
    """
    series = series.dropna().astype(float)
    
    try:
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False, maxiter=200)
        
        forecast_result = result.get_forecast(steps=forecast_steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)
        
        last_date = series.index[-1]
        freq = pd.infer_freq(series.index)
        if freq is None:
            freq = "D"
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq=freq)
        
        forecast_series = pd.Series(forecast_mean.values, index=future_dates, name="Forecast")
        lower = pd.Series(conf_int.iloc[:, 0].values, index=future_dates, name="Lower")
        upper = pd.Series(conf_int.iloc[:, 1].values, index=future_dates, name="Upper")
        
        fitted = result.fittedvalues
        
        return {
            "forecast": forecast_series,
            "lower_bound": lower,
            "upper_bound": upper,
            "fitted": fitted,
            "model_summary": str(result.summary()),
            "order": order,
            "seasonal_order": seasonal_order,
            "aic": round(result.aic, 2),
        }
    except Exception as e:
        raise Exception(f"SARIMA fitting failed: {str(e)}")
