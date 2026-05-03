"""
Exponential Smoothing (Holt-Winters) forecasting model.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")


def fit_exponential_smoothing(
    series: pd.Series,
    forecast_steps: int = 30,
    seasonal: str = "mul",  # Upgraded to Multiplicative to handle explosive crypto/stock rallies
    seasonal_periods: int = None,
    trend: str = "mul",     # Upgraded to Multiplicative for compound growth tracking
    damped_trend: bool = True,
    use_boxcox: bool = False,
) -> dict:
    """
    Fit Holt-Winters Exponential Smoothing model.
    
    Args:
        series: Time series with DatetimeIndex
        forecast_steps: Number of periods to forecast
        seasonal: Type of seasonal component ('add', 'mul', or None)
        seasonal_periods: Number of periods in a seasonal cycle
        trend: Type of trend component ('add', 'mul', or None)
        damped_trend: Whether to use damped trend
        use_boxcox: Whether to apply Box-Cox transformation
    
    Returns:
        dict with forecast, bounds, model parameters
    """
    series = series.dropna().astype(float)
    
    # Auto-detect seasonal period
    if seasonal_periods is None:
        n = len(series)
        freq = pd.infer_freq(series.index)
        if freq and "D" in str(freq):
            seasonal_periods = 7  # Weekly seasonality for daily data
        elif freq and "H" in str(freq):
            seasonal_periods = 24  # Daily seasonality for hourly data
        elif freq and "M" in str(freq):
            seasonal_periods = 12  # Yearly seasonality for monthly data
        else:
            seasonal_periods = 7
    
    # Ensure enough data for seasonal decomposition
    if len(series) < 2 * seasonal_periods:
        seasonal = None
        seasonal_periods = None
    
    # Ensure positive values for multiplicative model
    if seasonal == "mul" or trend == "mul":
        if (series <= 0).any():
            seasonal = "add" if seasonal else None
            trend = "add" if trend else None
    
    try:
        model = ExponentialSmoothing(
            series,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            trend=trend,
            damped_trend=damped_trend if trend else False,
            use_boxcox=use_boxcox,
            initialization_method="estimated",
        )
        
        result = model.fit(optimized=True)
        
        # Forecast
        forecast_values = result.forecast(forecast_steps)
        
        # Generate future dates
        last_date = series.index[-1]
        freq = pd.infer_freq(series.index) or "D"
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_steps,
            freq=freq,
        )
        
        forecast_series = pd.Series(forecast_values.values, index=future_dates, name="Forecast")
        
        # Estimate confidence intervals using residuals
        fitted = result.fittedvalues
        residuals = series - fitted
        residual_std = residuals.std()
        
        # Expanding confidence interval for further out forecasts
        steps = np.arange(1, forecast_steps + 1)
        interval_width = 1.96 * residual_std * np.sqrt(steps / steps.mean())
        
        lower = pd.Series(
            forecast_series.values - interval_width,
            index=future_dates,
            name="Lower",
        )
        upper = pd.Series(
            forecast_series.values + interval_width,
            index=future_dates,
            name="Upper",
        )
        
        # Model parameters
        params = {
            "smoothing_level (alpha)": round(result.params.get("smoothing_level", 0), 4),
            "smoothing_trend (beta)": round(result.params.get("smoothing_trend", 0), 4),
            "smoothing_seasonal (gamma)": round(result.params.get("smoothing_seasonal", 0), 4),
            "damping_trend (phi)": round(result.params.get("damping_trend", 0), 4),
        }
        
        return {
            "forecast": forecast_series,
            "lower_bound": lower,
            "upper_bound": upper,
            "fitted": fitted,
            "residuals": residuals,
            "params": params,
            "aic": round(result.aic, 2) if hasattr(result, "aic") else None,
            "bic": round(result.bic, 2) if hasattr(result, "bic") else None,
            "sse": round(result.sse, 2),
        }
    except Exception as e:
        raise Exception(f"Exponential Smoothing fitting failed: {str(e)}")
