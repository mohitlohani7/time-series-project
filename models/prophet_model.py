"""
Facebook Prophet forecasting model.
"""
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def fit_prophet(
    series: pd.Series,
    forecast_steps: int = 30,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    changepoint_prior_scale: float = 0.15,  # Increased from 0.05 to better track sudden volatility
    seasonality_prior_scale: float = 15.0,  # Increased for stronger seasonal fitting
    interval_width: float = 0.95,
) -> dict:
    """
    Fit Facebook Prophet model and generate forecast.
    
    Args:
        series: Time series with DatetimeIndex
        forecast_steps: Number of periods to forecast
        yearly_seasonality: Enable yearly seasonality
        weekly_seasonality: Enable weekly seasonality
        daily_seasonality: Enable daily seasonality
        changepoint_prior_scale: Flexibility of the trend (higher = more flexible)
        seasonality_prior_scale: Strength of seasonality
        interval_width: Width of uncertainty interval
    
    Returns:
        dict with forecast, bounds, components
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Prophet is not installed. Run: pip install prophet")
    
    series = series.dropna().astype(float)
    
    # Prepare data in Prophet format (ds, y)
    prophet_df = pd.DataFrame({
        "ds": series.index,
        "y": series.values,
    })
    
    # Initialize and fit model
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        interval_width=interval_width,
    )
    
    # Add holidays to improve predictions
    model.add_country_holidays(country_name='US')
    
    # Suppress Stan output
    model.fit(prophet_df)
    
    # Create future DataFrame
    future = model.make_future_dataframe(periods=forecast_steps)
    forecast = model.predict(future)
    
    # Extract forecast for future periods only
    forecast_future = forecast.iloc[-forecast_steps:]
    forecast_series = pd.Series(
        forecast_future["yhat"].values,
        index=pd.to_datetime(forecast_future["ds"].values),
        name="Forecast",
    )
    lower = pd.Series(
        forecast_future["yhat_lower"].values,
        index=forecast_series.index,
        name="Lower",
    )
    upper = pd.Series(
        forecast_future["yhat_upper"].values,
        index=forecast_series.index,
        name="Upper",
    )
    
    # Fitted values (in-sample)
    fitted = pd.Series(
        forecast.iloc[:-forecast_steps]["yhat"].values,
        index=series.index[:len(forecast) - forecast_steps],
        name="Fitted",
    )
    
    # Components
    components = {}
    if yearly_seasonality and "yearly" in forecast.columns:
        components["yearly"] = forecast.set_index("ds")["yearly"]
    if weekly_seasonality and "weekly" in forecast.columns:
        components["weekly"] = forecast.set_index("ds")["weekly"]
    components["trend"] = forecast.set_index("ds")["trend"]
    
    return {
        "forecast": forecast_series,
        "lower_bound": lower,
        "upper_bound": upper,
        "fitted": fitted,
        "components": components,
        "full_forecast": forecast,
        "model": model,
    }
