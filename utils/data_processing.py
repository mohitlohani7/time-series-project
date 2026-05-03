"""
Data processing and feature engineering utilities.
"""
import pandas as pd
import numpy as np
from scipy import stats


def clean_timeseries(df: pd.DataFrame, value_col: str = None) -> pd.DataFrame:
    """
    Clean a time series DataFrame:
    - Ensure datetime index
    - Sort by date
    - Handle missing values with interpolation
    - Remove duplicates
    """
    df = df.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        else:
            # Try to parse the first column as dates
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass
    
    # Sort by index
    df = df.sort_index()
    
    # Remove duplicate indices
    df = df[~df.index.duplicated(keep="first")]
    
    # Interpolate missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].interpolate(method="time", limit_direction="both")
            df[col] = df[col].ffill().bfill()
    
    return df


def compute_returns(df: pd.DataFrame, col: str = "Close", periods: int = 1) -> pd.Series:
    """Compute percentage returns."""
    return df[col].pct_change(periods) * 100


def compute_rolling_stats(df: pd.DataFrame, col: str, windows: list = None) -> pd.DataFrame:
    """
    Compute rolling statistics (mean, std) for given windows.
    """
    if windows is None:
        windows = [7, 14, 30, 90]
    
    result = df[[col]].copy()
    for w in windows:
        if len(df) >= w:
            result[f"MA_{w}"] = df[col].rolling(window=w).mean()
            result[f"STD_{w}"] = df[col].rolling(window=w).std()
    
    return result


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute common technical indicators for stock/crypto data.
    Expects columns: Open, High, Low, Close, Volume
    """
    result = df.copy()
    
    # Moving Averages
    for period in [7, 20, 50, 200]:
        if len(df) >= period:
            result[f"SMA_{period}"] = df["Close"].rolling(window=period).mean()
            result[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    if len(df) >= 14:
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    if len(df) >= 26:
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        result["MACD"] = ema12 - ema26
        result["MACD_Signal"] = result["MACD"].ewm(span=9, adjust=False).mean()
        result["MACD_Hist"] = result["MACD"] - result["MACD_Signal"]
    
    # Bollinger Bands
    if len(df) >= 20:
        sma20 = df["Close"].rolling(window=20).mean()
        std20 = df["Close"].rolling(window=20).std()
        result["BB_Upper"] = sma20 + (std20 * 2)
        result["BB_Lower"] = sma20 - (std20 * 2)
        result["BB_Middle"] = sma20
    
    # ATR (Average True Range)
    if len(df) >= 14:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result["ATR"] = true_range.rolling(window=14).mean()
    
    # Daily Returns
    result["Returns"] = df["Close"].pct_change() * 100
    
    # Volume Moving Average
    if "Volume" in df.columns and len(df) >= 20:
        result["Volume_MA20"] = df["Volume"].rolling(window=20).mean()
    
    return result


def decompose_timeseries(series: pd.Series, period: int = None) -> dict:
    """
    Decompose time series into trend, seasonal, and residual components.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    if period is None:
        # Auto-detect period
        n = len(series)
        if n >= 730:
            period = 365
        elif n >= 60:
            period = 30
        elif n >= 14:
            period = 7
        else:
            period = max(2, n // 3)
    
    # Ensure no missing values
    series = series.dropna()
    
    if len(series) < 2 * period:
        period = max(2, len(series) // 3)
    
    result = seasonal_decompose(series, model="additive", period=period)
    
    return {
        "observed": result.observed,
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
    }


def detect_anomalies(series: pd.Series, method: str = "zscore", threshold: float = 3.0) -> pd.Series:
    """
    Detect anomalies in a time series.
    Returns a boolean Series (True = anomaly).
    """
    if method == "zscore":
        z_scores = np.abs(stats.zscore(series.dropna()))
        anomalies = pd.Series(False, index=series.index)
        anomalies[series.dropna().index] = z_scores > threshold
    elif method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        anomalies = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
    else:
        anomalies = pd.Series(False, index=series.index)
    
    return anomalies


def compute_stationarity_test(series: pd.Series) -> dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    """
    from statsmodels.tsa.stattools import adfuller
    
    series = series.dropna()
    if len(series) < 20:
        return {"error": "Not enough data points for stationarity test"}
    
    result = adfuller(series, autolag="AIC")
    
    return {
        "test_statistic": round(result[0], 4),
        "p_value": round(result[1], 6),
        "lags_used": result[2],
        "observations": result[3],
        "critical_values": {k: round(v, 4) for k, v in result[4].items()},
        "is_stationary": result[1] < 0.05,
    }


def prepare_forecast_data(df: pd.DataFrame, target_col: str, train_ratio: float = 0.8) -> tuple:
    """
    Split time series into train and test sets.
    Returns (train, test)
    """
    n = len(df)
    split_idx = int(n * train_ratio)
    
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    return train, test
