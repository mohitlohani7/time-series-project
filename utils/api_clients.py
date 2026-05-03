"""
API client wrappers for all data sources.
Each function handles requests, error handling, and response parsing.
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import time


# ============================================================================
# ALPHA VANTAGE — Stock Market
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_daily(symbol: str, api_key: str, outputsize: str = "full") -> pd.DataFrame:
    """
    Fetch daily stock prices from Alpha Vantage.
    Returns DataFrame with columns: Open, High, Low, Close, Volume
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key,
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if "Time Series (Daily)" not in data:
            if "Note" in data or "Information" in data:
                raise Exception("API rate limit reached. Please wait and try again.")
            if "Error Message" in data:
                raise Exception(f"Invalid symbol: {symbol}")
            raise Exception(f"Unexpected API response: {data}")
        
        ts = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.astype(float)
        return df
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error fetching stock data: {e}")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_intraday(symbol: str, api_key: str, interval: str = "5min") -> pd.DataFrame:
    """Fetch intraday stock prices from Alpha Vantage."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": "compact",
        "apikey": api_key,
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        key = f"Time Series ({interval})"
        if key not in data:
            if "Note" in data:
                raise Exception("API rate limit reached.")
            raise Exception(f"Unexpected response for intraday data")
        
        ts = data[key]
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.astype(float)
        return df
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {e}")


# ============================================================================
# COINGECKO — Cryptocurrency
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_crypto_market_chart(coin_id: str, vs_currency: str = "usd", days: int = 365) -> pd.DataFrame:
    """
    Fetch historical price data for a cryptocurrency from CoinGecko.
    Returns DataFrame with columns: Price, Market_Cap, Volume
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": "daily" if days > 90 else "",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "Price"])
        market_caps = pd.DataFrame(data["market_caps"], columns=["timestamp", "Market_Cap"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "Volume"])
        
        df = prices.merge(market_caps, on="timestamp").merge(volumes, on="timestamp")
        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("Date").drop("timestamp", axis=1)
        df = df.sort_index()
        return df
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error fetching crypto data: {e}")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_crypto_current_price(coin_ids: list, vs_currency: str = "usd") -> dict:
    """Fetch current price for multiple coins."""
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ",".join(coin_ids),
        "vs_currencies": vs_currency,
        "include_24hr_change": "true",
        "include_market_cap": "true",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {e}")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_trending_coins() -> list:
    """Fetch trending coins from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/search/trending"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [item["item"] for item in data.get("coins", [])]
    except Exception:
        return []


# ============================================================================
# OPENWEATHERMAP — Weather
# ============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather_forecast(city: str, api_key: str) -> pd.DataFrame:
    """
    Fetch 5-day / 3-hour weather forecast from OpenWeatherMap.
    Returns DataFrame with temp, humidity, wind_speed, description.
    """
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("cod") != "200":
            raise Exception(f"City not found: {city}")
        
        records = []
        for item in data["list"]:
            records.append({
                "DateTime": item["dt_txt"],
                "Temperature": item["main"]["temp"],
                "Feels_Like": item["main"]["feels_like"],
                "Humidity": item["main"]["humidity"],
                "Pressure": item["main"]["pressure"],
                "Wind_Speed": item["wind"]["speed"],
                "Description": item["weather"][0]["description"],
                "Clouds": item["clouds"]["all"],
            })
        
        df = pd.DataFrame(records)
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = df.set_index("DateTime")
        return df
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error fetching weather: {e}")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_current(city: str, api_key: str) -> dict:
    """Fetch current weather for a city."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {e}")


def generate_synthetic_weather(city: str, days: int = 365) -> pd.DataFrame:
    """
    Generate realistic synthetic weather data for demo purposes.
    Uses sinusoidal patterns with noise to simulate seasonal temperature.
    """
    np.random.seed(hash(city) % 2**31)
    
    # City-specific base temperatures
    city_temps = {
        "New York": (12, 15), "London": (11, 8), "Tokyo": (16, 12),
        "Mumbai": (28, 4), "Sydney": (18, 8), "Paris": (12, 10),
        "Berlin": (10, 12), "Toronto": (8, 16), "Dubai": (30, 8),
        "Singapore": (28, 2), "Los Angeles": (18, 8), "Chicago": (10, 16),
    }
    
    base_temp, amplitude = city_temps.get(city, (15, 12))
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    day_of_year = dates.dayofyear
    
    # Seasonal pattern with noise
    temperature = base_temp + amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temperature += np.random.normal(0, 2.5, days)
    
    humidity = 60 + 20 * np.sin(2 * np.pi * (day_of_year - 170) / 365) + np.random.normal(0, 8, days)
    humidity = np.clip(humidity, 20, 100)
    
    wind_speed = 5 + 3 * np.sin(2 * np.pi * day_of_year / 365) + np.random.exponential(2, days)
    
    pressure = 1013 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5, days)
    
    df = pd.DataFrame({
        "Temperature": temperature.round(1),
        "Humidity": humidity.round(1),
        "Wind_Speed": wind_speed.round(1),
        "Pressure": pressure.round(1),
    }, index=dates)
    df.index.name = "Date"
    
    return df


# ============================================================================
# FRANKFURTER — Forex / Currencies
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_forex_rates(base_currency: str = "USD", target_currency: str = "EUR", days: int = 365) -> pd.DataFrame:
    """
    Fetch historical exchange rates using the free Frankfurter API.
    Returns DataFrame with a 'Value' column.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    url = f"https://api.frankfurter.app/{start_str}..{end_str}"
    params = {"from": base_currency, "to": target_currency}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if "rates" not in data:
            raise Exception("No rates returned.")
            
        rates = data["rates"]
        records = []
        for date_str, values in rates.items():
            if target_currency in values:
                records.append({
                    "Date": date_str,
                    "Value": values[target_currency]
                })
                
        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        
        # Fill missing weekend days using forward fill
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        df = df.reindex(full_index).ffill()
        df.index.name = "Date"
        
        return df
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error fetching forex data: {e}")
