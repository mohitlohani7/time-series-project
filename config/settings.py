"""
Configuration and settings for the Time Series Platform.
API keys should be set via Streamlit secrets or environment variables.
"""
import os
import streamlit as st

# ---------------------------------------------------------------------------
# API Keys — pulled from Streamlit secrets first, then environment variables
# ---------------------------------------------------------------------------

def get_secret(key: str, default: str = "") -> str:
    """Retrieve a secret from Streamlit secrets or environment."""
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


# Alpha Vantage (Stock Market)
ALPHA_VANTAGE_API_KEY = get_secret("ALPHA_VANTAGE_API_KEY", "demo")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# CoinGecko (Cryptocurrency) — no key needed for free tier
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# OpenWeatherMap (Weather)
OPENWEATHER_API_KEY = get_secret("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
OPENWEATHER_HISTORY_URL = "https://history.openweathermap.org/data/2.5"

# OpenAI (ChatGPT)
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Application Constants
# ---------------------------------------------------------------------------

APP_TITLE = "⏱️ TimeSeries Pro"
APP_SUBTITLE = "Multi-Segment Time Series Forecasting Platform"
APP_VERSION = "1.0.0"

# Default forecast horizon
DEFAULT_FORECAST_DAYS = 30

# Supported models
MODELS = {
    "ARIMA": "Auto-Regressive Integrated Moving Average",
    "Prophet": "Facebook Prophet — handles seasonality & holidays",
    "LSTM": "Long Short-Term Memory neural network",
    "Exponential Smoothing": "Holt-Winters triple exponential smoothing",
}

# Segment definitions
SEGMENTS = {
    "📈 Stock Market": "stock_market",
    "💰 Cryptocurrency": "cryptocurrency",
    "🌤️ Weather": "weather",
    "💱 Forex (Currencies)": "forex",
    "📂 Custom CSV": "custom_csv",
}

# Color palette for charts
CHART_COLORS = {
    "primary": "#6C63FF",
    "secondary": "#FF6584",
    "accent1": "#43E97B",
    "accent2": "#F9D423",
    "accent3": "#38F9D7",
    "background": "#0E1117",
    "card": "#1A1F2E",
    "text": "#FAFAFA",
    "grid": "#2D3446",
}

# Popular stock symbols
POPULAR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "V", "WMT",
    "DIS", "NFLX", "PYPL", "INTC", "AMD",
    "BA", "GS", "IBM", "ORCL", "CRM",
]

# Popular cryptocurrencies (CoinGecko IDs)
POPULAR_CRYPTOS = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Solana": "solana",
    "Cardano": "cardano",
    "Dogecoin": "dogecoin",
    "Polkadot": "polkadot",
    "Avalanche": "avalanche-2",
    "Chainlink": "chainlink",
    "Polygon": "matic-network",
    "Litecoin": "litecoin",
}

# Popular cities for weather
POPULAR_CITIES = [
    "New York", "London", "Tokyo", "Mumbai", "Sydney",
    "Paris", "Berlin", "Toronto", "Dubai", "Singapore",
    "Los Angeles", "Chicago", "Hong Kong", "Shanghai", "Seoul",
]

# Popular Fiat Currencies (Frankfurter API)
POPULAR_CURRENCIES = {
    "Euro": "EUR",
    "British Pound": "GBP",
    "Indian Rupee": "INR",
    "Japanese Yen": "JPY",
    "Australian Dollar": "AUD",
    "Canadian Dollar": "CAD",
    "Swiss Franc": "CHF",
    "Chinese Yuan": "CNY",
    "Singapore Dollar": "SGD",
}
