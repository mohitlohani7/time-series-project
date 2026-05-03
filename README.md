# ⏱️ TimeSeries Pro

A comprehensive, deployment-ready **Streamlit** web application that provides time series analysis and forecasting across **5 major segments**, using real-time data from free APIs.

## 🚀 Features

- **5 Segments:**
  - 📈 **Stock Market** (Alpha Vantage API)
  - 💰 **Cryptocurrency** (CoinGecko API)
  - 🌤️ **Weather** (OpenWeatherMap API & Synthetic data)
  - ⚡ **Energy** (EIA API & Synthetic data)
  - 📂 **Custom CSV** (Upload your own data)
- **Advanced Forecasting Models:**
  - ARIMA / SARIMA
  - Facebook Prophet
  - LSTM (Neural Network)
  - Holt-Winters Exponential Smoothing
- **Interactive UI:** Dark mode, Plotly charts, Anomaly Detection, and technical indicators.

## 🛠️ Installation & Usage

1. **Clone/Download the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## 🔑 API Keys

To use real data, you need free API keys for the following services. You can enter them in the UI sidebar or set them up in `.streamlit/secrets.toml`:

- **Alpha Vantage:** [Get Free Key](https://www.alphavantage.co/support/#api-key)
- **OpenWeatherMap:** [Get Free Key](https://home.openweathermap.org/users/sign_up)
- **EIA (US Energy):** [Get Free Key](https://www.eia.gov/opendata/register.php)

(Note: CoinGecko works without an API key for the free tier).

## 🚀 Deployment

This project is fully ready to be deployed on **Streamlit Community Cloud**.
1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and link your GitHub repo.
3. Add your API keys to the Streamlit Cloud Secrets management.
4. Click Deploy!
