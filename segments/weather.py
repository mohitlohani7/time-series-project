"""
🌤️ Weather Segment
Weather data analysis with OpenWeatherMap API and synthetic historical data.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from config.settings import OPENWEATHER_API_KEY, POPULAR_CITIES, MODELS
from utils.api_clients import fetch_weather_forecast, generate_synthetic_weather
from utils.data_processing import (
    decompose_timeseries, detect_anomalies, compute_stationarity_test,
    compute_rolling_stats,
)
from utils.visualization import (
    plot_timeseries, plot_forecast, plot_decomposition,
    plot_anomalies, plot_distribution, plot_correlation_heatmap,
    metric_card_html,
)
from utils.metrics import compute_all_metrics


def render():
    """Render the Weather segment."""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1A1F2E 0%, #1B2D3A 100%);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 24px;
        border: 1px solid rgba(56,249,215,0.3);
    ">
        <h2 style="margin:0; color:#FAFAFA;">🌤️ Weather Analysis</h2>
        <p style="color:#9CA3AF; margin-top:8px; margin-bottom:0;">
            Weather forecasting powered by OpenWeatherMap • Temperature • Humidity • Wind Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Sidebar Controls ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🌤️ Weather Settings")
        
        api_key = st.text_input(
            "OpenWeatherMap API Key",
            value=OPENWEATHER_API_KEY,
            type="password",
            help="Get free key at openweathermap.org",
            key="weather_api_key",
        )
        
        city = st.selectbox("Select City", POPULAR_CITIES, index=0)
        custom_city = st.text_input("Or enter custom city:", placeholder="e.g. San Francisco")
        if custom_city:
            city = custom_city
        
        history_days = st.slider("Historical Data (days)", 90, 730, 365, key="weather_history")
        
        weather_metric = st.selectbox(
            "Primary Metric",
            ["Temperature", "Humidity", "Wind_Speed", "Pressure"],
            index=0,
        )
        
        st.markdown("---")
        st.markdown("### 🔮 Forecast Settings")
        forecast_days = st.slider("Forecast Horizon (days)", 7, 60, 30, key="weather_forecast")
        
        selected_models = st.multiselect(
            "Select Models",
            list(MODELS.keys()),
            default=["ARIMA", "Exponential Smoothing"],
            key="weather_models",
        )
    
    # ── Generate Historical Data ──────────────────────────────────────────
    # (OpenWeatherMap free tier doesn't include historical data, so we use
    #  synthetic data that matches the city's climate patterns)
    
    with st.spinner(f"Loading weather data for {city}..."):
        df = generate_synthetic_weather(city, days=history_days)
    
    # Try to get live forecast data
    live_forecast = None
    if api_key:
        try:
            live_forecast = fetch_weather_forecast(city, api_key)
        except Exception:
            pass
    
    # ── Key Metrics ───────────────────────────────────────────────────────
    latest = df.iloc[-1]
    avg_temp = df["Temperature"].mean()
    max_temp = df["Temperature"].max()
    min_temp = df["Temperature"].min()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(metric_card_html(
            f"{city} Current",
            f"{latest['Temperature']:.1f}°C",
            f"Avg: {avg_temp:.1f}°C",
            "green",
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(metric_card_html("Record High", f"{max_temp:.1f}°C"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(metric_card_html("Record Low", f"{min_temp:.1f}°C"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(metric_card_html(
            "Humidity",
            f"{latest['Humidity']:.0f}%",
        ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "🔮 Forecast", "🔬 Analysis", "📋 Data",
    ])
    
    # ── Tab 1: Overview ───────────────────────────────────────────────────
    with tab1:
        # Temperature chart
        rolling = compute_rolling_stats(df, "Temperature", windows=[7, 30])
        temp_cols = ["Temperature"]
        if "MA_7" in rolling.columns:
            temp_cols.append("MA_7")
        if "MA_30" in rolling.columns:
            temp_cols.append("MA_30")
        
        temp_fig = plot_timeseries(
            rolling, columns=temp_cols,
            title=f"{city} — Temperature History",
            fill=True,
            colors=["#FF6584", "#F9D423", "#6C63FF"],
        )
        st.plotly_chart(temp_fig, use_container_width=True)
        
        # Multi-metric view
        col1, col2 = st.columns(2)
        
        with col1:
            humidity_fig = plot_timeseries(
                df, columns=["Humidity"],
                title="Humidity (%)", height=300,
                fill=True, colors=["#38F9D7"],
            )
            st.plotly_chart(humidity_fig, use_container_width=True)
        
        with col2:
            wind_fig = plot_timeseries(
                df, columns=["Wind_Speed"],
                title="Wind Speed (m/s)", height=300,
                fill=True, colors=["#43E97B"],
            )
            st.plotly_chart(wind_fig, use_container_width=True)
        
        # Live 5-day forecast
        if live_forecast is not None:
            st.markdown("#### 🌐 Live 5-Day Forecast (OpenWeatherMap)")
            live_fig = plot_timeseries(
                live_forecast, columns=["Temperature", "Humidity"],
                title=f"{city} — Live 5-Day Forecast",
                colors=["#FF6584", "#38F9D7"],
            )
            st.plotly_chart(live_fig, use_container_width=True)
    
    # ── Tab 2: Forecast ───────────────────────────────────────────────────
    with tab2:
        series = df[weather_metric]
        
        if not selected_models:
            st.info("Select at least one model from the sidebar.")
            return
        
        all_results = {}
        
        for model_name in selected_models:
            with st.spinner(f"Training {model_name}..."):
                try:
                    if model_name == "ARIMA":
                        from models.arima_model import fit_arima
                        result = fit_arima(series, forecast_steps=forecast_days, auto_order=False)
                    elif model_name == "Prophet":
                        from models.prophet_model import fit_prophet
                        result = fit_prophet(series, forecast_steps=forecast_days)
                    elif model_name == "LSTM":
                        from models.lstm_model import fit_lstm
                        result = fit_lstm(series, forecast_steps=forecast_days, epochs=50)
                    elif model_name == "Exponential Smoothing":
                        from models.exp_smoothing import fit_exponential_smoothing
                        result = fit_exponential_smoothing(series, forecast_steps=forecast_days)
                    
                    all_results[model_name] = result
                except Exception as e:
                    st.warning(f"⚠️ {model_name}: {str(e)}")
        
        for model_name, result in all_results.items():
            st.markdown(f"#### {model_name} Forecast")
            fig = plot_forecast(
                series.tail(90),
                result["forecast"],
                result.get("lower_bound"),
                result.get("upper_bound"),
                title=f"{city} {weather_metric} — {model_name} ({forecast_days}d)",
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if len(all_results) > 1:
            st.markdown("### 🏆 Model Comparison")
            metrics = {}
            for model_name, result in all_results.items():
                fitted = result.get("fitted")
                if fitted is not None and len(fitted) > 10:
                    common_idx = fitted.index.intersection(series.index)
                    if len(common_idx) > 10:
                        metrics[model_name] = compute_all_metrics(
                            series[common_idx].values, fitted[common_idx].values
                        )
            if metrics:
                st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)
    
    # ── Tab 3: Analysis ───────────────────────────────────────────────────
    with tab3:
        series = df[weather_metric]
        
        # Decomposition
        st.markdown("#### Seasonal Decomposition")
        try:
            decomp = decompose_timeseries(series)
            decomp_fig = plot_decomposition(decomp, title=f"{weather_metric} Decomposition")
            st.plotly_chart(decomp_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Decomposition failed: {e}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Distribution")
            dist_fig = plot_distribution(series, title=f"{weather_metric} Distribution")
            st.plotly_chart(dist_fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Anomaly Detection")
            anomalies = detect_anomalies(series)
            anom_fig = plot_anomalies(series, anomalies, title=f"Anomalies ({anomalies.sum()} found)")
            st.plotly_chart(anom_fig, use_container_width=True)
        
        # Correlation
        st.markdown("#### Feature Correlations")
        corr_fig = plot_correlation_heatmap(df, title="Weather Feature Correlations")
        st.plotly_chart(corr_fig, use_container_width=True)
    
    # ── Tab 4: Data ───────────────────────────────────────────────────────
    with tab4:
        st.dataframe(df.tail(100).sort_index(ascending=False), use_container_width=True)
        csv = df.to_csv()
        st.download_button("📥 Download CSV", csv, f"{city}_weather_data.csv", "text/csv")
