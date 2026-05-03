"""
Forex (Currencies) Analysis Segment.
Fetches historical exchange rates from Frankfurter API and provides forecasting.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from config.settings import POPULAR_CURRENCIES
from utils.api_clients import fetch_forex_rates
from utils.data_processing import (
    compute_returns, compute_technical_indicators,
    compute_stationarity_test, decompose_timeseries, detect_anomalies
)
from utils.visualization import (
    plot_timeseries, plot_forecast, plot_decomposition,
    plot_anomalies, metric_card_html
)

def render():
    st.header("💱 Forex Analysis")
    st.markdown("Exchange rate forecasting powered by the open-source Frankfurter API")
    
    # -----------------------------------------------------------------------
    # Sidebar Controls
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.subheader("💱 Forex Settings")
        
        # Pre-defined popular currency pairs
        currency_pairs = {
            "USD to EUR (Euro)": ("USD", "EUR"),
            "USD to INR (Indian Rupee)": ("USD", "INR"),
            "USD to GBP (British Pound)": ("USD", "GBP"),
            "USD to JPY (Japanese Yen)": ("USD", "JPY"),
            "EUR to GBP (British Pound)": ("EUR", "GBP"),
            "EUR to JPY (Japanese Yen)": ("EUR", "JPY"),
            "GBP to INR (Indian Rupee)": ("GBP", "INR"),
        }
        
        selected_pair_label = st.selectbox("Select Currency Pair", options=list(currency_pairs.keys()), index=1) # Default to USD/INR based on earlier context
        base_currency, target_code = currency_pairs[selected_pair_label]
        
        history_days = st.slider("Historical Data (days)", min_value=90, max_value=1825, value=730, step=30)
        
        st.subheader("🔮 Forecast Settings")
        forecast_days = st.number_input("Forecast Horizon (days)", min_value=7, max_value=365, value=30)
        
        model_name = st.selectbox(
            "Select Forecasting Model",
            ["ARIMA", "Facebook Prophet", "LSTM", "Exponential Smoothing"]
        )
        
        run_forecast = st.button("Run Forecast", use_container_width=True, type="primary")

    if base_currency == target_code:
        st.warning("Base and Target currencies cannot be the same.")
        return

    # -----------------------------------------------------------------------
    # Data Fetching
    # -----------------------------------------------------------------------
    df = None
    try:
        with st.spinner(f"Fetching {base_currency}/{target_code} exchange rates..."):
            df = fetch_forex_rates(base_currency, target_code, history_days)
    except Exception as e:
        st.warning(f"⚠️ **API Unreachable:** {str(e)}. Falling back to simulated data.")
        np.random.seed(hash(target_code) % 2**31)
        dates = pd.date_range(end=datetime.now(), periods=history_days, freq="D")
        base_rate = 1.2 if target_code == "EUR" else 80.0
        prices = base_rate + np.cumsum(np.random.normal(0, 0.01 * base_rate, history_days))
        df = pd.DataFrame({"Value": prices}, index=dates)
        df.index.name = "Date"

    if df is None or df.empty:
        st.error("No data available.")
        return

    # Compute features
    df["Return"] = compute_returns(df, "Value")
    tech_df = df.copy()
    tech_df["MA_7"] = tech_df["Value"].rolling(window=7).mean()
    tech_df["MA_30"] = tech_df["Value"].rolling(window=30).mean()
    
    # -----------------------------------------------------------------------
    # Top KPI Metrics
    # -----------------------------------------------------------------------
    current_price = df["Value"].iloc[-1]
    prev_price = df["Value"].iloc[-2]
    pct_change = ((current_price - prev_price) / prev_price) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card_html(
            f"{base_currency}/{target_code} CURRENT", 
            f"{current_price:.4f}", 
            f"{pct_change:.2f}% (24h)"
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card_html("RECORD HIGH", f"{df['Value'].max():.4f}", ""), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card_html("RECORD LOW", f"{df['Value'].min():.4f}", ""), unsafe_allow_html=True)
    with col4:
        volatility = df["Return"].std() * np.sqrt(252) * 100
        st.markdown(metric_card_html("ANNUAL VOLATILITY", f"{volatility:.1f}%", ""), unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # -----------------------------------------------------------------------
    # Main Tabs
    # -----------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Forecast", "🔬 Analysis", "📄 Data"])
    
    with tab1:
        st.subheader(f"{base_currency}/{target_code} — Exchange Rate History")
        fig = plot_timeseries(tech_df, ["Value", "MA_7", "MA_30"], f"{base_currency}/{target_code} Rate")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        if run_forecast:
            st.subheader(f"{model_name} Forecast")
            series = df["Value"]
            
            with st.spinner(f"Training {model_name} model..."):
                try:
                    if model_name == "ARIMA":
                        from models.arima_model import fit_arima
                        result = fit_arima(series, forecast_steps=forecast_days)
                    elif model_name == "Facebook Prophet":
                        from models.prophet_model import fit_prophet
                        result = fit_prophet(series, forecast_steps=forecast_days)
                    elif model_name == "LSTM":
                        from models.lstm_model import fit_lstm
                        result = fit_lstm(series, forecast_steps=forecast_days, epochs=50)
                    elif model_name == "Exponential Smoothing":
                        from models.exp_smoothing import fit_exponential_smoothing
                        result = fit_exponential_smoothing(series, forecast_steps=forecast_days)
                    
                    fig = plot_forecast(series, result["forecast"], result.get("lower_bound"), result.get("upper_bound"), f"{base_currency}/{target_code} — {model_name} ({forecast_days}d)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if "params" in result:
                        st.write("**Model Parameters:**", result["params"])
                        
                    st.markdown("---")
                    st.markdown("### 🤖 AI Financial Analyst Report")
                    report_key = f"ai_report_forex_{base_currency}_{target_code}"
                    
                    if st.button("Generate Hedge Fund Report (Powered by ChatGPT)", type="primary", key="forex_report_btn"):
                        from utils.llm_report import generate_financial_report
                        with st.spinner("Analyzing forex trajectory with OpenAI..."):
                            st.session_state[report_key] = generate_financial_report(f"{base_currency}/{target_code} Forex Pair", df, {model_name: result})
                            st.session_state[f"{report_key}_time"] = datetime.now().strftime('%H:%M:%S')
                            
                    if report_key in st.session_state:
                        st.markdown(f"> **AI Report Generated at {st.session_state[f'{report_key}_time']}**")
                        st.markdown(st.session_state[report_key])
                        st.download_button(
                            label="📥 Download Full Report (.md)",
                            data=st.session_state[report_key],
                            file_name=f"{base_currency}_{target_code}_AI_Financial_Report.md",
                            mime="text/markdown",
                        )
                except Exception as e:
                    st.error(f"Forecasting error: {e}")
        else:
            st.info("👆 Click 'Run Forecast' in the sidebar to generate predictions.")
            
    with tab3:
        st.subheader("Statistical Analysis")
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown("#### Stationarity Test (ADF)")
            adf_res = compute_stationarity_test(df["Value"])
            if "error" in adf_res:
                st.warning(adf_res["error"])
            else:
                st.write(f"**ADF Statistic:** {adf_res['test_statistic']}")
                st.write(f"**p-value:** {adf_res['p_value']}")
                st.write(f"**Is Stationary:** {'Yes ✅' if adf_res['is_stationary'] else 'No ❌'}")
            
        with colB:
            st.markdown("#### Anomaly Detection")
            anomalies = detect_anomalies(df["Value"])
            anomaly_count = anomalies.sum()
            st.write(f"**Anomalies Detected:** {anomaly_count}")
            if anomaly_count > 0:
                fig_anom = plot_anomalies(df["Value"], anomalies, "Exchange Rate Anomalies")
                st.plotly_chart(fig_anom, use_container_width=True)
                
        st.markdown("#### Seasonal Decomposition")
        try:
            decomp = decompose_timeseries(df["Value"])
            fig_decomp = plot_decomposition(decomp)
            st.plotly_chart(fig_decomp, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not perform decomposition: {e}")
            
    with tab4:
        st.subheader("Raw Data")
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)
