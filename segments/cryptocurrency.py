"""
💰 Cryptocurrency Segment
Live data from CoinGecko API with market analysis and forecasting.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from config.settings import POPULAR_CRYPTOS, MODELS
from utils.api_clients import fetch_crypto_market_chart, fetch_crypto_current_price, fetch_trending_coins
from utils.data_processing import (
    clean_timeseries, compute_rolling_stats, decompose_timeseries,
    detect_anomalies, compute_stationarity_test, prepare_forecast_data,
)
from utils.visualization import (
    plot_timeseries, plot_forecast, plot_decomposition,
    plot_anomalies, plot_distribution, plot_correlation_heatmap,
    metric_card_html,
)
from utils.metrics import compute_all_metrics


def render():
    """Render the Cryptocurrency segment."""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1A1F2E 0%, #1B3A2D 100%);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 24px;
        border: 1px solid rgba(67,233,123,0.3);
    ">
        <h2 style="margin:0; color:#FAFAFA;">💰 Cryptocurrency Analysis</h2>
        <p style="color:#9CA3AF; margin-top:8px; margin-bottom:0;">
            Live crypto data from CoinGecko API • Market Analysis • Price Forecasting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Sidebar Controls ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 💰 Crypto Settings")
        
        coin_name = st.selectbox("Select Cryptocurrency", list(POPULAR_CRYPTOS.keys()))
        coin_id = POPULAR_CRYPTOS[coin_name]
        
        vs_currency = st.selectbox("vs Currency", ["usd", "eur", "gbp", "inr", "jpy"], index=0)
        currency_symbol = {"usd": "$", "eur": "€", "gbp": "£", "inr": "₹", "jpy": "¥"}.get(vs_currency, "$")
        
        days = st.selectbox("History Period", [30, 90, 180, 365, 730], index=3, format_func=lambda x: f"{x} days")
        
        st.markdown("---")
        st.markdown("### 🔮 Forecast Settings")
        forecast_days = st.slider("Forecast Horizon (days)", 7, 60, 14, key="crypto_forecast")
        
        selected_models = st.multiselect(
            "Select Models",
            list(MODELS.keys()),
            default=["ARIMA", "Exponential Smoothing"],
            key="crypto_models",
        )
    
    # ── Fetch Data ────────────────────────────────────────────────────────
    try:
        with st.spinner(f"Fetching {coin_name} data..."):
            df = fetch_crypto_market_chart(coin_id, vs_currency, days)
        
        if df is None or df.empty:
            st.error("No data returned from CoinGecko.")
            return
    
    except Exception as e:
        st.warning(f"⚠️ **CoinGecko API Unreachable:** {str(e)}. Falling back to highly realistic simulated crypto data so you can continue exploring the models.")
        
        np.random.seed(hash(coin_id) % 2**31)
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
        base = {"bitcoin": 45000, "ethereum": 3000, "solana": 100}.get(coin_id, 1000)
        trend = np.linspace(0, base * 0.3, days)
        noise = np.random.normal(0, base * 0.03, days)
        seasonal = base * 0.1 * np.sin(2 * np.pi * np.arange(days) / 90)
        price = base + trend + noise + seasonal
        
        df = pd.DataFrame({
            "Price": price,
            "Market_Cap": price * 19e6,
            "Volume": np.random.uniform(1e9, 5e10, days),
        }, index=dates)
    
    # ── Key Metrics ───────────────────────────────────────────────────────
    current_price = df["Price"].iloc[-1]
    prev_price = df["Price"].iloc[-2]
    change_24h = ((current_price - prev_price) / prev_price) * 100
    
    high_price = df["Price"].max()
    low_price = df["Price"].min()
    avg_volume = df["Volume"].mean() if "Volume" in df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(metric_card_html(
            f"{coin_name} Price",
            f"{currency_symbol}{current_price:,.2f}",
            f"{change_24h:+.2f}%",
            "green" if change_24h >= 0 else "red",
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(metric_card_html(
            f"{days}d High",
            f"{currency_symbol}{high_price:,.2f}",
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(metric_card_html(
            f"{days}d Low",
            f"{currency_symbol}{low_price:,.2f}",
        ), unsafe_allow_html=True)
    
    with col4:
        if avg_volume > 1e9:
            vol_str = f"{currency_symbol}{avg_volume/1e9:.1f}B"
        elif avg_volume > 1e6:
            vol_str = f"{currency_symbol}{avg_volume/1e6:.1f}M"
        else:
            vol_str = f"{currency_symbol}{avg_volume:,.0f}"
        st.markdown(metric_card_html("Avg Volume", vol_str), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Price Chart", "🔮 Forecast", "🔬 Analysis", "📋 Data",
    ])
    
    # ── Tab 1: Price Chart ────────────────────────────────────────────────
    with tab1:
        # Price chart with moving averages
        rolling_df = compute_rolling_stats(df, "Price", windows=[7, 30, 90])
        
        cols_to_plot = ["Price"]
        show_ma = st.multiselect("Overlay Moving Averages", ["MA_7", "MA_30", "MA_90"], default=["MA_30"])
        cols_to_plot.extend([ma for ma in show_ma if ma in rolling_df.columns])
        
        price_fig = plot_timeseries(
            rolling_df, columns=cols_to_plot,
            title=f"{coin_name} Price ({vs_currency.upper()})",
            fill=True,
        )
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Volume
        if "Volume" in df.columns:
            vol_fig = plot_timeseries(
                df, columns=["Volume"],
                title="Trading Volume", height=250,
                fill=True, colors=["#38F9D7"],
            )
            st.plotly_chart(vol_fig, use_container_width=True)
        
        # Market Cap
        if "Market_Cap" in df.columns:
            cap_fig = plot_timeseries(
                df, columns=["Market_Cap"],
                title="Market Capitalization", height=250,
                fill=True, colors=["#F9D423"],
            )
            st.plotly_chart(cap_fig, use_container_width=True)
    
    # ── Tab 2: Forecast ───────────────────────────────────────────────────
    with tab2:
        series = df["Price"]
        
        if not selected_models:
            st.info("Select at least one model from the sidebar.")
            return
        
        all_results = {}
        
        for model_name in selected_models:
            with st.spinner(f"Training {model_name} on {coin_name}..."):
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
                series.tail(60),
                result["forecast"],
                result.get("lower_bound"),
                result.get("upper_bound"),
                title=f"{coin_name} — {model_name} ({forecast_days}d forecast)",
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Metrics comparison
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
        
        st.markdown("---")
        st.markdown("### 🤖 AI Financial Analyst Report")
        report_key = f"ai_report_crypto_{coin_id}_{vs_currency}"
        
        if st.button("Generate Hedge Fund Report (Powered by ChatGPT)", type="primary", key="crypto_report_btn"):
            from utils.llm_report import generate_financial_report
            with st.spinner("Analyzing crypto trajectory with OpenAI..."):
                st.session_state[report_key] = generate_financial_report(f"{coin_name} ({vs_currency.upper()})", df, all_results)
                st.session_state[f"{report_key}_time"] = datetime.now().strftime('%H:%M:%S')
                
        if report_key in st.session_state:
            st.markdown(f"> **AI Report Generated at {st.session_state[f'{report_key}_time']}**")
            st.markdown(st.session_state[report_key])
            st.download_button(
                label="📥 Download Full Report (.md)",
                data=st.session_state[report_key],
                file_name=f"{coin_name}_AI_Financial_Report.md",
                mime="text/markdown",
            )
    
    # ── Tab 3: Analysis ───────────────────────────────────────────────────
    with tab3:
        series = df["Price"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Price Distribution")
            returns = series.pct_change().dropna() * 100
            dist_fig = plot_distribution(returns, title="Daily Returns (%)")
            st.plotly_chart(dist_fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Stationarity Test")
            adf = compute_stationarity_test(series)
            if "error" not in adf:
                st.metric("ADF Statistic", f"{adf['test_statistic']:.4f}")
                st.metric("P-Value", f"{adf['p_value']:.6f}")
                st.metric("Stationary?", "✅ Yes" if adf["is_stationary"] else "❌ No")
        
        # Decomposition
        st.markdown("#### Time Series Decomposition")
        try:
            decomp = decompose_timeseries(series)
            decomp_fig = plot_decomposition(decomp, title=f"{coin_name} Decomposition")
            st.plotly_chart(decomp_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Decomposition failed: {e}")
        
        # Anomaly detection
        st.markdown("#### Anomaly Detection")
        anomalies = detect_anomalies(series, method="zscore")
        anom_fig = plot_anomalies(series, anomalies, title=f"Anomalies ({anomalies.sum()} detected)")
        st.plotly_chart(anom_fig, use_container_width=True)
        
        # Correlation
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            st.markdown("#### Correlation Matrix")
            corr_fig = plot_correlation_heatmap(df, title="Feature Correlations")
            st.plotly_chart(corr_fig, use_container_width=True)
    
    # ── Tab 4: Data ───────────────────────────────────────────────────────
    with tab4:
        st.dataframe(df.tail(100).sort_index(ascending=False), use_container_width=True)
        csv = df.to_csv()
        st.download_button("📥 Download CSV", csv, f"{coin_id}_crypto_data.csv", "text/csv")
