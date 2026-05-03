"""
📈 Stock Market Segment
Live data from Alpha Vantage API with technical analysis and forecasting.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from config.settings import ALPHA_VANTAGE_API_KEY, POPULAR_STOCKS, MODELS
from utils.api_clients import fetch_stock_daily
from utils.data_processing import (
    clean_timeseries, compute_technical_indicators,
    decompose_timeseries, detect_anomalies, compute_stationarity_test,
    prepare_forecast_data,
)
from utils.visualization import (
    plot_timeseries, plot_candlestick, plot_forecast,
    plot_decomposition, plot_anomalies, plot_correlation_heatmap,
    plot_distribution, metric_card_html,
)
from utils.metrics import compute_all_metrics


def render():
    """Render the Stock Market segment."""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1A1F2E 0%, #2D1B69 100%);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 24px;
        border: 1px solid rgba(108,99,255,0.3);
    ">
        <h2 style="margin:0; color:#FAFAFA;">📈 Stock Market Analysis</h2>
        <p style="color:#9CA3AF; margin-top:8px; margin-bottom:0;">
            Real-time stock data powered by Alpha Vantage API • Technical Indicators • AI Forecasting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Sidebar Controls ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📈 Stock Settings")
        
        api_key = st.text_input(
            "Alpha Vantage API Key",
            value=ALPHA_VANTAGE_API_KEY,
            type="password",
            help="Get free key at alphavantage.co",
        )
        
        symbol = st.selectbox(
            "Select Stock Symbol",
            POPULAR_STOCKS,
            index=0,
        )
        
        custom_symbol = st.text_input("Or enter custom symbol:", placeholder="e.g. TSLA")
        if custom_symbol:
            symbol = custom_symbol.upper()
        
        st.markdown("---")
        
        st.markdown("### 🔮 Forecast Settings")
        forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
        
        selected_models = st.multiselect(
            "Select Models",
            list(MODELS.keys()),
            default=["ARIMA", "Exponential Smoothing"],
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Technical Indicators")
        show_sma = st.checkbox("Moving Averages", value=True)
        show_bollinger = st.checkbox("Bollinger Bands", value=False)
        show_rsi = st.checkbox("RSI", value=False)
        show_macd = st.checkbox("MACD", value=False)
    
    # ── Fetch Data ────────────────────────────────────────────────────────
    try:
        with st.spinner(f"Fetching {symbol} data..."):
            df = fetch_stock_daily(symbol, api_key)
        
        if df is None or df.empty:
            st.error("No data returned. Please check the symbol and API key.")
            return
        
        df = clean_timeseries(df)
        tech_df = compute_technical_indicators(df)
        
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            st.warning("⚠️ **Alpha Vantage Free Tier Limit Reached:** Falling back to highly realistic simulated market data so you can continue exploring the models.")
        else:
            st.warning(f"⚠️ **Unable to fetch live data:** {error_msg}. Falling back to simulated market data.")
        
        # Generate demo data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=500, freq="B")
        N = len(dates)
        base = 150
        trend = np.linspace(0, 50, N)
        noise = np.random.normal(0, 3, N)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(N) / 252)
        close = base + trend + noise + seasonal
        
        df = pd.DataFrame({
            "Open": close - np.random.uniform(0, 2, N),
            "High": close + np.random.uniform(0, 3, N),
            "Low": close - np.random.uniform(0, 3, N),
            "Close": close,
            "Volume": np.random.randint(1000000, 50000000, N),
        }, index=dates)
        
        tech_df = compute_technical_indicators(df)
    
    # ── Key Metrics ───────────────────────────────────────────────────────
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    change = latest["Close"] - prev["Close"]
    change_pct = (change / prev["Close"]) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(metric_card_html(
            "Current Price",
            f"${latest['Close']:.2f}",
            f"{change_pct:+.2f}%",
            "green" if change >= 0 else "red",
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(metric_card_html(
            "Day High",
            f"${latest['High']:.2f}",
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(metric_card_html(
            "Day Low",
            f"${latest['Low']:.2f}",
        ), unsafe_allow_html=True)
    
    with col4:
        vol_str = f"{latest['Volume']:,.0f}" if latest['Volume'] > 0 else "N/A"
        st.markdown(metric_card_html(
            "Volume",
            vol_str,
        ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Price Chart", "🔮 Forecast", "📉 Technical Analysis",
        "🔬 Decomposition", "📋 Data Table",
    ])
    
    # ── Tab 1: Price Chart ────────────────────────────────────────────────
    with tab1:
        chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
        
        period = st.selectbox("Time Period", ["1M", "3M", "6M", "1Y", "2Y", "All"], index=3)
        period_map = {"1M": 22, "3M": 66, "6M": 132, "1Y": 252, "2Y": 504, "All": len(df)}
        display_df = df.tail(period_map.get(period, len(df)))
        
        if chart_type == "Candlestick":
            fig = plot_candlestick(display_df, title=f"{symbol} — Candlestick Chart")
        else:
            cols = ["Close"]
            if show_sma and "SMA_20" in tech_df.columns:
                cols.extend(["SMA_20", "SMA_50"])
            fig = plot_timeseries(
                tech_df.tail(period_map.get(period, len(df))),
                columns=cols,
                title=f"{symbol} — Price History",
                fill=True,
            )
            
            if show_bollinger and "BB_Upper" in tech_df.columns:
                display_tech = tech_df.tail(period_map.get(period, len(df)))
                fig.add_scatter(x=display_tech.index, y=display_tech["BB_Upper"],
                              name="BB Upper", line=dict(color="#38F9D7", width=1, dash="dot"))
                fig.add_scatter(x=display_tech.index, y=display_tech["BB_Lower"],
                              name="BB Lower", line=dict(color="#38F9D7", width=1, dash="dot"))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        if "Volume" in df.columns:
            vol_fig = plot_timeseries(
                display_df, columns=["Volume"],
                title="Trading Volume",
                height=250, fill=True,
                colors=["#38F9D7"],
            )
            st.plotly_chart(vol_fig, use_container_width=True)
    
    # ── Tab 2: Forecast ───────────────────────────────────────────────────
    with tab2:
        target_col = "Close"
        series = df[target_col]
        
        if not selected_models:
            st.info("Select at least one model from the sidebar to generate forecasts.")
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
                    st.warning(f"⚠️ {model_name} failed: {str(e)}")
        
        if all_results:
            # Plot forecasts
            for model_name, result in all_results.items():
                st.markdown(f"#### {model_name} Forecast")
                fig = plot_forecast(
                    series.tail(90),
                    result["forecast"],
                    result.get("lower_bound"),
                    result.get("upper_bound"),
                    title=f"{symbol} — {model_name} Forecast ({forecast_days} days)",
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison with back-testing
            if len(all_results) > 1:
                st.markdown("### 🏆 Model Comparison (Back-Test)")
                train, test = prepare_forecast_data(df[[target_col]], target_col, train_ratio=0.85)
                
                metrics = {}
                for model_name, result in all_results.items():
                    fitted = result.get("fitted")
                    if fitted is not None and len(fitted) > 0:
                        common_idx = fitted.index.intersection(series.index)
                        if len(common_idx) > 10:
                            actual = series[common_idx]
                            pred = fitted[common_idx]
                            metrics[model_name] = compute_all_metrics(actual.values, pred.values)
                
                if metrics:
                    metrics_df = pd.DataFrame(metrics).T
                    metrics_df.index.name = "Model"
                    st.dataframe(metrics_df.style.highlight_min(axis=0, subset=["MAE", "RMSE", "MAPE (%)", "sMAPE (%)"]).highlight_max(axis=0, subset=["R²"]), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 🤖 AI Financial Analyst Report")
            report_key = f"ai_report_stock_{symbol}"
            
            if st.button("Generate Hedge Fund Report (Powered by ChatGPT)", type="primary", key="stock_report_btn"):
                from utils.llm_report import generate_financial_report
                with st.spinner("Analyzing market trajectory with OpenAI..."):
                    st.session_state[report_key] = generate_financial_report(f"{symbol} Stock", df, all_results)
                    st.session_state[f"{report_key}_time"] = datetime.now().strftime('%H:%M:%S')
                    
            if report_key in st.session_state:
                st.markdown(f"> **AI Report Generated at {st.session_state[f'{report_key}_time']}**")
                st.markdown(st.session_state[report_key])
                st.download_button(
                    label="📥 Download Full Report (.md)",
                    data=st.session_state[report_key],
                    file_name=f"{symbol}_AI_Financial_Report.md",
                    mime="text/markdown",
                )
    
    # ── Tab 3: Technical Analysis ─────────────────────────────────────────
    with tab3:
        st.markdown("#### RSI (Relative Strength Index)")
        if "RSI" in tech_df.columns:
            rsi_fig = plot_timeseries(
                tech_df.tail(252), columns=["RSI"],
                title="RSI (14-period)", height=300,
                colors=["#F9D423"],
            )
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="#FF6584", annotation_text="Overbought (70)")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="#43E97B", annotation_text="Oversold (30)")
            st.plotly_chart(rsi_fig, use_container_width=True)
        
        st.markdown("#### MACD")
        if "MACD" in tech_df.columns:
            macd_fig = plot_timeseries(
                tech_df.tail(252), columns=["MACD", "MACD_Signal"],
                title="MACD", height=300,
                colors=["#6C63FF", "#FF6584"],
            )
            st.plotly_chart(macd_fig, use_container_width=True)
        
        st.markdown("#### Returns Distribution")
        if "Returns" in tech_df.columns:
            dist_fig = plot_distribution(tech_df["Returns"].dropna(), title="Daily Returns Distribution")
            st.plotly_chart(dist_fig, use_container_width=True)
        
        # Stationarity test
        st.markdown("#### Stationarity Test (ADF)")
        adf = compute_stationarity_test(series)
        if "error" not in adf:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test Statistic", f"{adf['test_statistic']:.4f}")
            with col2:
                st.metric("P-Value", f"{adf['p_value']:.6f}")
            with col3:
                st.metric("Stationary?", "✅ Yes" if adf["is_stationary"] else "❌ No")
    
    # ── Tab 4: Decomposition ──────────────────────────────────────────────
    with tab4:
        try:
            decomp = decompose_timeseries(series)
            decomp_fig = plot_decomposition(decomp, title=f"{symbol} — Time Series Decomposition")
            st.plotly_chart(decomp_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Decomposition failed: {e}")
        
        # Anomaly Detection
        st.markdown("#### 🔍 Anomaly Detection")
        method = st.selectbox("Method", ["zscore", "iqr"])
        anomalies = detect_anomalies(series, method=method)
        anomaly_fig = plot_anomalies(series, anomalies, title=f"{symbol} — Anomalies ({anomalies.sum()} detected)")
        st.plotly_chart(anomaly_fig, use_container_width=True)
    
    # ── Tab 5: Data Table ─────────────────────────────────────────────────
    with tab5:
        st.markdown("#### Raw Data")
        st.dataframe(df.tail(100).sort_index(ascending=False), use_container_width=True)
        
        csv = df.to_csv()
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name=f"{symbol}_stock_data.csv",
            mime="text/csv",
        )
