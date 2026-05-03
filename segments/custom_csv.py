"""
📂 Custom CSV Segment
Allow users to upload their own time series data.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from utils.data_processing import compute_returns, decompose_timeseries, detect_anomalies, compute_stationarity_test
from utils.visualization import plot_timeseries, plot_forecast, plot_decomposition, plot_anomalies, metric_card_html

def render():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1A1F2E 0%, #2E1A2D 100%); border-radius: 16px; padding: 30px; margin-bottom: 24px; border: 1px solid rgba(255,101,132,0.3);">
        <h2 style="margin:0; color:#FAFAFA;">📂 Custom Data Analysis</h2>
        <p style="color:#9CA3AF; margin-top:8px; margin-bottom:0;">Upload your own CSV • Any Domain • Advanced Forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 📂 Upload Data")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        st.markdown("---")
        st.markdown("### 🔮 Forecast Settings")
        forecast_days = st.slider("Forecast Horizon", 7, 365, 30, key="custom_forecast")
        
        AVAILABLE_MODELS = ["ARIMA", "Facebook Prophet", "LSTM", "Exponential Smoothing"]
        selected_models = st.multiselect("Select Models to Run", AVAILABLE_MODELS, default=["Facebook Prophet"], key="custom_models")

    # Clear session state if file is removed or changed
    if uploaded_file is None:
        if 'custom_df' in st.session_state:
            del st.session_state['custom_df']
        st.info("👆 Please upload a CSV file from the sidebar to begin analysis. The file must contain a date/time column and a numerical target column.")
        return

    # Track file changes to reset processing
    if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != uploaded_file.name:
        st.session_state['last_uploaded_file'] = uploaded_file.name
        if 'custom_df' in st.session_state:
            del st.session_state['custom_df']

    try:
        # Read just the header/columns first for selection
        df_raw = pd.read_csv(uploaded_file)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            date_col = st.selectbox("Select Date Column", df_raw.columns)
        with col2:
            numeric_cols = [c for c in df_raw.columns if c != date_col and pd.api.types.is_numeric_dtype(df_raw[c])]
            if not numeric_cols:
                st.error("No numeric columns found in the CSV for forecasting.")
                return
            value_col = st.selectbox("Select Target Value Column", numeric_cols)
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            process_btn = st.button("Process Data", use_container_width=True, type="primary")
            
        if process_btn:
            with st.spinner("Processing time series data..."):
                df = df_raw.copy()
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col]) # drop invalid dates
                df = df.set_index(date_col).sort_index()
                df = df[[value_col]].dropna()
                
                if len(df) < 30:
                    st.error("Dataset is too small. Please provide at least 30 valid data points.")
                    return
                    
                st.session_state['custom_df'] = df
                st.session_state['custom_val'] = value_col
                
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # If data is processed, show the dashboard
    if 'custom_df' in st.session_state:
        df = st.session_state['custom_df']
        value_col = st.session_state['custom_val']
        series = df[value_col]
        
        st.markdown("---")
        
        # -----------------------------------------------------------------------
        # Top KPI Metrics
        # -----------------------------------------------------------------------
        current_val = series.iloc[-1]
        prev_val = series.iloc[-2] if len(series) > 1 else current_val
        pct_change = ((current_val - prev_val) / prev_val) * 100 if prev_val != 0 else 0
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(metric_card_html("LATEST VALUE", f"{current_val:,.2f}", f"{pct_change:.2f}%"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card_html("MAXIMUM", f"{series.max():,.2f}", ""), unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card_html("MINIMUM", f"{series.min():,.2f}", ""), unsafe_allow_html=True)
        with m4:
            st.markdown(metric_card_html("TOTAL RECORDS", f"{len(series):,}", ""), unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)

        # -----------------------------------------------------------------------
        # Main Tabs
        # -----------------------------------------------------------------------
        t1, t2, t3 = st.tabs(["📊 Overview", "🔮 Forecast", "🔬 Analysis"])
        
        with t1:
            st.plotly_chart(plot_timeseries(df, columns=[value_col], title=f"{value_col} History", fill=True), use_container_width=True)
            
        with t2:
            if not selected_models:
                st.info("Select at least one model from the sidebar to view forecasts.")
            else:
                for m in selected_models:
                    with st.spinner(f"Training {m} model..."):
                        try:
                            if m == "ARIMA":
                                from models.arima_model import fit_arima
                                res = fit_arima(series, forecast_steps=forecast_days, auto_order=False)
                            elif m == "Facebook Prophet":
                                from models.prophet_model import fit_prophet
                                res = fit_prophet(series, forecast_steps=forecast_days)
                            elif m == "LSTM":
                                from models.lstm_model import fit_lstm
                                res = fit_lstm(series, forecast_steps=forecast_days, epochs=50)
                            elif m == "Exponential Smoothing":
                                from models.exp_smoothing import fit_exponential_smoothing
                                res = fit_exponential_smoothing(series, forecast_steps=forecast_days)
                            
                            st.plotly_chart(plot_forecast(series.tail(min(300, len(series))), res["forecast"], res.get("lower_bound"), res.get("upper_bound"), title=f"{m} Forecast ({forecast_days}d)"), use_container_width=True)
                        except Exception as e:
                            st.warning(f"⚠️ **{m} Model Failed:** {str(e)}. This dataset may not be compatible with this model's mathematical requirements.")
                            
        with t3:
            colA, colB = st.columns(2)
            with colA:
                st.markdown("#### Stationarity Test (ADF)")
                adf_res = compute_stationarity_test(series)
                if "error" in adf_res:
                    st.warning(adf_res["error"])
                else:
                    st.write(f"**ADF Statistic:** {adf_res['test_statistic']}")
                    st.write(f"**p-value:** {adf_res['p_value']}")
                    st.write(f"**Is Stationary:** {'Yes ✅' if adf_res['is_stationary'] else 'No ❌'}")
                    
            with colB:
                st.markdown("#### Anomaly Detection")
                anomalies = detect_anomalies(series)
                anomaly_count = anomalies.sum()
                st.write(f"**Anomalies Detected:** {anomaly_count}")
                if anomaly_count > 0:
                    fig_anom = plot_anomalies(series, anomalies, "Detected Anomalies")
                    st.plotly_chart(fig_anom, use_container_width=True)
                    
            st.markdown("#### Seasonal Decomposition")
            try:
                st.plotly_chart(plot_decomposition(decompose_timeseries(series)), use_container_width=True)
            except Exception as e:
                st.info(f"Could not perform seasonal decomposition: {e}")
