def render():
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import datetime

    st.markdown("## 📈 Stock Market Analysis")

    # --- Sidebar Controls ---
    with st.sidebar:
        st.markdown("### 📈 Stock Settings")

        api_key = st.text_input(
            "Alpha Vantage API Key",
            type="password",
        )

        symbol = st.text_input("Stock Symbol", value="AAPL")

        forecast_days = st.slider("Forecast Horizon", 7, 90, 30)

        selected_models = st.multiselect(
            "Select Models",
            ["ARIMA", "Prophet", "Exponential Smoothing"],
            default=["ARIMA"]
        )

    # --- IMPORTANT: CONTROL EXECUTION ---
    if not st.button("🚀 Run Analysis"):
        st.info("👉 Click 'Run Analysis' to start")
        return

    # --- FETCH DATA ---
    try:
        with st.spinner("Fetching data..."):
            df = fetch_stock_daily(symbol, api_key)

        if df is None or df.empty:
            st.error("No data found")
            return

        df = clean_timeseries(df)
        tech_df = compute_technical_indicators(df)

    except Exception as e:
        st.warning(f"Using demo data due to error: {e}")

        # fallback demo
        dates = pd.date_range(end=datetime.now(), periods=300)
        df = pd.DataFrame({
            "Close": np.random.randn(300).cumsum() + 100
        }, index=dates)

        tech_df = df.copy()

    st.success("✅ Data Loaded")

    # --- SIMPLE VISUAL ---
    st.line_chart(df["Close"])
