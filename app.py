"""
🚀 TimeSeries Pro - Main Streamlit Application
Multi-Segment Time Series Forecasting Platform
"""

import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="TimeSeries Pro",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DEBUG (IMPORTANT) ---
st.write("✅ App Loaded Successfully")

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    header {
        background-color: transparent !important;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Import settings ---
try:
    from config.settings import APP_TITLE, APP_SUBTITLE, SEGMENTS
except ImportError:
    st.error("❌ Missing configuration. Please check project structure.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.markdown(f"*{APP_SUBTITLE}*")
    st.markdown("---")
    
    st.markdown("### 🧭 Navigation")
    selected_segment = st.radio(
        "Choose Segment",
        list(SEGMENTS.keys()),
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #9CA3AF;">
    <b>Models Available:</b><br>
    • ARIMA/SARIMA<br>
    • Facebook Prophet<br>
    • LSTM Neural Network<br>
    • Exponential Smoothing
    </div>
    """, unsafe_allow_html=True)

# --- Main UI ---
st.title("⏱️ TimeSeries Pro")
st.write(f"Selected Segment: **{selected_segment}**")

# --- Routing ---
segment_module_name = SEGMENTS[selected_segment]

module = None

try:
    if segment_module_name == "stock_market":
        import segments.stock_market as module
    elif segment_module_name == "cryptocurrency":
        import segments.cryptocurrency as module
    elif segment_module_name == "weather":
        import segments.weather as module
    elif segment_module_name == "forex":
        import segments.forex as module
    elif segment_module_name == "custom_csv":
        import segments.custom_csv as module
    else:
        st.error(f"Segment '{segment_module_name}' not implemented yet.")

except ImportError as e:
    st.error(f"❌ Failed to load segment: {e}")

# --- CONTROLLED EXECUTION (KEY FIX) ---
if module and hasattr(module, 'render'):
    
    st.markdown("### ▶️ Run Analysis")

    if st.button("🚀 Run Selected Segment"):
        try:
            with st.spinner("Processing... Please wait ⏳"):
                module.render()
                
        except Exception as e:
            st.error(f"❌ Error occurred: {e}")

else:
    st.warning("⚠️ Module not ready or missing render() function")
