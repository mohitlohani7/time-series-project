"""
AI Financial Analysis Report Generator using OpenAI ChatGPT API.
"""
import streamlit as st
import pandas as pd
from openai import OpenAI
from config.settings import OPENAI_API_KEY

def generate_financial_report(
    asset_name: str,
    historical_df: pd.DataFrame,
    forecast_results: dict,
    api_key: str = None
) -> str:
    key = api_key or OPENAI_API_KEY
    if not key or key.strip() == "":
        return "⚠️ **OpenAI API Key is missing!** Please add `OPENAI_API_KEY = 'your_key'` to your `.streamlit/secrets.toml` or provide it in the UI to generate AI reports."
        
    client = OpenAI(api_key=key)
    
    # Prepare data summary
    hist_tail = historical_df.tail(14)
    # Handle both Series and DataFrame
    if isinstance(hist_tail, pd.DataFrame):
        target_col = "Close" if "Close" in hist_tail.columns else "Price" if "Price" in hist_tail.columns else "Value"
        recent_trend = "Upward" if hist_tail[target_col].iloc[-1] > hist_tail[target_col].iloc[0] else "Downward"
        latest_price = hist_tail[target_col].iloc[-1]
    else:
        recent_trend = "Upward" if hist_tail.iloc[-1] > hist_tail.iloc[0] else "Downward"
        latest_price = hist_tail.iloc[-1]
    
    # Grab the first available model's forecast for the summary if there are multiple
    model_name, result = list(forecast_results.items())[0]
    forecast_series = result['forecast']
    
    pred_7_days = forecast_series.iloc[:7]
    pred_30_days = forecast_series.iloc[:30] if len(forecast_series) >= 30 else forecast_series
    
    prompt = f"""
    You are an elite hedge fund quantitative analyst. 
    Write a detailed, professional financial analysis report for {asset_name}.
    
    Here is the recent historical context:
    - Recent 14-day trend: {recent_trend}
    - Latest Price: {latest_price:.2f}
    
    Here is our Mathematical AI model ({model_name}) prediction context:
    - 7-Day Strict Prediction: Starts at {pred_7_days.iloc[0]:.2f}, ends at {pred_7_days.iloc[-1]:.2f}. Average: {pred_7_days.mean():.2f}
    - 30-Day Outlook: Ends at {pred_30_days.iloc[-1]:.2f}.
    
    Write a comprehensive report formatted in elegant Markdown including:
    1. **Executive Summary**
    2. **When to Buy and Why** (Specifically analyze the optimal entry point based on the 7-day and 30-day outlook)
    3. **7-Day Strict Prediction Analysis**
    4. **30-Day Outlook & Trajectory**
    5. **Key Factors & Risks Affecting this Prediction** (Discuss macroeconomic factors, volume patterns, and typical volatility for this asset)
    6. **Final Hedge Fund Verdict** (Strong Buy, Buy, Hold, Sell, Strong Sell)
    
    Make it look extremely professional, data-driven, and structured exactly like a real Wall Street analyst report. Be decisive and insightful.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a highly sophisticated Wall Street quantitative analyst. Output only the report in markdown format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ **Error generating AI report:** {str(e)}"
