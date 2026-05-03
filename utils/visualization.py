"""
Plotly-based visualization builders for the Time Series Platform.
All charts use a consistent dark theme with the app's color palette.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ── Global Layout Template ──────────────────────────────────────────────────

DARK_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(family="Inter, sans-serif", color="#FAFAFA", size=13),
        xaxis=dict(gridcolor="#2D3446", zerolinecolor="#2D3446"),
        yaxis=dict(gridcolor="#2D3446", zerolinecolor="#2D3446"),
        legend=dict(bgcolor="rgba(26,31,46,0.8)", bordercolor="#2D3446", borderwidth=1),
        margin=dict(l=60, r=30, t=60, b=50),
    )
)


def _base_layout(title: str = "", height: int = 500) -> dict:
    """Return a base layout dictionary."""
    return dict(
        title=dict(text=title, font=dict(size=18, color="#FAFAFA"), x=0.02),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(family="Inter, sans-serif", color="#FAFAFA", size=13),
        xaxis=dict(
            gridcolor="#2D3446",
            zerolinecolor="#2D3446",
            showgrid=True,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(gridcolor="#2D3446", zerolinecolor="#2D3446", showgrid=True),
        legend=dict(
            bgcolor="rgba(26,31,46,0.8)",
            bordercolor="#2D3446",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=60, b=50),
        height=height,
        hovermode="x unified",
    )


# ── Price / Line Charts ─────────────────────────────────────────────────────

def plot_timeseries(
    df: pd.DataFrame,
    columns: list = None,
    title: str = "Time Series",
    height: int = 500,
    colors: list = None,
    fill: bool = False,
) -> go.Figure:
    """Plot one or more time series columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
    
    if colors is None:
        colors = ["#6C63FF", "#FF6584", "#43E97B", "#F9D423", "#38F9D7"]
    
    fig = go.Figure()
    
    for i, col in enumerate(columns):
        if col in df.columns:
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                name=col,
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy" if fill and i == 0 else None,
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)" if fill and i == 0 else None,
            ))
    
    fig.update_layout(**_base_layout(title, height))
    return fig


def plot_candlestick(
    df: pd.DataFrame,
    title: str = "Candlestick Chart",
    height: int = 600,
    show_volume: bool = True,
) -> go.Figure:
    """Create a candlestick chart with optional volume bars."""
    if show_volume and "Volume" in df.columns:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )
    else:
        fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing=dict(line=dict(color="#43E97B"), fillcolor="rgba(67,233,123,0.3)"),
        decreasing=dict(line=dict(color="#FF6584"), fillcolor="rgba(255,101,132,0.3)"),
        name="OHLC",
    ), row=1, col=1)
    
    if show_volume and "Volume" in df.columns:
        colors = ["#43E97B" if c >= o else "#FF6584" for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=colors,
            opacity=0.5,
            name="Volume",
            showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="#2D3446")
    
    layout = _base_layout(title, height)
    layout["xaxis"]["rangeslider"] = dict(visible=False)
    fig.update_layout(**layout)
    
    return fig


# ── Forecast Chart ───────────────────────────────────────────────────────────

def plot_forecast(
    historical: pd.Series,
    forecast: pd.Series,
    lower_bound: pd.Series = None,
    upper_bound: pd.Series = None,
    title: str = "Forecast",
    height: int = 500,
    actual: pd.Series = None,
) -> go.Figure:
    """Plot historical data with forecast and confidence intervals."""
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical.values,
        name="Historical",
        mode="lines",
        line=dict(color="#6C63FF", width=2),
    ))
    
    # Confidence interval
    if lower_bound is not None and upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=list(upper_bound.index) + list(lower_bound.index[::-1]),
            y=list(upper_bound.values) + list(lower_bound.values[::-1]),
            fill="toself",
            fillcolor="rgba(108,99,255,0.15)",
            line=dict(color="rgba(108,99,255,0)"),
            name="Confidence Interval",
            showlegend=True,
        ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        name="Forecast",
        mode="lines",
        line=dict(color="#F9D423", width=2.5, dash="dash"),
    ))
    
    # Actual (if provided for back-testing)
    if actual is not None:
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual.values,
            name="Actual",
            mode="lines",
            line=dict(color="#FF6584", width=2),
        ))
    
    fig.update_layout(**_base_layout(title, height))
    return fig


# ── Decomposition Chart ─────────────────────────────────────────────────────

def plot_decomposition(decomp: dict, title: str = "Time Series Decomposition", height: int = 800) -> go.Figure:
    """Plot decomposed components: observed, trend, seasonal, residual."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
        vertical_spacing=0.06,
    )
    
    colors = ["#6C63FF", "#43E97B", "#F9D423", "#FF6584"]
    components = ["observed", "trend", "seasonal", "residual"]
    
    for i, (comp, color) in enumerate(zip(components, colors)):
        data = decomp[comp]
        if data is not None:
            fig.add_trace(go.Scatter(
                x=data.index, y=data.values,
                name=comp.capitalize(),
                line=dict(color=color, width=1.5),
                showlegend=False,
            ), row=i+1, col=1)
    
    layout = _base_layout(title, height)
    fig.update_layout(**layout)
    
    for i in range(1, 5):
        fig.update_yaxes(gridcolor="#2D3446", row=i, col=1)
    
    return fig


# ── Model Comparison Chart ──────────────────────────────────────────────────

def plot_model_comparison(
    metrics: dict,
    title: str = "Model Comparison",
    height: int = 400,
) -> go.Figure:
    """
    Bar chart comparing model metrics.
    metrics: {model_name: {metric_name: value, ...}, ...}
    """
    models = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())
    
    fig = make_subplots(
        rows=1, cols=len(metric_names),
        subplot_titles=metric_names,
    )
    
    colors = ["#6C63FF", "#FF6584", "#43E97B", "#F9D423", "#38F9D7"]
    
    for j, metric in enumerate(metric_names):
        values = [metrics[m].get(metric, 0) for m in models]
        fig.add_trace(go.Bar(
            x=models,
            y=values,
            name=metric,
            marker_color=[colors[i % len(colors)] for i in range(len(models))],
            showlegend=False,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            textfont=dict(color="#FAFAFA", size=11),
        ), row=1, col=j+1)
    
    layout = _base_layout(title, height)
    fig.update_layout(**layout)
    
    return fig


# ── Correlation Heatmap ─────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix", height: int = 500) -> go.Figure:
    """Create an interactive correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[
            [0, "#FF6584"],
            [0.5, "#0E1117"],
            [1, "#6C63FF"],
        ],
        zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=11, color="#FAFAFA"),
    ))
    
    fig.update_layout(**_base_layout(title, height))
    return fig


# ── Distribution Chart ──────────────────────────────────────────────────────

def plot_distribution(series: pd.Series, title: str = "Distribution", height: int = 400) -> go.Figure:
    """Plot histogram with KDE overlay."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=series.dropna(),
        nbinsx=50,
        marker_color="#6C63FF",
        opacity=0.7,
        name="Distribution",
    ))
    
    fig.update_layout(**_base_layout(title, height))
    fig.update_layout(bargap=0.05)
    
    return fig


# ── Anomaly Chart ────────────────────────────────────────────────────────────

def plot_anomalies(series: pd.Series, anomalies: pd.Series, title: str = "Anomaly Detection", height: int = 500) -> go.Figure:
    """Plot time series with anomalies highlighted."""
    fig = go.Figure()
    
    # Normal data
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        name="Data",
        mode="lines",
        line=dict(color="#6C63FF", width=1.5),
    ))
    
    # Anomaly points
    anomaly_points = series[anomalies]
    if len(anomaly_points) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_points.index,
            y=anomaly_points.values,
            name="Anomalies",
            mode="markers",
            marker=dict(color="#FF6584", size=10, symbol="x", line=dict(width=2, color="#FF6584")),
        ))
    
    fig.update_layout(**_base_layout(title, height))
    return fig


# ── Metric Cards (returns HTML) ─────────────────────────────────────────────

def metric_card_html(label: str, value: str, delta: str = None, delta_color: str = "green") -> str:
    """Generate HTML for a styled metric card."""
    delta_html = ""
    if delta:
        arrow = "↑" if delta_color == "green" else "↓"
        delta_html = f'<div style="color: {"#43E97B" if delta_color == "green" else "#FF6584"}; font-size: 14px; margin-top: 4px;">{arrow} {delta}</div>'
    
    return f"""
    <div style="
        background: linear-gradient(135deg, #1A1F2E 0%, #252B3B 100%);
        border: 1px solid #2D3446;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    ">
        <div style="color: #9CA3AF; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">{label}</div>
        <div style="color: #FAFAFA; font-size: 28px; font-weight: 700; margin-top: 8px;">{value}</div>
        {delta_html}
    </div>
    """
