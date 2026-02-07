import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- PALETTE DEFINITION ---
# Matching the "Midnight Teal" aesthetic
MAIN_TEAL = "#00d4aa"
DARK_BG = "#0e1117"
ACCENT_RED = "#ff4b4b"
CHART_COLORS = ["#00d4aa", "#008a73", "#004d40", "#7ef4da", "#b2fcf0"]

st.set_page_config(
    page_title="Quant Portfolio Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# --- FORCED DARK CSS ---
st.markdown(f"""
<style>
    /* Main App Background */
    .stApp {{
        background-color: #0e1117;
        color: #ffffff;
    }}
    /* Sidebar Background */
    section[data-testid="stSidebar"] {{
        background-color: #161b22 !important;
    }}
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {{
        color: #00d4aa !important;
        font-weight: bold;
    }}
    [data-testid="stMetricLabel"] {{
        color: #8b949e !important;
    }}
    /* Make tables transparent to show background */
    .stTable {{
        background-color: transparent !important;
        color: #ffffff !important;
    }}
    /* Title and Header colors */
    h1, h2, h3 {{
        color: #00d4aa !important;
    }}
</style>
""", unsafe_allow_html=True)


st.title("📊 Portfolio Analytics")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Portfolio Holdings")
default_input = "AAPL, 150, 50\nMSFT, 300, 30\nBTC-USD, 45000, 1.2\nGLD, 180, 100\nHSBA.L, 418, 1331"
user_input = st.sidebar.text_area("Tickers (Ticker, Avg Price, Qty)", default_input, height=200)

def parse_input(input_str):
    holdings = {}
    try:
        for line in input_str.split('\n'):
            if line.strip() and ',' in line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 3:
                    ticker, price, qty = parts
                    holdings[ticker.upper()] = [float(price), float(qty)]
        return holdings
    except Exception as e:
        st.error(f"Error parsing input: {e}")
        return None

if st.sidebar.button("Run Analytics"):
    holdings = parse_input(user_input)

    if holdings:
        tickers = list(holdings.keys())

        with st.spinner('Downloading market data...'):
            raw_data = yf.download(tickers, period="3y")['Close']
            if raw_data.empty:
                st.error("❌ No data found! Check symbols.")
                st.stop()
            if isinstance(raw_data, pd.Series):
                raw_data = raw_data.to_frame()
            df = raw_data.ffill().dropna()

        # --- THE CALCULATION ENGINE ---
        latest_prices = df.iloc[-1].copy()
        valid_tickers = latest_prices.index.tolist()

        # Currency Correction for London Stock Exchange
        for t in valid_tickers:
            if t.endswith('.L'):
                latest_prices[t] = latest_prices[t] / 100

        values = pd.Series({t: latest_prices[t] * holdings[t][1] for t in valid_tickers})
        total_value = values.sum()
        weights = values / total_value

        returns = df[valid_tickers].pct_change().dropna()
        port_returns = returns.dot(weights)

        # 1. Performance math
        cum_returns = (1 + port_returns).cumprod()
        window = 63
        rf_annual = 0.04
        rolling_mu = port_returns.rolling(window).mean() * 252
        rolling_std = port_returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mu - rf_annual) / rolling_std

        # 2. Sortino Calculation
        downside_returns = port_returns.copy()
        downside_returns[downside_returns > 0] = 0
        rolling_downside_std = downside_returns.rolling(window).std() * np.sqrt(252)
        rolling_sortino = (rolling_mu - rf_annual) / rolling_downside_std

        # 3. Drawdown & VaR
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        var_95 = np.percentile(port_returns, 5)

        # --- WEB DASHBOARD LAYOUT ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Portfolio Value", f"£{total_value:,.2f}")
        col2.metric("Max Drawdown", f"{drawdown.min():.2%}")
        col3.metric("95% Daily VaR", f"{var_95:.2%}")

        st.markdown("---")

        # --- ENHANCED PERFORMANCE & DRAWDOWN SECTION ---
        st.subheader("Performance & Risk Profile")
        
        # 1. Growth Chart
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Scatter(
            x=cum_returns.index, y=cum_returns, 
            name="Cumulative Growth", 
            line=dict(color=MAIN_TEAL, width=3)
        ))
        fig_growth.update_layout(
            title="Portfolio Cumulative Growth (Value of £1)",
            template="plotly_dark", 
            hovermode="x unified", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='#232a35'),
            xaxis=dict(gridcolor='#232a35')
        )
        st.plotly_chart(fig_growth, use_container_width=True)

        # 2. Dedicated Drawdown Chart (The "Underwater" Plot)
        fig_dd = go.Figure()
        # Multiply by 100 to get percentage
        drawdown_pct = drawdown * 100
        
        fig_dd.add_trace(go.Scatter(
            x=drawdown_pct.index, y=drawdown_pct, 
            name="Drawdown (%)", 
            fill='tozeroy', 
            line=dict(color=ACCENT_RED, width=1.5),
            hovertemplate="Drawdown: %{y:.2f}%<extra></extra>"
        ))
        
        fig_dd.update_layout(
            title="Portfolio Underwater Analysis (Drawdown %)",
            template="plotly_dark", 
            hovermode="x unified", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(
                title="Decline from Peak (%)",
                gridcolor='#232a35',
                ticksuffix="%"
            ),
            xaxis=dict(gridcolor='#232a35')
        )
        st.plotly_chart(fig_dd, use_container_width=True)
        

        # Secondary Charts Row
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Asset Allocation")
            fig_pie = px.pie(
                values=values, names=values.index, 
                hole=0.5, 
                color_discrete_sequence=CHART_COLORS
            )
            fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with c2:
            st.subheader("Rolling Ann. Sharpe Ratio")
            fig_sharpe = px.line(rolling_sharpe, color_discrete_sequence=[MAIN_TEAL])
            fig_sharpe.update_layout(template="plotly_dark", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sharpe, use_container_width=True)


        # --- NEW: Returns Distribution Chart ---
        st.subheader("Daily Returns Distribution")
        
        # Create Histogram with Plotly
        fig_dist = go.Figure()
        
        # Main histogram
        fig_dist.add_trace(go.Histogram(
            x=port_returns,
            nbinsx=50,
            name="Daily Returns",
            marker_color=MAIN_TEAL,
            opacity=0.75,
            hovertemplate="Return: %{x:.2%}<br>Frequency: %{y}<extra></extra>"
        ))
        
        # Add a vertical line for VaR
        fig_dist.add_vline(
            x=var_95, 
            line_dash="dash", 
            line_color=ACCENT_RED, 
            annotation_text=f"95% VaR ({var_95:.2%})", 
            annotation_position="top left"
        )
        
        fig_dist.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            xaxis_tickformat=".1%",
            showlegend=False
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)

        
        # --- SUMMARY TABLE ---
        st.subheader("📊 Quantitative Risk & Performance Summary")
        
        recovery_days = (drawdown < 0).groupby((drawdown == 0).cumsum()).cumcount().max()
        current_sharpe = rolling_sharpe.iloc[-1]
        current_sortino = rolling_sortino.iloc[-1]
        ann_vol = port_returns.std() * np.sqrt(252)
        mdd = drawdown.min()
        
        total_return = cum_returns.iloc[-1] - 1
        ann_return = (1 + total_return) ** (1 / (len(df) / 252)) - 1
        calmar = ann_return / abs(mdd) if mdd != 0 else 0
        hit_ratio = len(port_returns[port_returns > 0]) / len(port_returns)
        cvar_95 = port_returns[port_returns <= var_95].mean()

        summary = pd.DataFrame({
            "Metric": ["Current Sharpe", "Current Sortino", "Calmar Ratio", "Ann. Volatility", "95% VaR", "Expected Shortfall", "Max Drawdown", "Max Recovery", "Hit Ratio"],
            "Value": [f"{current_sharpe:.2f}", f"{current_sortino:.2f}", f"{calmar:.2f}", f"{ann_vol:.2%}", f"{var_95:.2%}", f"{cvar_95:.2%}", f"{mdd:.2%}", f"{recovery_days} Days", f"{hit_ratio:.2%}"]
        })
        st.table(summary)

        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")




