import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Quant Portfolio Analytics", layout="wide")
st.title("📊 Quant Portfolio Risk & Performance")

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
                st.error("❌ No data found! Check symbols (e.g., AAPL, BTC-USD, HSBA.L).")
                st.stop()
            if isinstance(raw_data, pd.Series):
                raw_data = raw_data.to_frame()
            df = raw_data.ffill().dropna()

        # --- THE CALCULATION ENGINE ---
        latest_prices = df.iloc[-1].copy()
        valid_tickers = latest_prices.index.tolist()

        # Currency Correction
        for t in valid_tickers:
            if t.endswith('.L'):
                latest_prices[t] = latest_prices[t] / 100

        values = pd.Series({t: latest_prices[t] * holdings[t][1] for t in valid_tickers})
        total_value = values.sum()
        weights = values / total_value

        returns = df[valid_tickers].pct_change().dropna()
        port_returns = returns.dot(weights)

        # 1. Performance & Rolling Sharpe
        cum_returns = (1 + port_returns).cumprod()
        window = 63
        rf_annual = 0.04
        rolling_mu = port_returns.rolling(window).mean() * 252
        rolling_std = port_returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mu - rf_annual) / rolling_std

        # 2. FIXED: Rolling Sortino Calculation
        downside_returns = port_returns.copy()
        downside_returns[downside_returns > 0] = 0
        rolling_downside_std = downside_returns.rolling(window).std() * np.sqrt(252)
        rolling_sortino = (rolling_mu - rf_annual) / rolling_downside_std

        # 3. Drawdown & VaR
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        var_95 = np.percentile(port_returns, 5)

        # --- WEB DASHBOARD ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Portfolio Value", f"£{total_value:,.2f}")
        col2.metric("Max Drawdown", f"{drawdown.min():.2%}")
        col3.metric("95% Daily VaR", f"{var_95:.2%}")

        st.subheader("Performance & Risk Profile")
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns, name="Cumulative Growth", line=dict(color="#2ecc71")))
        fig_perf.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown", fill='tozeroy', line=dict(color="#e74c3c")))
        st.plotly_chart(fig_perf, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Asset Allocation")
            st.plotly_chart(px.pie(values=values, names=values.index, hole=0.4), use_container_width=True)
        with c2:
            st.subheader("Rolling Annualized Sharpe Ratio")
            st.plotly_chart(px.line(rolling_sharpe), use_container_width=True)

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
