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

            if len(df) < 10:
                st.error("❌ Not enough historical data to calculate metrics.")
                st.stop()

        # --- THE CALCULATION ENGINE ---
        # 1. Get latest prices and define which tickers are valid
        latest_prices = df.iloc[-1].copy()
        valid_tickers = latest_prices.index.tolist()

        # 2. Currency Correction (LSE Pence to GBP)
        for t in valid_tickers:
            if t.endswith('.L'):
                latest_prices[t] = latest_prices[t] / 100

        # 3. Weights and Portfolio Returns
        values = pd.Series({t: latest_prices[t] * holdings[t][1] for t in valid_tickers})
        total_value = values.sum()
        weights = values / total_value

        # Note: We slice the df to only valid tickers to avoid errors
        returns = df[valid_tickers].pct_change().dropna()
        port_returns = returns.dot(weights)

        # 4. Performance & Risk Math
        cum_returns = (1 + port_returns).cumprod()
        window = 63
        rf_annual = 0.04

        rolling_mu = port_returns.rolling(window).mean() * 252
        rolling_std = port_returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mu - rf_annual) / rolling_std

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
        fig_perf.add_trace(
            go.Scatter(x=cum_returns.index, y=cum_returns, name="Cumulative Growth", line=dict(color="#2ecc71")))
        fig_perf.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown", fill='tozeroy', line=dict(color="#e74c3c")))
        st.plotly_chart(fig_perf, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Asset Allocation")
            st.plotly_chart(px.pie(values=values, names=values.index, hole=0.4), use_container_width=True)
        with c2:
            st.subheader("Rolling Annualized Sharpe Ratio")
            st.plotly_chart(px.line(rolling_sharpe), use_container_width=True)

        # --- ENHANCED ANALYTICS TABLE ---
        st.subheader("📊 Quantitative Risk & Performance Summary")

        # 1. Calculate Additional Metrics
        current_sharpe = rolling_sharpe.iloc[-1]
        current_sortino = rolling_sortino.iloc[-1]
        ann_vol = port_returns.std() * np.sqrt(252)
        mdd = drawdown.min()

        # Calmar Ratio (Ann. Return / Max Drawdown)
        total_return = cum_returns.iloc[-1] - 1
        years = len(df) / 252
        ann_return = (1 + total_return) ** (1 / years) - 1
        calmar = ann_return / abs(mdd) if mdd != 0 else 0

        # Hit Ratio (Percentage of positive days)
        hit_ratio = len(port_returns[port_returns > 0]) / len(port_returns)

        # Expected Shortfall (CVaR) - Average loss beyond VaR
        cvar_95 = port_returns[port_returns <= var_95].mean()

        # 2. Build the DataFrame
        summary = pd.DataFrame({
            "Metric": [
                "Current Sharpe Ratio",
                "Current Sortino Ratio",
                "Calmar Ratio (Efficiency)",
                "Annualized Volatility",
                "95% Daily VaR",
                "95% Expected Shortfall (Tail Risk)",
                "Maximum Drawdown",
                "Max Recovery Period",
                "Hit Ratio (% Positive Days)"
            ],
            "Value": [
                f"{current_sharpe:.2f}",
                f"{current_sortino:.2f}",
                f"{calmar:.2f}",
                f"{ann_vol:.2%}",
                f"{var_95:.2%}",
                f"{cvar_95:.2%}",
                f"{mdd:.2%}",
                f"{recovery_days} Days",
                f"{hit_ratio:.2%}"
            ],
            "Quant Significance": [
                "Risk-adjusted return (Higher is better)",
                "Downside-risk adjusted return",
                "Return vs Max Pain ratio",
                "Price fluctuations (Higher = riskier)",
                "Minimum expected loss on a bad day",
                "Average loss if VaR is exceeded",
                "Worst peak-to-trough drop",
                "Time taken to recover from losses",
                "Frequency of winning days"
            ]
        })

        st.table(summary)