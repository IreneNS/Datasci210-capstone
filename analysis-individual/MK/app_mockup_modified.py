import streamlit as st
import random
import pandas as pd
import plotly.express as px

# Define color palette
berkeley_blue = "#002676"
california_gold = "#FDB515"
white = "#FFFFFF"

st.set_page_config(page_title="AI Portfolio Optimizer", page_icon="📈", layout="wide")

# Apply custom CSS for header and subheader
st.markdown(f"""
    <div class='header-container' style='background-color:{berkeley_blue}; padding:20px; border-radius:10px; text-align:center; margin-bottom:30px;'>
        <h1 style='color:{california_gold};'>📊 AI-Powered Portfolio Optimizer</h1>
        <p style='color:{white}; font-size:18px;'>Enter your stock universe, specify risk tolerance, and get an optimized portfolio to minimize drawdowns during regime changes.</p>
    </div>
""", unsafe_allow_html=True)

# Input columns: Investment Universe, Risk Tolerance, Optimization Option
cols = st.columns(3)

# Column 1: Investment Universe
with cols[0]:
    ticker_options = ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "NFLX", "FB"]
    st.subheader("Investment Universe")
    selected_tickers = st.multiselect("Choose stocks from the universe", ticker_options)

# Column 2: Risk Tolerance Slider
with cols[1]:
    st.subheader("Risk Tolerance (Annual Volatility)")
    risk_tolerance = st.slider("Select your risk level (volatility)", 5, 30, 15)

# Column 3: Optimization Option Dropdown
with cols[2]:
    st.subheader("Optimization Option")
    optimization_options = ["Maximize Sharpe Ratio", "Minimize Risk"]
    selected_optimization = st.selectbox("Choose an optimization objective", optimization_options)

# Initialize session state for tickers and weights
if "tickers" not in st.session_state:
    st.session_state["tickers"] = []
if "weights" not in st.session_state:
    st.session_state["weights"] = {}

# Process selected tickers
if selected_tickers:
    st.session_state["tickers"] = selected_tickers
    st.session_state["weights"] = {t: st.session_state["weights"].get(t, 0) for t in selected_tickers}

# Keep the header and inputs visible initially
st.markdown("---")  # Optional divider for clarity

# Add placeholders for charts that are rendered after inputs
backtest_placeholder = st.empty()
cumulative_placeholder = st.empty()
weights_placeholder = st.empty()

# Specify Initial Weights and Sentiment and Pie Chart after cumulative chart
if len(st.session_state["tickers"]) > 0:  # Only display when tickers are selected
    # Specify Initial Weights and Sentiment
    st.subheader("Specify Initial Weights (%) and View Market Sentiment")
    columns = st.columns(len(st.session_state["tickers"]))

    for idx, ticker in enumerate(st.session_state["tickers"]):
        with columns[idx % len(columns)]:
            # Use number_input without excessive re-renders
            current_value = st.session_state["weights"].get(ticker, 0)
            new_value = st.number_input(
                f"{ticker} Weight (%)", min_value=0, max_value=100, 
                value=current_value, step=1,
                key=f"weight_{ticker}"
            )
            # Only update state if there is an actual change in the value
            if new_value != current_value:
                st.session_state["weights"][ticker] = new_value

            # Placeholder for sentiment score (randomized as a mock)
            st.session_state["sentiment"] = {ticker: random.uniform(-1, 1) for ticker in st.session_state["tickers"]}
            st.metric(label=f"{ticker} Sentiment Score", value=round(st.session_state["sentiment"][ticker], 2))

    total_weight = sum(st.session_state["weights"].values())

    if total_weight == 100:
        st.success(f"✅ Total Weight: {total_weight}% (Balanced)")
    else:
        st.error(f"⚠️ Total Weight: {total_weight}% (Please balance to 100%)")

    # Show Weights Pie Chart
    st.subheader("Weights Distribution")
    weights_data = pd.DataFrame({
        "Ticker": st.session_state["tickers"],
        "Weight": [st.session_state["weights"][t] for t in st.session_state["tickers"]]
    })
    pie_chart_fig = px.pie(weights_data, names="Ticker", values="Weight", title="Portfolio Weights")
    weights_placeholder.plotly_chart(pie_chart_fig)

else:
    st.write("Please select tickers to assign weights and view distribution.")

# Now render the charts in the placeholders

# Add Backtest Chart (Placeholder)
backtest_data = {
    "Date": pd.date_range(start="2021-01-01", periods=100, freq="D"),
    "Backtest Performance": [random.uniform(80, 120) for _ in range(100)]
}
backtest_df = pd.DataFrame(backtest_data)
backtest_fig = px.line(backtest_df, x="Date", y="Backtest Performance", title="Backtest Performance Over Time")
backtest_placeholder.plotly_chart(backtest_fig)

# Add Cumulative Return Chart
data = {
    "Date": pd.date_range(start="2022-01-01", periods=100, freq="D"),
    "Optimized Portfolio": [random.uniform(90, 110) for _ in range(100)],
    "Benchmark Portfolio": [random.uniform(85, 105) for _ in range(100)]
}
df = pd.DataFrame(data)
fig = px.line(df, x="Date", y=["Optimized Portfolio", "Benchmark Portfolio"], title="Cumulative Returns")
cumulative_placeholder.plotly_chart(fig)

# Add footer manually with Streamlit's markdown component
st.markdown(
    f"<div style='background-color:{berkeley_blue}; padding:10px; color:{california_gold}; text-align:right;'>MIDS 2025</div>",
    unsafe_allow_html=True
)
