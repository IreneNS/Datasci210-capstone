
import streamlit as st
import random
import pandas as pd
import plotly.express as px

# Define color palette
berkeley_blue = "#002676"
california_gold = "#FDB515"
white = "#FFFFFF"

st.set_page_config(page_title="AI Portfolio Optimizer", page_icon="📈", layout="wide")

# Apply custom CSS for color theme, header, and padding
st.markdown(f'''
    <style>
    /* Set Berkeley Blue as the background color */
    .reportview-container {{
        background-color: {white};
    }}
    .sidebar .sidebar-content {{
        background-color: {berkeley_blue};
    }}
    /* Header styling */
    .main-header {{
        background-color: {berkeley_blue};
        padding: 20px;
        color: {california_gold};
        text-align: center;
        border-radius: 8px;
    }}
    .subheader {{
        background-color: {berkeley_blue};
        padding: 15px;
        color: {white};
        text-align: center;
        border-radius: 8px;
        font-size: 18px;
    }}
    </style>
''', unsafe_allow_html=True)

# Render the header and subheader
st.markdown(f"""
    <div class='header-container' style='background-color:{berkeley_blue}; padding:20px; border-radius:10px; text-align:center; margin-bottom:30px;'>
        <h1 style='color:{california_gold};'>📊 AI-Powered Portfolio Optimizer</h1>
        <p style='color:{white}; font-size:18px;'>Enter your stock universe, specify risk tolerance, and get an optimized portfolio to minimize drawdowns during regime changes.</p>
    </div>
""", unsafe_allow_html=True)


# Initialize session state for tickers, weights, and sentiment
if "tickers" not in st.session_state:
    st.session_state["tickers"] = []
if "weights" not in st.session_state:
    st.session_state["weights"] = {}
if "sentiment" not in st.session_state:
    st.session_state["sentiment"] = {}

# Create a row with columns for investment universe, risk tolerance, stock tickers input, and optimization option
cols = st.columns(4)

# Input: User-specified investment universe
with cols[0]:
    st.subheader("Investment Universe")
    universe_options = ["Technology", "Healthcare", "Finance", "Energy", "Consumer Goods"]
    selected_universe = st.selectbox("Choose a sector or asset class", universe_options)

# Input: User-specified risk tolerance (ex-ante volatility level)
with cols[1]:
    st.subheader("Risk Tolerance (Annual Volatility)")
    risk_tolerance = st.slider("Select your risk level (volatility)", 5, 30, 10)

# Input: Stock tickers (within the selected universe)
with cols[2]:
    st.subheader("Stock Tickers")
    tickers_input = st.text_input("Enter Tickers", placeholder="e.g., AAPL, TSLA, AMZN")

# Input: Optimization option (dropdown)
with cols[3]:
    st.subheader("Optimization Option")
    optimization_options = [
        "Minimize Volatility",
        "Maximize Sharpe Ratio",
        "Maximize Return",
        "Minimize Maximum Drawdown",
        "Target Volatility",
        "Equal Weighting"
    ]
    selected_optimization = st.selectbox("Choose an optimization objective", optimization_options)

# Process tickers and reset weights if tickers change
if tickers_input:
    new_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if set(new_tickers) != set(st.session_state["tickers"]):
        st.session_state["weights"] = {t: st.session_state["weights"].get(t, 0) for t in new_tickers}
    st.session_state["tickers"] = new_tickers

# Show weights and market sentiment
if st.session_state["tickers"]:
    st.subheader("Specify Initial Weights (%) and View Market Sentiment")
    columns = st.columns(len(st.session_state["tickers"]))

    for idx, ticker in enumerate(st.session_state["tickers"]):
        with columns[idx % len(columns)]:
            st.session_state["weights"][ticker] = st.number_input(
                f"{ticker} Weight (%)", min_value=0, max_value=100, 
                value=st.session_state["weights"].get(ticker, 0), step=1,
                key=f"weight_{ticker}"
            )
            # Placeholder for sentiment score (this will be integrated with the ML model)
            st.session_state["sentiment"][ticker] = random.uniform(-1, 1)
            st.metric(label=f"{ticker} Sentiment Score", value=round(st.session_state["sentiment"][ticker], 2))

    total_weight = sum(st.session_state["weights"].values())

    if total_weight == 100:
        st.success(f"✅ Total Weight: {total_weight}% (Balanced)")
    else:
        st.error(f"⚠️ Total Weight: {total_weight}% (Please balance to 100%)")

    # Placeholder for portfolio suggestions based on model (can be connected to ML model later)
    st.subheader("Proposed Trades Based on Optimized Portfolio")
    proposed_weights = {t: random.uniform(5, 20) for t in st.session_state["tickers"]}  # Mock proposed weights

    # Show proposed weights and difference
    st.write(pd.DataFrame({
        "Ticker": st.session_state["tickers"],
        "Current Weight (%)": [st.session_state["weights"][t] for t in st.session_state["tickers"]],
        "Proposed Weight (%)": [proposed_weights[t] for t in st.session_state["tickers"]],
        "Weight Difference (%)": [proposed_weights[t] - st.session_state["weights"][t] for t in st.session_state["tickers"]]
    }))

# Cumulative return chart (mock data)
st.subheader("Cumulative Return Comparison")
data = {
    "Date": pd.date_range(start="2022-01-01", periods=100, freq="D"),
    "Optimized Portfolio": [random.uniform(90, 110) for _ in range(100)],
    "Benchmark Portfolio": [random.uniform(85, 105) for _ in range(100)]
}
df = pd.DataFrame(data)
fig = px.line(df, x="Date", y=["Optimized Portfolio", "Benchmark Portfolio"], title="Cumulative Returns")
st.plotly_chart(fig)

# Add footer manually with Streamlit's markdown component
st.markdown(f"""
    <div style='background-color:{berkeley_blue}; padding:10px; color:{california_gold}; text-align:left; border-radius:10px;'>
        © MIDS 2025
    </div>
""", unsafe_allow_html=True)

