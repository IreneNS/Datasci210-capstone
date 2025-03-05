import streamlit as st
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Define color palette
berkeley_blue = "#002676"
california_gold = "#FDB515"
white = "#FFFFFF"
negative_color = "#D62728"  # Red
neutral_color = "#7F7F7F"  # Gray
positive_color = "#2CA02C"  # Green

st.set_page_config(page_title="AI Portfolio Optimizer", page_icon="📈", layout="wide")

# Apply custom CSS for cleaner UI aesthetics
st.markdown(f"""
    <div class='header-container' style='background-color:{berkeley_blue}; padding:20px; border-radius:10px; text-align:center; margin-bottom:30px;'>
        <h1 style='color:{california_gold};'>📊 AI-Powered Portfolio Optimizer</h1>
        <p style='color:{white}; font-size:18px;'>Enter your stock universe, specify risk tolerance, and get an optimized portfolio to minimize drawdowns during regime changes.</p>
    </div>
""", unsafe_allow_html=True)

# -------------------- USER INPUT SECTION -------------------- #

# Arrange inputs in a grid format
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

# Ensure session state for tickers and weights
if "tickers" not in st.session_state:
    st.session_state["tickers"] = []
if "weights" not in st.session_state:
    st.session_state["weights"] = {}

# Process selected tickers and store in session state
if selected_tickers:
    st.session_state["tickers"] = selected_tickers
    st.session_state["weights"] = {t: st.session_state["weights"].get(t, 0) for t in selected_tickers}

# Ensure all portfolio inputs are filled before displaying charts
if len(st.session_state["tickers"]) > 0:
    
    st.subheader("Specify Initial Weights (%)")
    columns = st.columns(len(st.session_state["tickers"]))

    for idx, ticker in enumerate(st.session_state["tickers"]):
        with columns[idx % len(columns)]:
            current_value = st.session_state["weights"].get(ticker, 0)
            new_value = st.number_input(
                f"{ticker} Weight (%)", min_value=0, max_value=100, 
                value=current_value, step=1,
                key=f"weight_{ticker}"
            )
            if new_value != current_value:
                st.session_state["weights"][ticker] = new_value

    total_weight = sum(st.session_state["weights"].values())

    if total_weight == 100:
        st.success(f"✅ Total Weight: {total_weight}% (Balanced)")
    else:
        st.error(f"⚠️ Total Weight: {total_weight}% (Please balance to 100%)")

    # -------------------- PORTFOLIO CHARTS -------------------- #

    # Show Weights Pie Chart
    st.subheader("Weights Distribution")
    weights_data = pd.DataFrame({
        "Ticker": st.session_state["tickers"],
        "Weight": [st.session_state["weights"][t] for t in st.session_state["tickers"]]
    })
    pie_chart_fig = px.pie(weights_data, names="Ticker", values="Weight", title="Portfolio Weights")
    st.plotly_chart(pie_chart_fig, use_container_width=True)  # Full-width pie chart

    # Add Backtest Chart
    backtest_placeholder = st.empty()
    backtest_data = {
        "Date": pd.date_range(start="2021-01-01", periods=100, freq="D"),
        "Backtest Performance": [random.uniform(80, 120) for _ in range(100)]
    }
    backtest_df = pd.DataFrame(backtest_data)
    backtest_fig = px.line(backtest_df, x="Date", y="Backtest Performance", title="Backtest Performance Over Time")
    backtest_placeholder.plotly_chart(backtest_fig, use_container_width=True)  # Full-width backtest chart

    # Add Cumulative Return Chart
    cumulative_placeholder = st.empty()
    data = {
        "Date": pd.date_range(start="2022-01-01", periods=100, freq="D"),
        "Optimized Portfolio": [random.uniform(90, 110) for _ in range(100)],
        "Benchmark Portfolio": [random.uniform(85, 105) for _ in range(100)]
    }
    df = pd.DataFrame(data)
    cumulative_fig = px.line(df, x="Date", y=["Optimized Portfolio", "Benchmark Portfolio"], title="Cumulative Returns")
    cumulative_placeholder.plotly_chart(cumulative_fig, use_container_width=True)  # Full-width cumulative chart

    # -------------------- SENTIMENT INSIGHTS -------------------- #
    st.markdown("---")
    st.subheader("Sentiment Insights 📰")

    # Arrange sentiment breakdown and next-day return in side-by-side columns (equal width)
    sentiment_col, return_col = st.columns([1, 1], gap="large")  # Equal column width

    # Mock FinBERT sentiment probabilities
    finbert_sentiment_probs = {
        "Negative": random.uniform(0, 0.5),
        "Neutral": random.uniform(0, 0.5),
        "Positive": random.uniform(0, 0.5),
    }
    # Normalize to sum to 1
    total = sum(finbert_sentiment_probs.values())
    finbert_sentiment_probs = {k: v / total for k, v in finbert_sentiment_probs.items()}

    # Create horizontal stacked bar chart with improved height and styling
    with sentiment_col:
        fig_sentiment = go.Figure(
            data=[
                go.Bar(
                    y=["Sentiment"],
                    x=[finbert_sentiment_probs["Negative"]],
                    name="Negative",
                    marker_color=negative_color,
                    orientation="h",
                    hoverinfo="none",  # Remove excessive hover details
                ),
                go.Bar(
                    y=["Sentiment"],
                    x=[finbert_sentiment_probs["Neutral"]],
                    name="Neutral",
                    marker_color=neutral_color,
                    orientation="h",
                    hoverinfo="none",
                ),
                go.Bar(
                    y=["Sentiment"],
                    x=[finbert_sentiment_probs["Positive"]],
                    name="Positive",
                    marker_color=positive_color,
                    orientation="h",
                    hoverinfo="none",
                ),
            ]
        )
        fig_sentiment.update_layout(
            barmode="stack",
            title="FinBERT Sentiment Breakdown",
            xaxis=dict(title="Probability"),
            height=180,  # Reduced height for a sleek look
            margin=dict(l=20, r=20, t=40, b=20),  # Add margin for breathing room
            showlegend=True,
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=-0.5,  # Adjust to place legend below the chart
                xanchor="center",
                x=0.5,
            ),
            modebar_remove=["zoom", "pan", "reset", "autoscale"],  # Hide unnecessary UI buttons
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    # Mock Next-Day Return Probability with improved layout
    next_day_return_prob = round(random.uniform(0, 1), 2)

    with return_col:
        # Center text and progress bar
        st.markdown(f"""
            <div style="text-align: center;">
                <h3>📈 Probability of Positive Next-Day Return: <b>{next_day_return_prob * 100:.1f}%</b></h3>
            </div>
        """, unsafe_allow_html=True)

        # Display progress bar
        st.progress(next_day_return_prob)


    # Footer
    st.markdown(
        f"<div style='background-color:{berkeley_blue}; padding:10px; color:{california_gold}; text-align:right;'>MIDS 2025</div>",
        unsafe_allow_html=True
    )
