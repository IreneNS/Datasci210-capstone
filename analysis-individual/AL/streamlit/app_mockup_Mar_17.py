import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
import random

# Define color palette
berkeley_blue = "#002676"
california_gold = "#FDB515"
white = "#FFFFFF"
negative_color = "#D62728"
neutral_color = "#7F7F7F"
positive_color = "#2CA02C"

st.set_page_config(page_title="AI Portfolio Optimizer", page_icon="📈", layout="wide")

# Header
st.markdown(f"""
    <div style='background-color:{berkeley_blue}; padding:20px; border-radius:10px; text-align:center; margin-bottom:30px;'>
        <h1 style='color:{california_gold};'>📊 AI-Powered Portfolio Optimizer</h1>
        <p style='color:{white}; font-size:18px;'>Enter your stock universe, specify risk tolerance, and get an optimized portfolio.</p>
    </div>
""", unsafe_allow_html=True)

# -------------------- USER INPUT SECTION -------------------- #

cols = st.columns(3)

# Column 1: Investment Universe
with cols[0]:
    ticker_options = ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "NFLX", "META"]
    st.subheader("Investment Universe")
    selected_tickers = st.multiselect("Choose stocks", ticker_options)

# Column 2: Risk Tolerance Slider
with cols[1]:
    st.subheader("Risk Tolerance (Annual Volatility)")
    risk_tolerance = st.slider("Select your risk level (volatility)", 5, 30, 15)

# Column 3: Optimization Option Dropdown
with cols[2]:
    st.subheader("Optimization Option")
    optimization_options = ["Maximize Sharpe Ratio", "Minimize Risk"]
    selected_optimization = st.selectbox("Choose an optimization objective", optimization_options)
    opt_flag = "max_sharpe" if selected_optimization == "Maximize Sharpe Ratio" else "min_risk"

# Ensure session state for tickers and weights
if "tickers" not in st.session_state:
    st.session_state["tickers"] = []
if "weights" not in st.session_state:
    st.session_state["weights"] = {}

if selected_tickers:
    st.session_state["tickers"] = selected_tickers
    st.session_state["weights"] = {t: st.session_state["weights"].get(t, 0) for t in selected_tickers}

# Portfolio Weights Input
if len(st.session_state["tickers"]) > 0:
    st.subheader("Specify Initial Weights (%)")
    weight_cols = st.columns(len(st.session_state["tickers"]))

    for idx, ticker in enumerate(st.session_state["tickers"]):
        with weight_cols[idx % len(weight_cols)]:
            current_value = st.session_state["weights"].get(ticker, 0)
            new_value = st.number_input(
                f"{ticker} Weight (%)", min_value=0, max_value=100, 
                value=current_value, step=1, key=f"weight_{ticker}"
            )
            if new_value != current_value:
                st.session_state["weights"][ticker] = new_value

    total_weight = sum(st.session_state["weights"].values())

    if total_weight == 100:
        st.success(f"✅ Total Weight: {total_weight}% (Balanced)")
    else:
        st.error(f"⚠️ Total Weight: {total_weight}% (Please balance to 100%)")

# -------------------- CALL AWS BENCHMARK API -------------------- #
def fetch_benchmark_data(tickers, risk, freq):
    url = "http://52.41.158.44:8000/benchmark"
    payload = {
        "ticker_list": tickers,
        "rebal_freq": freq,
        "opt_flag": opt_flag,
        "target_risk": risk / 100  
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()  
    else:
        st.error("API request failed. Please check input parameters.")
        return None

# -------------------- FETCHING & DISPLAYING OPTIMIZED PORTFOLIO DATA -------------------- #
if selected_tickers:
    st.subheader("Fetching Optimized Portfolio Data... ⏳")
    
    benchmark_data = fetch_benchmark_data(selected_tickers, risk_tolerance, "M")
    st.write(benchmark_data)
    if benchmark_data:
    #     # Portfolio Weights Pie Chart
    #     scaled_weights_df = pd.DataFrame(benchmark_data.get("scaled_weight_df_test", []))
    #     if not scaled_weights_df.empty:
    #         st.subheader("Optimized Portfolio Weights")
    #         fig_pie = px.pie(scaled_weights_df, values="Weight", names="Ticker", title="Portfolio Allocation")
    #         st.plotly_chart(fig_pie, use_container_width=True)
    #     else:
    #         st.error("Optimized weights not found in API response.")

    #     # Backtest Performance Graph
    #     st.subheader("Backtest Performance Over Time")
    #     portf_rtn_test = pd.DataFrame(benchmark_data.get("portf_rtn_test", []))
    #     portf_mkt_rtn_test = pd.DataFrame(benchmark_data.get("portf_mkt_rtn_test", []))

    #     if not portf_rtn_test.empty and not portf_mkt_rtn_test.empty:
    #         portf_rtn_test['Date'] = pd.to_datetime(portf_rtn_test['Date'])
    #         portf_mkt_rtn_test['Date'] = pd.to_datetime(portf_mkt_rtn_test['Date'])
    #         merged_df = pd.merge(portf_rtn_test, portf_mkt_rtn_test, on='Date', suffixes=('_ML', '_Benchmark'))

    #         fig_performance = go.Figure()
    #         fig_performance.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Return_ML'], mode='lines', name="ML Portfolio Return", line=dict(color=positive_color)))
    #         fig_performance.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Return_Benchmark'], mode='lines', name="Benchmark Portfolio Return", line=dict(color=neutral_color)))

    #         fig_performance.update_layout(title="Backtest Performance", xaxis_title="Date", yaxis_title="Return")
    #         st.plotly_chart(fig_performance, use_container_width=True)
    #     else:
    #         st.error("Backtest performance data missing in API response.")

        # -------------------- SENTIMENT INSIGHTS -------------------- #
        st.markdown("---")        
        sentiment_df = pd.read_parquet('Sentiment_Predictions_w_Imputed_Values.parquet')
        last_day = sentiment_df.iloc[-1:,:]
        st.write(last_day)

        st.subheader("Sentiment Insights 📰")

        # Arrange sentiment breakdown and next-day return in side-by-side columns (equal width)
        sentiment_col, return_col = st.columns([1, 1], gap="large")  # Equal column width

        # Mock FinBERT sentiment probabilities
        finbert_sentiment_probs = {
            "Negative": last_day['Negative_Prob'].values[0],
            "Neutral": last_day['Neutral_Prob'].values[0],
            "Positive": last_day['Positive_Prob'].values[0],
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
        next_day_return_prob = last_day['predicted_continuous_return'].values[0]

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
    f"""
    <div style="
        background-color:{berkeley_blue}; 
        padding:10px; 
        color:{california_gold}; 
        text-align:right; 
        border-radius:10px; 
        margin-top:20px;">
        © MIDS 2025
    </div>
    """,
    unsafe_allow_html=True
)
