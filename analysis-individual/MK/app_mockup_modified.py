import streamlit as st
import random
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64

# Define color palette
berkeley_blue = "#002676"
california_gold = "#FDB515"
white = "#FFFFFF"
negative_color = "#D62728"  # Red
neutral_color = "#7F7F7F"  # Gray
positive_color = "#2CA02C"  # Green

st.set_page_config(page_title="AI Portfolio Optimizer", page_icon="📈", layout="wide")

# Header & subheader 
st.markdown(f"""
    <div class='header-container' style='background-color:{berkeley_blue}; padding:20px; border-radius:10px; text-align:center; margin-bottom:30px;'>
        <h1 style='color:{california_gold};'>📊 AI-Powered Portfolio Optimizer</h1>
        <p style='color:{white}; font-size:18px;'>Enter your stock universe, specify risk tolerance, and get an optimized portfolio to minimize drawdowns during regime changes.</p>
    </div>
""", unsafe_allow_html=True)

# Apply custom CSS for a modern UI look
st.markdown(f"""
    <style>
        .sidebar .sidebar-content {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
        }}
        .footer {{
            background-color: {berkeley_blue};
            padding: 10px;
            text-align: right;
            color: {california_gold};
            font-size: 12px;
        }}
    </style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR: USER INPUTS -------------------- #
st.sidebar.header("Portfolio Configuration")
ticker_options = ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "NFLX", "META"]
selected_tickers = st.sidebar.multiselect("Choose stocks", ticker_options)

risk_tolerance = st.sidebar.slider("Risk Tolerance (Annual Volatility)", 5, 30, 15, help="Higher values indicate a willingness to take on more volatility.")
rebal_freq = "M" #st.sidebar.selectbox("Rebalancing Frequency", ["Monthly", "Quarterly", "Yearly"])
opt_flag = "max_sharpe"  # Fixed to optimize for max Sharpe ratio

# Optimization strategy disclaimer (Moved to footer style)
st.sidebar.markdown("""
    ---
    **Optimization Objective:**  
    This portfolio is optimized to maximize the **Sharpe ratio**, balancing return and risk.
""")

# -------------------- CALL AWS BENCHMARK API -------------------- #
def fetch_benchmark_data(tickers, risk, freq):
    url = "http://52.41.158.44:8000/benchmark"
    payload = {
        "ticker_list": tickers,
        "rebal_freq": freq,
        "opt_flag": opt_flag,
        "target_risk": risk / 100  # Convert percentage to decimal
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()  # Returns 4x DataFrames as JSON
    else:
        st.error("API request failed. Please check input parameters.")
        return None

# -------------------- FETCHING & DISPLAYING OPTIMIZED PORTFOLIO DATA -------------------- #
if selected_tickers:
    st.subheader("Fetching Optimized Portfolio Data... ⏳")
    
    benchmark_data = fetch_benchmark_data(selected_tickers, risk_tolerance, rebal_freq)
    
    if benchmark_data:
        st.write("API Response Keys:", benchmark_data.keys())
        st.write("Full API Response:", benchmark_data)

        # -------------------- PIE CHART: Optimized Portfolio Weights -------------------- #
        scaled_weights_df = pd.DataFrame(benchmark_data.get("scaled_weight_df_test", []))
        if not scaled_weights_df.empty:
            st.subheader("Optimized Portfolio Weights (Pie Chart)")
            fig_pie = px.pie(scaled_weights_df, values="Weight", names="Ticker", title="Optimized Portfolio Asset Allocation")
            st.plotly_chart(fig_pie, use_container_width=True, key="optimized_pie_chart")
        else:
            st.error("Optimized weights not found in API response.")
        
        # -------------------- BACKTEST PERFORMANCE: Benchmark vs ML Portfolio -------------------- #
        st.subheader("Backtest Performance Over Time")

        # Retrieve the benchmark and ML-based portfolio returns
        portf_rtn_test = pd.DataFrame(benchmark_data.get("portf_rtn_test", []))  # ML Portfolio Return
        portf_mkt_rtn_test = pd.DataFrame(benchmark_data.get("portf_mkt_rtn_test", []))  # Benchmark Portfolio Return
        
        if not portf_rtn_test.empty and not portf_mkt_rtn_test.empty:
            # Ensure we have a Date column, assuming dates are indexed
            portf_rtn_test['Date'] = pd.to_datetime(portf_rtn_test['Date'])
            portf_mkt_rtn_test['Date'] = pd.to_datetime(portf_mkt_rtn_test['Date'])
            
            # Merge the two dataframes on Date for plotting
            merged_df = pd.merge(portf_rtn_test[['Date', 'Return']], portf_mkt_rtn_test[['Date', 'Return']], on='Date', suffixes=('_ML', '_Benchmark'))
            
            # Plot both returns
            fig_performance = go.Figure()

            fig_performance.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Return_ML'], mode='lines', name="ML Portfolio Return", line=dict(color=positive_color)))
            fig_performance.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Return_Benchmark'], mode='lines', name="Benchmark Portfolio Return", line=dict(color=neutral_color)))

            fig_performance.update_layout(
                title="Backtest Performance: ML Portfolio vs Benchmark",
                xaxis_title="Date",
                yaxis_title="Return",
                template="plotly_dark",
                height=600
            )

            st.plotly_chart(fig_performance, use_container_width=True, key="backtest_performance")

        else:
            st.error("Backtest performance data for portfolio or benchmark return is missing in the API response.")
        
        # -------------------- CUMULATIVE RETURN GRAPH -------------------- #
        st.subheader("Cumulative Portfolio Returns")

        # Dummy Data for Cumulative Returns (For illustration)
        cumulative_data = {
            "Date": pd.date_range(start="2022-01-01", periods=100, freq="D"),
            "Optimized Portfolio": [random.uniform(90, 110) for _ in range(100)],
            "Benchmark Portfolio": [random.uniform(85, 105) for _ in range(100)],
        }
        cumulative_df = pd.DataFrame(cumulative_data)
        
        # Calculate Cumulative Returns
        cumulative_df["Optimized Portfolio"] = (1 + cumulative_df["Optimized Portfolio"] / 100).cumprod()
        cumulative_df["Benchmark Portfolio"] = (1 + cumulative_df["Benchmark Portfolio"] / 100).cumprod()

        # Plot the Cumulative Return graph
        fig_cumulative = px.line(cumulative_df, x="Date", y=["Optimized Portfolio", "Benchmark Portfolio"], 
                                 title="Cumulative Portfolio Returns Over Time")
        st.plotly_chart(fig_cumulative, use_container_width=True, key="cumulative_returns")

        # -------------------- SENTIMENT INSIGHTS -------------------- #
        st.subheader("Sentiment Insights")

        # Create a row with 2 columns: One for the pie chart and one for the progress bar
        col1, col2 = st.columns([3, 2])  # Adjust the ratio of the columns

        with col1:
            sentiment_data = {
                "Sentiment": ["Positive", "Negative", "Neutral"],
                "Percentage": [60, 20, 20]
            }
            sentiment_df = pd.DataFrame(sentiment_data)
            
            # Plot the FinBERT Sentiment Breakdown Pie Chart
            fig_sentiment = px.pie(sentiment_df, values="Percentage", names="Sentiment", title="FinBERT Sentiment Breakdown")
            st.plotly_chart(fig_sentiment, use_container_width=True, key="finbert_sentiment")

        with col2:
            prob_data = {
                "Next-Day Return": ["Positive", "Negative"],
                "Probability": [0.75, 0.25]
            }
            prob_df = pd.DataFrame(prob_data)
            positive_prob = prob_df.loc[prob_df["Next-Day Return"] == "Positive", "Probability"].values[0]
            st.subheader("Probability of Positive Next-Day Return")
            st.progress(positive_prob)
            st.write(f"Probability of Positive Next-Day Return: {positive_prob * 100:.2f}%")

        # -------------------- DOWNLOADABLE REPORT -------------------- #
        st.subheader("Download Portfolio Report 📥")
        csv = scaled_weights_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="portfolio_report.csv">📥 Click here to download</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.success("Portfolio data successfully retrieved! ✅")

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
