import streamlit as st
import random
import requests
import pandas as pd
import json

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
rebal_freq = "D" #st.sidebar.selectbox("Rebalancing Frequency", ["Monthly", "Quarterly", "Yearly"])
opt_flag = "max_sharpe"  # Fixed to optimize for max Sharpe ratio

# Optimization strategy disclaimer (Moved to footer style)
st.sidebar.markdown("""
    ---
    **Optimization Objective:**  
    This portfolio is optimized to maximize the **Sharpe ratio**, balancing return and risk.
""")

# -------------------- CALL AWS BENCHMARK API -------------------- #
def fetch_benchmark_data(tickers, risk, freq):
    url = "http://52.41.158.44:8000/benchmark" #TODO -- will be changed later on, Elaine to tell when
    payload = {
        "ticker_list": tickers,
        "rebal_freq": freq,
        "opt_flag": opt_flag,
        "target_risk": risk / 100  # Convert percentage to decimal
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an HTTPError if the response code is 4xx/5xx
        
        if response.status_code == 200:
            # Try parsing the response as JSON
            try:
                response_json = response.json()
                # st.write("API Response (Raw):", response.text)  # Print raw response text for debugging
                return response_json
            except ValueError as e:
                st.error(f"Error parsing JSON response: {e}")
                # st.write("Raw Response:", response.text)  # Print raw response content
                return None
        else:
            st.error(f"API request failed with status code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        # Handle other request errors like network issues
        st.error(f"Request failed: {e}")
        return None


def parse_api_data(data):
    """
    Checks if the input data is a string and tries to parse it as JSON. 
    If it's already a list of dictionaries, it leaves the data as is.
    
    Args:
    data: The data to be parsed (can be a string or a list of dictionaries).
    
    Returns:
    Parsed data as a list of dictionaries or the original data if already parsed.
    """
    # If data is a string, try to parse it as JSON
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON data: {e}")
            return None  # Return None or handle the error as needed
    return data

def convert_to_dataframe(data):
    """
    Converts the parsed data (e.g., portf_rtn_test or portf_mkt_rtn_test) into a pandas DataFrame.
    
    Args:
    data: List of dictionaries containing either portfolio returns or market returns data.
    
    Returns:
    A pandas DataFrame with appropriate columns based on the type of data.
    """
    if data is None or not isinstance(data, list):
        print("Invalid data format. Could not convert to DataFrame.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Convert 'date' field to datetime if it exists
    if 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'], unit='ms')
        df.drop('date', axis=1, inplace=True)
    
    # Check for the presence of columns to differentiate between portf_rtn_test and portf_mkt_rtn_test
    if '0' in df.columns:
        # This is likely portf_rtn_test data, rename accordingly
        df.rename(columns={'0': 'Return'}, inplace=True)
    
    elif 'Daily-Benchmark-max_sharpe-test-last window' in df.columns:
        # This is likely portf_mkt_rtn_test data, rename accordingly
        df.rename(columns={
            'Daily-Benchmark-max_sharpe-test-last window': 'Benchmark Return',
            'Unscaled Market': 'Unscaled Market Return',
            'cum_rtn': 'Cumulative Return',
            'drawdown': 'Drawdown',
            'max_drawdown': 'Max Drawdown'
        }, inplace=True)
    
    return df


def convert_to_weight_dataframe(data):
    """
    Converts the parsed scaled weight data into a pandas DataFrame.
    
    Args:
    data: List of dictionaries containing the scaled weight data.
    
    Returns:
    A pandas DataFrame where keys are identifiers and values are the weights (including nulls).
    """
    if data is None or not isinstance(data, list):
        print("Invalid data format. Could not convert to DataFrame.")
        return None
    
    # Convert list of dictionaries into a single dictionary
    combined_data = {k: v for d in data for k, v in d.items()}
    
    return pd.DataFrame(list(combined_data.items()), columns=['Identifier', 'Weight'])



# -------------------- FETCHING & DISPLAYING OPTIMIZED PORTFOLIO DATA -------------------- #
if selected_tickers:
    st.subheader("Fetching Optimized Portfolio Data... ⏳")
    
    benchmark_data = fetch_benchmark_data(selected_tickers, risk_tolerance, rebal_freq)

    if benchmark_data:
        st.write("API Response Keys:", benchmark_data.keys())
        st.write("Full API Response:", benchmark_data)

        # -------------------- PIE CHART: Optimized Portfolio Weights -------------------- #
        # Check the content of 'scaled_weight_df_test'
        scaled_weight_data = benchmark_data.get("scaled_weight_df_test", [])
        scaled_weight_data_parsed = parse_api_data(scaled_weight_data)
        scaled_weight_df = convert_to_weight_dataframe(scaled_weight_data_parsed)

        if scaled_weight_data and isinstance(scaled_weight_data, list) and len(scaled_weight_data) > 0:
            # Try creating DataFrame only if the data is valid
            scaled_weights_df = pd.DataFrame(scaled_weight_data)
            st.subheader("Optimized Portfolio Weights (Pie Chart)")
            fig_pie = px.pie(scaled_weights_df, values="Weight", names="Ticker", title="Optimized Portfolio Asset Allocation")
            st.plotly_chart(fig_pie, use_container_width=True, key="optimized_pie_chart")
        else:
            st.error("Optimized weights not found or invalid in API response.")
        
        # -------------------- BACKTEST PERFORMANCE: Benchmark vs ML Portfolio -------------------- #
        st.subheader("Backtest Performance Over Time (ML vs. Benchmark)")

        # Retrieve the benchmark and ML-based portfolio returns
        portf_rtn_test = benchmark_data.get("portf_rtn_test", [])  # ML Portfolio Return
        portf_rtn_test_parsed = parse_api_data(portf_rtn_test)
        portf_rtn_df = convert_to_dataframe(portf_rtn_test_parsed)

        portf_mkt_rtn_test = benchmark_data.get("portf_mkt_rtn_test", [])  # Benchmark Portfolio Return
        portf_mkt_rtn_test_parsed = parse_api_data(portf_mkt_rtn_test)
        portf_mkt_rtn_df = convert_to_dataframe(portf_mkt_rtn_test_parsed)

        if portf_rtn_df is not None and portf_mkt_rtn_df is not None:
            fig_backtest = go.Figure()

            # Plot ML-optimized portfolio returns
            fig_backtest.add_trace(go.Scatter(x=portf_rtn_df['Date'], y=portf_rtn_df['Return'],
                                              mode='lines', name='ML Portfolio Return',
                                              line=dict(color=positive_color, width=2)))

            # Plot benchmark returns
            fig_backtest.add_trace(go.Scatter(x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Benchmark Return'],
                                              mode='lines', name='Benchmark Return',
                                              line=dict(color=berkeley_blue, dash='dash', width=2)))

            # Update layout to position legend on the right
            fig_backtest.update_layout(
                # title="Backtest Performance (Machine Learning vs Benchmark)",
                xaxis_title="Date",
                yaxis_title="Return (%)",
                legend=dict(x=0, y=1.05, orientation="h", xanchor="left", yanchor="bottom"),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            st.plotly_chart(fig_backtest, use_container_width=True)
        else:
            st.error("Portfolio returns data unavailable for backtesting.")

        # -------------------- CUMULATIVE RETURNS CHART -------------------- #
        st.subheader("Cumulative Returns Comparison (Benchmark vs. Unscaled Market)")

        if portf_mkt_rtn_df is not None:
            portf_mkt_rtn_df['Cumulative Benchmark Return'] = (1 + portf_mkt_rtn_df['Benchmark Return']).cumprod()
            portf_mkt_rtn_df['Cumulative Unscaled Market Return'] = (1 + portf_mkt_rtn_df['Unscaled Market Return']).cumprod()

            fig_cumulative = go.Figure()

            # Plot cumulative returns for the benchmark
            fig_cumulative.add_trace(go.Scatter(x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Cumulative Benchmark Return'],
                                                mode='lines', name='Cumulative Benchmark Return',
                                                line=dict(color=berkeley_blue, width=2)))

            # Plot cumulative unscaled market returns
            fig_cumulative.add_trace(go.Scatter(x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Cumulative Unscaled Market Return'],
                                                mode='lines', name='Cumulative Unscaled Market Return',
                                                line=dict(color=california_gold, width=2)))

            # Update layout for the cumulative returns comparison
            fig_cumulative.update_layout(
                xaxis_title="Date", 
                yaxis_title="Cumulative Return (%)",
                legend=dict(x=0, y=1.05, orientation="h", xanchor="left", yanchor="bottom"), 
                margin=dict(l=40, r=40, t=40, b=40)
            )

            st.plotly_chart(fig_cumulative, use_container_width=True)

        # -------------------- STATS DATAFRAME -------------------- #
        if "stats_df_test" in benchmark_data:
            st.subheader("Portfolio Statistics")

            # Extract and display 'stats_df_test' as a dataframe/table
            stats_df = benchmark_data.get("stats_df_test", [])
            stats_df_parsed = parse_api_data(stats_df)

            if stats_df_parsed:
                stats_df_display = pd.DataFrame(stats_df_parsed)  # Convert parsed data to DataFrame
                
                # Display DataFrame using Streamlit's st.dataframe
                st.dataframe(stats_df_display)  # You can use st.table() for a static table view

            else:
                st.error("Error parsing or displaying portfolio statistics.")

        # -------------------- SENTIMENT INSIGHTS -------------------- #
        st.markdown("---")        
        sentiment_df = pd.read_parquet('Sentiment_Predictions_w_Imputed_Values.parquet')
        last_day = sentiment_df.iloc[-1:,:]
        # st.write(last_day)

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
            st.markdown(f"""
                <div style="text-align: left; display: flex; align-items: center; justify-content: flex-start; height: 100%;">
                    <h6>FinBERT Sentiment Breakdown:</h6>
                </div>
            """, unsafe_allow_html=True)
            
            # Horizontal stacked bar chart for sentiment breakdown
            fig_sentiment = go.Figure(
                data=[
                    go.Bar(
                        y=["Sentiment"],
                        x=[finbert_sentiment_probs["Negative"]],
                        name="Negative",
                        marker_color=berkeley_blue,
                        orientation="h",
                        hoverinfo="none",
                    ),
                    go.Bar(
                        y=["Sentiment"],
                        x=[finbert_sentiment_probs["Neutral"]],
                        name="Neutral",
                        marker_color="rgba(45, 85, 115, 0.5)",
                        orientation="h",
                        hoverinfo="none",
                    ),
                    go.Bar(
                        y=["Sentiment"],
                        x=[finbert_sentiment_probs["Positive"]],
                        name="Positive",
                        marker_color=california_gold,
                        orientation="h",
                        hoverinfo="none",
                    ),
                ]
            )
            fig_sentiment.update_layout(
                barmode="stack",
                xaxis=dict(title="Probability"),
                height=180,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True,
                legend=dict(traceorder="normal", x=0, y=1.05, orientation="h", xanchor="left", yanchor="bottom"), 
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

        # Mock Next-Day Return Probability with improved layout
        next_day_return_prob = last_day['predicted_continuous_return'].values[0]

        with return_col:
            st.markdown(f"""
                <div style="text-align: left; display: flex; align-items: center; justify-content: flex-start; height: 100%;">
                    <h6>📈 Probability of Positive Next-Day Return: <b>{next_day_return_prob * 100:.1f}%</b></h6>
                </div>
            """, unsafe_allow_html=True)

            # Centered progress bar
            st.progress(next_day_return_prob)


else:
    st.warning("Please select tickers to optimize your portfolio.")


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
