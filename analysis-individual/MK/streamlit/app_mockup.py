import streamlit as st
from fetch_api import fetch_benchmark_data, fetch_model_data, fetch_s3_data
from parse_utils import parse_api_data, convert_to_dataframe, convert_to_weight_dataframe
from charts import ChartGenerator
from theme import berkeley_blue, california_gold, white, light_grey  # Import colors

import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib

import requests
import pickle
import io
import sys

print('sys: {}'.format(sys.version))
# print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pd.__version__))



# Initialize the chart generator
chart_generator = ChartGenerator()

st.set_page_config(page_title="AI Portfolio Optimizer", page_icon="📈", layout="wide")

# Header & subheader
st.markdown(f"""
    <div class='header-container' style='background-color:{berkeley_blue}; padding:20px; border-radius:10px; text-align:center; margin-bottom:30px;'>
        <h1 style='color:{california_gold};'>📊 AI-Powered Portfolio Optimizer</h1>
        <p style='color:{white}; font-size:18px;'>Enter your stock universe, specify risk tolerance, and get an optimized portfolio to minimize drawdowns during regime changes.</p>
    </div>
""", unsafe_allow_html=True)

# Apply custom CSS
st.markdown(f"""
    <style>
        .sidebar .sidebar-content {{
            background-color: {light_grey};
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

# Sidebar and other app logic...


# -------------------- SIDEBAR: USER INPUTS -------------------- #
st.sidebar.header("Portfolio Configuration")
stock_options = [100, 200, 300]
selected_stock_count = st.sidebar.selectbox("Select number of stocks:", stock_options)

# Fetch tickers from the S3 based on the selected stock count
tickers_df = fetch_s3_data(data_type="tickers", stock_count=selected_stock_count)

if tickers_df is not None:
    selected_tickers = st.sidebar.multiselect(f"Choose stocks from the Top {selected_stock_count} list:", tickers_df)
    st.write(selected_tickers)

if selected_tickers == 200:
    ticker_list = ['top_200_ticker_l']
elif selected_tickers == 300:
    ticker_list = ['top_300_ticker_l']
else:
    ticker_list = ['top_100_ticker_l']

risk_tolerance = st.sidebar.slider(
    "Risk Tolerance (Annual Volatility)", 
    0.01, 0.99, 0.2, 
    help="Select a risk tolerance between 0 and 1."
)

benchmark_data = fetch_benchmark_data(ticker_list, "D", "max_sharpe", risk_tolerance, False)
model_data = fetch_model_data(ticker_list, "D", "max_sharpe", risk_tolerance, False)

fetching_message = st.empty()

# Fetching portfolio data
if selected_tickers:
    fetching_message.subheader("Fetching Optimized Portfolio Data... ⏳")


    if model_data: #benchmark_data and model_data:
        # st.write("API Response Keys:", benchmark_data.keys())
        # st.write("Full API Response:", benchmark_data)
        st.write("API Response Keys:", model_data.keys())
        st.write("Full API Response:", model_data)


        # Clear the "Fetching Optimized Portfolio Data..." message
        fetching_message.empty()

        benchmark_portf_rtn_df = convert_to_dataframe(parse_api_data(benchmark_data.get("portf_rtn_test", [])))
        benchmark_portf_mkt_rtn_df = convert_to_dataframe(parse_api_data(benchmark_data.get("portf_mkt_rtn_test", [])))
        benchmark_scaled_weight_df = convert_to_weight_dataframe(parse_api_data(benchmark_data.get("scaled_weight_df_test", [])))
        
        # Generate and display charts
        st.subheader("Optimized Portfolio Weights (Pie Chart)")
        # chart_generator.generate_pie_chart(scaled_weight_df)

        st.subheader("Performance Comparison  (Benchmark vs. Unscaled Market)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("###### Cumulative Returns Comparison")
            # st.write("Columns in DataFrame:", portf_mkt_rtn_df.columns)
            chart_generator.generate_cumulative_returns_chart(benchmark_portf_mkt_rtn_df)
        with col2:
            st.markdown("###### Rolling Annual Volatility Comparison")
            chart_generator.generate_volatility_chart(benchmark_portf_mkt_rtn_df)

        st.subheader("Portfolio Weights Table")
        chart_generator.generate_weights_table(benchmark_scaled_weight_df)

        st.subheader("Portfolio Statistics")
        chart_generator.generate_stats_table(benchmark_stats_parsed_df)

        st.subheader("Sentiment Breakdown")
        last_day = pd.DataFrame(...)  # Assuming you have the sentiment data in a dataframe.
        chart_generator.generate_sentiment_chart(last_day)

        # Backtest performance chart
        st.subheader("Backtest Performance")
        chart_generator.generate_backtest_chart(benchmark_portf_rtn_df, benchmark_portf_mkt_rtn_df)
    else:
        st.error("Could not fetch data for the selected portfolio configuration.")
else: #s3
    stats_parsed_df = fetch_s3_data("stats")
    st.subheader("Portfolio Statistics")
    if stats_parsed_df is not None:
        chart_generator.generate_stats_table(stats_parsed_df, True)
    else:
        st.write('here')
        benchmark_stats_parsed_df = pd.DataFrame(parse_api_data(benchmark_data.get("stats_df_test", [])))
        chart_generator.generate_stats_table(stats_parsed_df, False)
    
    cum_df, vol_df = fetch_s3_data("backtest") #backtest_fig, 
    # st.write(cum_df)
    # st.write(vol_df)
    st.subheader("Backtest Performance")
    if cum_df is not None and vol_df is not None:
        # st.write(backtest_fig)
        # st.write(cum_df)
        # st.write(vol_df)
        chart_generator.generate_backtest_chart(cum_df, vol_df)
    else:
        #temp, probs need to change bc of "calcs"
        st.write('here 2')
        benchmark_portf_mkt_rtn_df = convert_to_dataframe(parse_api_data(benchmark_data.get("portf_mkt_rtn_test", [])))
        benchmark_scaled_weight_df = convert_to_weight_dataframe(parse_api_data(benchmark_data.get("scaled_weight_df_test", [])))
        chart_generator.generate_backtest_chart(portf_rtn_df, portf_mkt_rtn_df)


# Footer
st.markdown(f"""
    <div class="footer">
        &copy; 2025 AI-Powered Portfolio Optimizer
    </div>
""", unsafe_allow_html=True)
