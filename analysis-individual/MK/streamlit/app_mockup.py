import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from fetch_api import fetch_benchmark_data, fetch_model_data
from fetch_s3 import fetch_s3_data
from parse_utils import parse_api_data, convert_to_dataframe, convert_to_weight_dataframe, convert_to_stats_dataframe, create_stats_dataframe
from charts import ChartGenerator
from theme import *
import view.view_top300 as view_top300
import view.view_full as view_full
import view.view_sentiment as view_sentiment
import view.view_backtest as view_backtest

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
        }}
        .footer {{
            background-color: {berkeley_blue};
            padding: 10px;
            text-align: center;
            color: {california_gold};
            font-size: 12px;
            border-radius: 10px;
        }}
        .footer a {{
            color: {white};
            text-decoration: none;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}

    </style>
""", unsafe_allow_html=True)

button_styles = """
                button {
                    background-color: white;
                    color: #003B5C !important;  /* Berkeley Blue */
                    border: 2px solid #003B5C !important;  /* Berkeley Blue Border */
                    border-radius: 5px;
                    padding: 10px 20px;
                    font-size: 16px;
                    font-weight: bold;
                }

                button:hover {
                background-color: #003B5C;  /* Berkeley Blue */
                color: #FDB515 !important;  /* Yellow (California Gold) */
                border: 2px solid #003B5C;  /* Berkeley Blue */
                }

                button:active {
                background-color: #FDB515 !important;  /* California Gold */
                color: #003B5C !important;  /* Berkeley Blue */
                border: 2px solid #FDB515 !important;  /* Yellow (California Gold) */
                }   

                button:focus { /* Reset the button appearance once clicked */
                    outline: none !important;  /* Remove default focus outline */
                }
                """
selectbox_styles = """
    /* Change border color of selectbox to blue */
    div.stSelectbox > select {
        border: 2px solid #003B5C !important;  /* Berkeley Blue */
        border-radius: 5px !important;
    }

    /* When selectbox is focused, maintain the blue border */
    div.stSelectbox > select:focus {
        border: 2px solid #003B5C !important;  /* Berkeley Blue */
    }
"""
st.markdown(f"""
    <style>
        {selectbox_styles}
    </style>
""", unsafe_allow_html=True)
# .footer {{
#     position: relative;  /* Ensures it's always at the bottom but scrolls with content */
#     width: 100%;
#     color: #6c757d;
#     text-align: center;
#     padding: 10px;
#     font-size: 14px;
#     border-top: 1px solid #ddd;
#     margin-top: 20px;  /* Adds spacing from content */
# }}

# -------------------- SIDEBAR: USER INPUTS -------------------- #
st.sidebar.header("Portfolio Configuration")
with st.sidebar.form('portfolio_form'):
    stock_options = {
                     # 'Top 100 S&P500 Stocks': 100, 
                     # 'Top 200 S&P500 Stocks': 200, 
                     'Top 300 S&P500 Stocks': 300,
                     'Full S&P500 Universe': 0}
    selected_stock_count = st.selectbox("Set Optimization Universe:", stock_options.keys())

    # Fetch tickers from the S3 based on the selected stock count
    tickers_df = fetch_s3_data(data_type="tickers", stock_count=stock_options[selected_stock_count])

    risk_tolerance = st.slider(
        "Risk Tolerance (Annual Volatility)", 
        0.01, 0.99, 0.2, 
        help="Select a risk tolerance between 0 and 1."
        )

    # Wrap the button with a custom style container
    with stylable_container(
        "submit_button_container", 
        css_styles=button_styles
    ):
        submit_button = st.form_submit_button("Run Optimization")

    # Optimization strategy disclaimer (Moved to footer style)
    st.sidebar.markdown("""
        ---
        **Optimization Objective:**\n
        This portfolio is optimized to maximize the **Sharpe ratio**, balancing return and risk.
    """)

cum_df, vol_df = fetch_s3_data("backtest")
# st.write(cum_df)
# st.write(vol_df)
stats_parsed_df = convert_to_stats_dataframe(fetch_s3_data("stats"), True, False)

full = False
if submit_button:
    ticker_list = []
    # if '100' in selected_stock_count:
    #     ticker_list = ['top_100_ticker_l']
    # elif '200' in selected_stock_count:
    #     ticker_list = ['top_200_ticker_l']
    if '300' in selected_stock_count:
        ticker_list = ['top_300_ticker_l']
    else: #full
        ticker_list = []
        full = True

    fetching_message = st.empty()
    fetching_message.subheader("Fetching Optimized Portfolio Data... ⏳")


    benchmark_data = fetch_benchmark_data(ticker_list, "D", "max_sharpe", risk_tolerance, False)
    model_data = fetch_model_data(ticker_list, "D", "max_sharpe", risk_tolerance, False)
    sentiment_df = pd.read_parquet('Sentiment_Predictions_w_Imputed_Values.parquet') #TODO
    last_day = sentiment_df.iloc[-1:,:]


    # Fetching portfolio data
    if selected_stock_count:
        # fetching_message.subheader("Fetching Optimized Portfolio Data... ⏳")

        if model_data:  # benchmark_data and model_data:
            # st.write("API Response Keys:", benchmark_data.keys())
            # st.write("Full API Response:", benchmark_data)
            # st.write("API Response Keys:", model_data.keys())
            # st.write("Full API Response:", model_data)

            # Clear the "Fetching Optimized Portfolio Data..." message
            fetching_message.empty()

            if not full: #only top 300
                print('portf_rtn')
                benchmark_portf_rtn_df = convert_to_dataframe(parse_api_data(benchmark_data.get("portf_rtn_comb", [])), True)
                benchmark_stats_parsed_df = create_stats_dataframe(benchmark_portf_rtn_df)
                # st.write(benchmark_portf_rtn_df)

                print('model portf_rtn')
                model_portf_rtn_df = convert_to_dataframe(parse_api_data(model_data.get("portf_rtn", [])), False)
                print('model portf_mkt_rtn')
                model_portf_mkt_rtn_df = convert_to_dataframe(parse_api_data(model_data.get("portf_mkt_rtn", [])), False)
                print('model scaled_weight')
                model_scaled_weight_df = convert_to_weight_dataframe(parse_api_data(model_data.get("scaled_weight_df", [])))
                print('model stats_parsed')
                model_stats_parsed_df = convert_to_stats_dataframe(parse_api_data(model_data.get("stats_df", [])), False, False)

                view_top300.render(benchmark_portf_rtn_df, benchmark_stats_parsed_df, model_portf_rtn_df, model_portf_mkt_rtn_df, model_scaled_weight_df, model_stats_parsed_df)
                st.divider()
                view_sentiment.render(sentiment_df, last_day, 1)
                st.divider()
                view_backtest.render(cum_df, vol_df, stats_parsed_df)
            else: #full
                print('portf_rtn')
                benchmark_portf_rtn_df = convert_to_dataframe(parse_api_data(benchmark_data.get("portf_rtn_comb", [])), False)
                print(benchmark_portf_rtn_df)
                print('port_mkt_rtn')
                benchmark_portf_mkt_rtn_df = convert_to_dataframe(parse_api_data(benchmark_data.get("portf_mkt_rtn_comb", [])), False)
                # print('scaled_weight')
                # benchmark_scaled_weight_df = convert_to_weight_dataframe(parse_api_data(benchmark_data.get("scaled_weight_df_comb", [])))
                print('stats_parsed')
                benchmark_stats_parsed_df = convert_to_stats_dataframe(parse_api_data(benchmark_data.get("stats_df_comb", [])), False, True)

                print('model portf_rtn')
                model_portf_rtn_df = convert_to_dataframe(parse_api_data(model_data.get("portf_rtn", [])), False)
                print('model portf_mkt_rtn')
                model_portf_mkt_rtn_df = convert_to_dataframe(parse_api_data(model_data.get("portf_mkt_rtn", [])), False)
                print('model scaled_weight')
                model_scaled_weight_df = convert_to_weight_dataframe(parse_api_data(model_data.get("scaled_weight_df", [])))
                print('model stats_parsed')
                model_stats_parsed_df = convert_to_stats_dataframe(parse_api_data(model_data.get("stats_df", [])), False, False)

                view_full.render(benchmark_portf_rtn_df, benchmark_portf_mkt_rtn_df, benchmark_stats_parsed_df, model_portf_rtn_df, model_portf_mkt_rtn_df, model_scaled_weight_df, model_stats_parsed_df)
                st.divider()
                view_sentiment.render(sentiment_df, last_day, 0)
                st.divider()
                view_backtest.render(cum_df, vol_df, stats_parsed_df)
        else:
            st.error("Could not fetch data for the selected portfolio configuration.")
else:
    view_backtest.render(cum_df, vol_df, stats_parsed_df)


# Footer
# st.markdown(f"""
#     <div class="footer">
#         &copy; 2025 AI-Powered Portfolio Optimizer
#     </div>
# """, unsafe_allow_html=True)

st.markdown(f"""    
    <div class="footer">
        &copy; 2025 AI-Powered Portfolio Optimizer
        <br>
        Developed by <strong>Deep Learning Sentiment Portfolio Capstone Team</strong> - UC Berkeley, MIDS School of Information
        <br>
        <a href="https://your-website.com" target="_blank">Visit Project Website</a> | 
        <a href="https://github.com/your-repo" target="_blank">GitHub</a>
        
    </div>
""", unsafe_allow_html=True)

# <a href="mailto:teamemail@example.com">Contact Us</a> |
# <span>Data powered by XYZ API</span>

