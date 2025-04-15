import streamlit as st
import pandas as pd

from view import view_top300, view_full, view_sentiment, view_backtest
from styles import apply_base_styles
from helperfiles import *  # imports everything exposed by your __init__.py



# Initialize the chart generator
chart_generator = ChartGenerator()

st.set_page_config(page_title="AI Portfolio Optimizer", page_icon="📈", layout="wide")

# Apply styles and render header
apply_base_styles()  # Apply the base styles from styles.py
render_header()

# Sidebar: User Inputs
selected_stock_count, risk_tolerance, submit_button = render_sidebar_form()

cum_df, vol_df = fetch_s3_data("backtest")
stats_parsed_df = convert_to_stats_dataframe(fetch_s3_data("stats"), True, False)

full = False
if submit_button:
    ticker_list = []
    if '300' in selected_stock_count:
        ticker_list = ['top_300_ticker_l']
    else:  # full
        ticker_list = []
        full = True

    fetching_message = st.empty()
    fetching_message.subheader("Fetching Optimized Portfolio Data... ⏳")

    benchmark_data = fetch_benchmark_data(ticker_list, "D", "max_sharpe", risk_tolerance, False)
    model_data = fetch_model_data(ticker_list, "D", "max_sharpe", risk_tolerance, False)
    sentiment_df = pd.read_parquet('Sentiment_Predictions_w_Imputed_Values.parquet')  # TODO
    last_day = sentiment_df.iloc[-1:, :]

    # Fetching portfolio data
    if selected_stock_count:
        if model_data:  # benchmark_data and model_data:
            # Clear the "Fetching Optimized Portfolio Data..." message
            fetching_message.empty()

            if not full:  # only top 300
                print('portf_rtn')
                benchmark_portf_rtn_df = convert_to_dataframe(parse_api_data(benchmark_data.get("portf_rtn_comb", [])), True)
                benchmark_stats_parsed_df = create_stats_dataframe(benchmark_portf_rtn_df)

                model_portf_mkt_rtn_df = convert_to_dataframe(parse_api_data(model_data.get("portf_mkt_rtn", [])), False)
                model_scaled_weight_df = convert_to_weight_dataframe(parse_api_data(model_data.get("scaled_weight_df", [])))
                model_stats_parsed_df = convert_to_stats_dataframe(parse_api_data(model_data.get("stats_df", [])), False, False)

                # Render views for Top 300 stocks
                view_top300.render(benchmark_portf_rtn_df, benchmark_stats_parsed_df, model_portf_mkt_rtn_df, model_scaled_weight_df, model_stats_parsed_df)
                st.divider()
                view_sentiment.render(sentiment_df, last_day, 1)
                st.divider()
                view_backtest.render(cum_df, vol_df, stats_parsed_df)
            else:  # full
                benchmark_portf_mkt_rtn_df = convert_to_dataframe(parse_api_data(benchmark_data.get("portf_mkt_rtn_comb", [])), False)
                benchmark_stats_parsed_df = convert_to_stats_dataframe(parse_api_data(benchmark_data.get("stats_df_comb", [])), False, True)

                model_portf_mkt_rtn_df = convert_to_dataframe(parse_api_data(model_data.get("portf_mkt_rtn", [])), False)
                model_scaled_weight_df = convert_to_weight_dataframe(parse_api_data(model_data.get("scaled_weight_df", [])))
                model_stats_parsed_df = convert_to_stats_dataframe(parse_api_data(model_data.get("stats_df", [])), False, False)

                # Render views for Full S&P500 universe
                view_full.render(benchmark_portf_mkt_rtn_df, benchmark_stats_parsed_df, model_portf_mkt_rtn_df, model_scaled_weight_df, model_stats_parsed_df)
                st.divider()
                view_sentiment.render(sentiment_df, last_day, 0)
                st.divider()
                view_backtest.render(cum_df, vol_df, stats_parsed_df)
        else:
            st.error("Could not fetch data for the selected portfolio configuration.")
else:
    view_backtest.render(cum_df, vol_df, stats_parsed_df)

# Footer
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
