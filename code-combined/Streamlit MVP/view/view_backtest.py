import streamlit as st
from charts import ChartGenerator


def render(cum_df, vol_df, stats_parsed_df):
    chart_generator = ChartGenerator()

    st.header("Backtest on Full S&P500 Universe")

    st.subheader("Backtest Performance")
    if cum_df is not None and vol_df is not None:
        chart_generator.generate_backtest_chart(cum_df, vol_df)

    st.subheader("Portfolio Statistics")
    if stats_parsed_df is not None:
        chart_generator.generate_stats_table(stats_parsed_df, None)
    
