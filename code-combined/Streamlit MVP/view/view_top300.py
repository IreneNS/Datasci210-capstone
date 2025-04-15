import streamlit as st
from charts import ChartGenerator


def render(benchmark_portf_rtn_df, benchmark_stats_parsed_df, model_portf_mkt_rtn_df, model_scaled_weight_df, model_stats_parsed_df):
    chart_generator = ChartGenerator()

    st.subheader("Optimized Portfolio Weights Treemap")
    chart_generator.generate_treemap(model_scaled_weight_df)

    with st.expander("View more details."):
        st.subheader("Portfolio Weights Table")
        chart_generator.generate_weights_table(model_scaled_weight_df)

    st.subheader("Performance Comparison")
    chart_generator.generate_performance_comparisons_chart(benchmark_portf_rtn_df, model_portf_mkt_rtn_df, False)

    st.subheader("Portfolio Statistics")
    chart_generator.generate_stats_table(benchmark_stats_parsed_df, model_stats_parsed_df)

    