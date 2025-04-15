import streamlit as st
from charts import ChartGenerator

def render(sentiment_df, last_day, loc):
    chart_generator = ChartGenerator()

    st.subheader("Sentiment Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###### FinBERT Sentiment Breakdown")
        chart_generator.generate_sentiment_chart(last_day, loc)
    with col2:
        st.markdown("###### Predicting Next Day’s Return")
        chart_generator.generate_next_day_return_chart(last_day)

