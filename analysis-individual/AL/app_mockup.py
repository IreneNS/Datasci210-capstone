import streamlit as st
import random
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Portfolio Optimizer", page_icon="📈", layout="wide")

st.title("📊 AI-Powered Portfolio Optimizer")
st.write("Enter your stock universe, adjust risk preference, and receive an optimized asset allocation.")

with st.form("portfolio_form"):
    tickers = st.text_input("Enter stock tickers (comma-separated)", placeholder="e.g., SPY, QQQ, DIA, IWM")
    risk_preference = st.slider("Adjust risk preference (volatility tolerance)", 0, 100, 50)
    submit_button = st.form_submit_button("Run Portfolio Optimization 🚀")

# Process input when the form is submitted
if submit_button:
    #clean ticker input (comma-separated)
    tickers_list = [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]
    
    if not tickers_list:
        st.error("Please enter at least one valid stock ticker.")
    else:
        # Simulated deep learning output: Generate random weights for each stock
        random_weights = [random.uniform(5, 50) for _ in tickers_list]
        total_weight = sum(random_weights)
        normalized_weights = [round((w / total_weight) * 100, 2) for w in random_weights]

        # Display df
        df = pd.DataFrame({"Ticker": tickers_list, "Weight (%)": normalized_weights})

        col1, col2 = st.columns([1, 1])  # Two equal-width columns for displaying output side by side

        with col1:
            st.subheader("📌 Recommended Asset Allocation")
            st.dataframe(df.style.format({"Weight (%)": "{:.2f}%"}))

        with col2:
            st.subheader("📈 Asset Distribution")
            fig = px.pie(df, names="Ticker", values="Weight (%)", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
