import streamlit as st
import random
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Portfolio Optimizer", page_icon="📈", layout="wide")

st.title("📊 AI-Powered Portfolio Optimizer")
st.write("Enter your stock universe, specify initial weights, adjust risk preference, and receive an optimized asset allocation.")

# Initialize session state for tickers and weights
if "tickers" not in st.session_state:
    st.session_state["tickers"] = []
if "weights" not in st.session_state:
    st.session_state["weights"] = {}

# User inputs tickers
st.subheader("Enter Stock Tickers (Comma-Separated)")
tickers_input = st.text_input("", placeholder="e.g., AAPL, TSLA, AMZN")

# Process tickers and reset removed ones
if tickers_input:
    new_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    # Reset session state weights if tickers have changed
    if set(new_tickers) != set(st.session_state["tickers"]):
        st.session_state["weights"] = {t: st.session_state["weights"].get(t, 0) for t in new_tickers}
    
    st.session_state["tickers"] = new_tickers

# Show weights (only if tickers are entered)
if st.session_state["tickers"]:
    # st.subheader("Specify Initial Weights (%)")
    
    # columns = st.columns(len(st.session_state["tickers"]))
    # for idx, ticker in enumerate(st.session_state["tickers"]):
    #     with columns[idx % len(columns)]:  # Distribute inputs evenly
    #         st.session_state["weights"][ticker] = st.number_input(
    #             f"{ticker}", 
    #             min_value=0, 
    #             max_value=100, 
    #             value=st.session_state["weights"][ticker], 
    #             step=1,
    #             key=f"weight_{ticker}"
    #         )

    # # Only show total weight when tickers exist
    # total_weight = sum(st.session_state["weights"].values())

    # if total_weight == 100:
    #     st.success(f"✅ Total Weight: {total_weight}% (Balanced)")
    # else:
    #     st.error(f"⚠️ Total Weight: {total_weight}% (Must be 100%)")

# Risk preference slider
risk_preference = st.slider("Adjust risk preference (volatility tolerance)", 0, 100, 50)

# Submit button
if st.button("Run Portfolio Optimization 🚀") and st.session_state["tickers"]:
    df = pd.DataFrame({
        "Ticker": st.session_state["tickers"], 
        "Initial Weight (%)": [st.session_state["weights"][t] for t in st.session_state["tickers"]]
    })

    # Validate weights
    if df["Initial Weight (%)"].sum() != 100:
        st.error("Total initial weights must sum to 100%. Adjust your allocations.")
    else:
        # Simulated optimization (apply random variation) and calc change in weights
        df["Optimized Weight (%)"] = [round(w * (1 + random.uniform(-0.1, 0.1)), 2) for w in df["Initial Weight (%)"]]
        df["Change in Weight (%)"] = df["Optimized Weight (%)"] - df["Initial Weight (%)"]

        # Generate Random Market Sentiment Score
        market_sentiment = round(random.uniform(40, 70), 2)

        col1, col2 = st.columns([2, 1], gap="small")

        with col1:
            st.subheader("📌 Recommended Asset Allocation", divider=True)

            # Remove index and format DataFrame
            styled_df = df.style.format(
                {"Initial Weight (%)": "{:.2f}%", 
                 "Optimized Weight (%)": "{:.2f}%", 
                 "Change in Weight (%)": "{:.2f}%"}
            ).hide(axis="index")

            st.dataframe(styled_df, use_container_width=True)

        with col2:
            st.subheader("📊 Asset Reallocation", divider=True)

            fig = px.pie(df, names="Ticker", values="Optimized Weight (%)", hole=0.4)

            # 🎯 Apply custom formatting to pie chart
            fig.update_traces(
                textinfo="percent+label",   # Show percentages with labels
                textfont_size=16,           # Bigger font size
                marker=dict(line=dict(color="#000000", width=1))  # Black outline for better visibility
            )

            fig.update_layout(
                showlegend=False, height=270
            )

            st.plotly_chart(fig, use_container_width=True)

        # Display market sentiment slider
        st.markdown("<br>", unsafe_allow_html=True)  # 🔥 Add only a tiny margin
        st.subheader("📈 Market Sentiment Score", divider=True)

        st.markdown(
            f"<h3 style='text-align: center; padding-top: 0px;'>📉 {market_sentiment}% Probability Market Goes Up Tomorrow</h3>",
            unsafe_allow_html=True,
        )

        st.progress(market_sentiment / 100)
