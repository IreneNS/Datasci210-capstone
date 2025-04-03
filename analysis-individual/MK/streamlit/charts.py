from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import streamlit as st
import pandas as pd  # Ensure this is imported for DataFrame handling
from theme import berkeley_blue, california_gold  # Import colors

class ChartGenerator:
    def __init__(self):
        pass

    # Pie Chart for Portfolio Weights
    def generate_pie_chart(self, scaled_weight_df):
        if not scaled_weight_df.empty:
            # Debug: Print the DataFrame
            st.write(scaled_weight_df)
            st.write('sum:', sum(scaled_weight_df['Weight']))

            # Ensure the 'Identifier' and 'Weight' columns are present and valid
            if 'Identifier' in scaled_weight_df.columns and 'Weight' in scaled_weight_df.columns:
                # Check if the Weight column has valid numerical data
                if scaled_weight_df['Weight'].isna().any():
                    st.error("The 'Weight' column contains NaN values.")
                else:
                    # Create the pie chart
                    fig_pie = px.pie(scaled_weight_df, values="Weight", names="Identifier", title="Optimized Portfolio Asset Allocation")
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.error("The DataFrame is missing 'Identifier' or 'Weight' columns.")
        else:
            st.error("No weight data available.")

    # Cumulative Returns Comparison
    def generate_cumulative_returns_chart(self, portf_mkt_rtn_df):
        if not portf_mkt_rtn_df.empty:
            portf_mkt_rtn_df['Cumulative Benchmark Return'] = (1 + portf_mkt_rtn_df['Benchmark Return']).cumprod()
            fig_cumulative = go.Figure()
            fig_cumulative.add_trace(go.Scatter(x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Cumulative Benchmark Return'],
                                                mode='lines', name='Cumulative Benchmark Return',
                                                line=dict(color=berkeley_blue, width=2)))
            fig_cumulative.add_trace(go.Scatter(x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Cumulative Benchmark Return'],
                                                mode='lines', name='Cumulative Unscaled Market Return',
                                                line=dict(color=california_gold, width=2)))
            fig_cumulative.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return (%)", margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig_cumulative, use_container_width=True)
        else:
            st.error("No cumulative return data available.")

    # Rolling Annual Volatility Comparison
    def generate_volatility_chart(self, portf_mkt_rtn_df):
        if not portf_mkt_rtn_df.empty:
            portf_mkt_rtn_df['Rolling Vol Benchmark'] = portf_mkt_rtn_df['Benchmark Return'].rolling(window=60).std() * np.sqrt(252)
            portf_mkt_rtn_df['Rolling Vol Unscaled Market'] = portf_mkt_rtn_df['Unscaled Market Return'].rolling(window=60).std() * np.sqrt(252)
            fig_rolling_vol = go.Figure()
            fig_rolling_vol.add_trace(go.Scatter(x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Rolling Vol Benchmark'],
                                                 mode='lines', name='Rolling Volatility (Benchmark)', line=dict(color=berkeley_blue, width=2)))
            fig_rolling_vol.add_trace(go.Scatter(x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Rolling Vol Unscaled Market'],
                                                 mode='lines', name='Rolling Volatility (Unscaled Market)', line=dict(color=california_gold, width=2)))
            fig_rolling_vol.update_layout(xaxis_title="Date", yaxis_title="Annualized Volatility (%)", margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig_rolling_vol, use_container_width=True)
        else:
            st.error("No rolling volatility data available.")

    # Weights Table
    def generate_weights_table(self, scaled_weight_df):
        """Generate the portfolio weights table."""
        st.subheader("Portfolio Weights Table")
        weights_placeholder = st.empty()

        if scaled_weight_df is not None and isinstance(scaled_weight_df, pd.DataFrame) and not scaled_weight_df.empty:
            weights_placeholder.dataframe(scaled_weight_df)
        else:
            weights_placeholder.markdown("Weights data unavailable or invalid.")  # Placeholder message

    # Static Stats Table
    def generate_stats_table(self, stats_parsed_df, pkl):
        if stats_parsed_df is not None:
            if pkl:
                new_col_names = {"Daily-DL-Max-Sharpe": "Daily DL Max Sharpe",
                                 "Daily-DL-Max-Sharpe w senti": "Daily DL Max Sharpe with Sentiment",
                                 "Daily-Benchmark": "Daily Benchmark"}

                new_idx_names = {"avg_rtn_ann": "Average Annual Return",
                                 "vol_ann": "Annual Volatility",
                                 "sharpe_ann": "Annual Max Sharpe",
                                 "max_drawdown": "Max Drawdown"}

                stats_parsed_df = stats_parsed_df.rename(columns=new_col_names, index = new_idx_names)
                st.dataframe(stats_parsed_df)
            else:
                st.dataframe(stats_parsed_df)
        else:
            st.error("Error parsing or displaying portfolio statistics.")

    # Sentiment Breakdown Chart
    def generate_sentiment_chart(self, last_day):
        finbert_sentiment_probs = {
            "Negative": last_day['Negative_Prob'].values[0],
            "Neutral": last_day['Neutral_Prob'].values[0],
            "Positive": last_day['Positive_Prob'].values[0],
        }
        total = sum(finbert_sentiment_probs.values())
        finbert_sentiment_probs = {k: v / total for k, v in finbert_sentiment_probs.items()}
        fig_sentiment = go.Figure(
            data=[
                go.Bar(y=["Sentiment"], x=[finbert_sentiment_probs["Negative"]], name="Negative", marker_color=berkeley_blue, orientation="h"),
                go.Bar(y=["Sentiment"], x=[finbert_sentiment_probs["Neutral"]], name="Neutral", marker_color="rgba(45, 85, 115, 0.5)", orientation="h"),
                go.Bar(y=["Sentiment"], x=[finbert_sentiment_probs["Positive"]], name="Positive", marker_color=california_gold, orientation="h"),
            ]
        )
        fig_sentiment.update_layout(barmode="stack", xaxis=dict(title="Probability"), height=180, margin=dict(l=20, r=20, t=40, b=20), showlegend=True)
        st.plotly_chart(fig_sentiment, use_container_width=True)

    # Static Backtest Performance Chart
    # def generate_backtest_chart(self, portf_rtn_df, portf_mkt_rtn_df):
    #     fig_backtest = go.Figure()
    #     fig_backtest.add_trace(go.Scatter(x=portf_rtn_df['Date'], y=portf_rtn_df['Return'], mode='lines', name='ML Portfolio Return', line=dict(color="#2CA02C", width=2)))
    #     fig_backtest.add_trace(go.Scatter(x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Benchmark Return'], mode='lines', name='Benchmark Return', line=dict(color=berkeley_blue, dash='dash', width=2)))
    #     fig_backtest.update_layout(xaxis_title="Date", yaxis_title="Return (%)", margin=dict(l=40, r=40, t=40, b=40))
    #     st.plotly_chart(fig_backtest, use_container_width=True)

    def generate_backtest_chart(self, cum_df, vol_df):
        fig = make_subplots(rows=1, cols=2, shared_xaxes=True, subplot_titles=("Cumulative Return", "Rolling Annual Volatility"))

        # Define a color map to ensure the same label gets the same color in both subplots
        color_map = {}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_idx = 0

        

        # Add Cumulative Return lines to the first subplot (left)
        for label in cum_df['label'].unique():
            filtered_df = cum_df[cum_df['label'] == label]
            # Assign a color to the label if it hasn't been assigned one already
            if label not in color_map:
                color_map[label] = colors[color_idx % len(colors)]  # Ensure it wraps around the color list
                color_idx += 1
            
            if "Division" in label:
                line = dict(color = california_gold, dash = 'dash')
            else: 
                line = dict(color=color_map[label])

            fig.add_trace(
                go.Scatter(x=filtered_df['datetime'], y=filtered_df['Y'], mode='lines', name=label,
                           line=line),
                row=1, col=1
            )

        # Add Volume Data lines to the second subplot (right)
        for label in vol_df['label'].unique():
            filtered_df = vol_df[vol_df['label'] == label]
            # Use the same color for the label as in the cumulative plot
            if label not in color_map:
                color_map[label] = colors[color_idx % len(colors)]  # Ensure it wraps around the color list
                color_idx += 1
            
            if "Division" in label:
                line = dict(color = california_gold, dash = 'dash')
            else: 
                line = dict(color=color_map[label])

            fig.add_trace(
                go.Scatter(x=filtered_df['datetime'], y=filtered_df['Y'], mode='lines', name=label,
                           line=line, showlegend=False),
                row=1, col=2
            )



        # # Add Vertical Line at Train vs. Test Division Date
        # vline_date_cum = cum_df.loc[cum_df["label"] == "Train vs. Test Division", "datetime"]
        # vline_date_vol = vol_df.loc[vol_df["label"] == "Train vs. Test Division", "datetime"]

        # # Ensure we are passing a single datetime value in native Python format
        # if not vline_date_cum.empty and not vline_date_vol.empty:
        #     vline_date_cum = vline_date_cum.iloc[0].to_pydatetime()  # Convert to native datetime
        #     vline_date_vol = vline_date_vol.iloc[0].to_pydatetime()  

        #     fig.add_vline(
        #         x=vline_date_cum, line=dict(color="black", width=2, dash="dash"),
        #         row=1, col=1, xref="x1",
        #         annotation_text="Train vs. Test Division", annotation_position="top"
        #     )

        #     fig.add_vline(
        #         x=vline_date_vol, line=dict(color="black", width=2, dash="dash"),
        #         row=1, col=2, xref="x2",
        #         annotation_text="Train vs. Test Division", annotation_position="top"
        #     )

        # Update layout for the figure
        fig.update_layout(
            title_text="Proposed Portfolio (Daily vs. Unscaled) S&P500 Cumulative Return & Rolling Volatility Comparison",
            height=600, width=1000,
            showlegend=True,
            xaxis_title="Date",
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)

        st.plotly_chart(fig)




