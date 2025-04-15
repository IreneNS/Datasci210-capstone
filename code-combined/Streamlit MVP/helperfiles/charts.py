from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import streamlit as st
import pandas as pd  # Ensure this is imported for DataFrame handling
from theme import *

__all__ = ["ChartGenerator"]

class ChartGenerator:
    def __init__(self) -> None:
        """Initialize the ChartGenerator class."""
        pass


    # Pie Chart for Portfolio Weights
    def generate_treemap(self, scaled_weight_df) -> None:
        """
        Generate a treemap pie chart of portfolio weights using Plotly.
        
        Args:
        - scaled_weight_df (pd.DataFrame): DataFrame containing stock symbols and their corresponding weights.
        
        Displays:
        - A treemap visualizing the weight distribution across the portfolio.
        """
        if not scaled_weight_df.empty:
            # Ensure the 'Identifier' and 'Weight' columns are present and valid
            if 'Stock Symbol' in scaled_weight_df.columns and 'Weight' in scaled_weight_df.columns:
                # Check if the Weight column has valid numerical data
                if scaled_weight_df['Weight'].isna().any():
                    st.error("The 'Weight' column contains NaN values.")
                else:
                    scaled_weight_df = scaled_weight_df.iloc[1:]

                    colorscale = [
                        [0, california_gold],
                        [0.5, white],
                        [1, berkeley_blue]
                    ]

                    fig_tree = px.treemap(
                        scaled_weight_df,
                        path=[px.Constant("Portfolio"), 'Stock Symbol'],
                        values='Weight',
                        color='Weight',
                        color_continuous_scale=colorscale
                    )

                    fig_tree.update_traces(
                        hovertemplate='<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>'
                    )

                    fig_tree.update_layout(
                        height=600,
                        margin=dict(t=50, l=25, r=25, b=25),
                        coloraxis_colorbar=dict(
                            title="Weight",
                            tickformat=".2f"
                        )
                    )

                    st.plotly_chart(fig_tree, use_container_width=True)
            else:
                st.error("The DataFrame is missing 'Stock Symbol' or 'Weight' columns.")
        else:
            st.error("No weight data available.")

    # Weights Table
    def generate_weights_table(self, model_df) -> None:
        """
        Generate and display a sorted table of deep learning portfolio weights.
        
        Args:
        - model_df (pd.DataFrame): DataFrame containing stock symbols and model-generated weights.
        
        Displays:
        - A sorted Streamlit table of stocks and their associated portfolio weights (in percent).
        """
        if not model_df.empty:
            # Fill missing weights with 0
            model_df["Weight_Model"] = model_df["Weight"].fillna(0)

            # Format weights as percentages
            model_df["Deep Learning Weight(%)"] = (model_df["Weight_Model"] * 100).round(2)

            # Prepare display dataframe
            display_df = model_df[["Stock Symbol", "Deep Learning Weight(%)"]]

            # Sort by Deep Learning Weight (%) descending
            display_df = display_df.sort_values(by="Deep Learning Weight(%)", ascending=False)
            display_df = display_df.iloc[1:]
            display_df.index = range(1, len(display_df) + 1)

            st.dataframe(display_df)
        else:
            st.error("Weights data unavailable or invalid.")


    # Performance Comparison Charts
    def generate_performance_comparisons_chart(self, portf_mkt_rtn_bm, portf_mkt_rtn_dl, last_win_only=False) -> None:
        """
        Generate a dual-subplot chart showing cumulative return and rolling volatility
        for the benchmark, deep learning model, and unscaled market.
        
        Args:
        - portf_mkt_rtn_bm (pd.DataFrame): Benchmark return time series data.
        - portf_mkt_rtn_dl (pd.DataFrame): Deep learning return time series data.
        - last_win_only (bool): If True, use a 60-day window for rolling volatility. If False, use 252-day (annual).
        
        Displays:
        - A Plotly figure with two side-by-side subplots comparing return and volatility.
        """
        if not portf_mkt_rtn_bm.empty and not portf_mkt_rtn_dl.empty:
            # Prepare data for plotting
            portf_mkt_rtn_bm = portf_mkt_rtn_bm.copy()
            portf_mkt_rtn_dl = portf_mkt_rtn_dl.copy()

            # Benchmark data
            portf_mkt_rtn_bm['label'] = 'Benchmark'
            portf_mkt_rtn_bm['Date'] = pd.to_datetime(portf_mkt_rtn_bm['Date'])
            portf_mkt_rtn_bm['Benchmark Return'] = portf_mkt_rtn_bm['Benchmark Return'].fillna(0)
            portf_mkt_rtn_bm['Cumulative Return'] = portf_mkt_rtn_bm['Benchmark Return'].cumsum() 
            portf_mkt_rtn_bm['Rolling Vol'] = portf_mkt_rtn_bm['Benchmark Return'].rolling(
                60 if last_win_only else 252).std() * np.sqrt(252)

            # Deep learning data
            portf_mkt_rtn_dl['label'] = 'Deep Learning'
            portf_mkt_rtn_dl['Date'] = pd.to_datetime(portf_mkt_rtn_dl['Date'])
            portf_mkt_rtn_dl['Deep Learning Return'] = portf_mkt_rtn_dl['Deep Learning Return'].fillna(0)
            portf_mkt_rtn_dl['Cumulative Return'] = portf_mkt_rtn_dl['Deep Learning Return'].cumsum()
            portf_mkt_rtn_dl['Rolling Vol'] = portf_mkt_rtn_dl['Deep Learning Return'].rolling(
                60 if last_win_only else 252).std() * np.sqrt(252)

            # Unscaled market data
            unscaled_market = portf_mkt_rtn_dl[['Date', 'Unscaled Market Return']].copy()

            unscaled_market['label'] = 'Unscaled Market'
            unscaled_market['Unscaled Market Return'] = unscaled_market['Unscaled Market Return'].fillna(0)
            unscaled_market['Cumulative Return'] = unscaled_market['Unscaled Market Return'].cumsum()
            unscaled_market['Rolling Vol'] = unscaled_market['Unscaled Market Return'].rolling(
                60 if last_win_only else 252).std() * np.sqrt(252)

            # In case of any mismatch dates
            portf_mkt_rtn_bm['Date'] = portf_mkt_rtn_dl['Date']

            # Combine all data into one DataFrame
            combined_df = pd.concat([
                portf_mkt_rtn_bm[['Date', 'Cumulative Return', 'Rolling Vol', 'label']],
                portf_mkt_rtn_dl[['Date', 'Cumulative Return', 'Rolling Vol', 'label']],
                unscaled_market[['Date', 'Cumulative Return', 'Rolling Vol', 'label']]
            ])

            fig = make_subplots(
                rows=1, cols=2, shared_xaxes=True, subplot_titles=("Cumulative Return", "Rolling Annual Volatility")
            )

            color_map = {
                "Deep Learning": berkeley_blue,
                "Benchmark": california_gold,
                "Unscaled Market": grey
            }

            # Plot cumulative return for each label
            for label in combined_df['label'].unique():
                df_filtered = combined_df[combined_df['label'] == label]
                fig.add_trace(go.Scatter(
                    x=df_filtered['Date'], y=df_filtered['Cumulative Return'],
                    mode='lines', name=label,
                    line=dict(color=color_map[label])
                ), row=1, col=1)

            # Plot rolling volatility for each label
            for label in combined_df['label'].unique():
                df_filtered = combined_df[combined_df['label'] == label]
                fig.add_trace(go.Scatter(
                    x=df_filtered['Date'], y=df_filtered['Rolling Vol'],
                    mode='lines', name=label,
                    line=dict(color=color_map[label]),
                    showlegend=False  # Turn off legend for rolling volatility plot
                ), row=1, col=2)

            fig.update_layout(
                title_text="Proposed Portfolio (Daily vs. Unscaled) S&P500 Cumulative Return & Rolling Volatility Comparison",
                height=600, width=1000,
                showlegend=True
            )
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Cumulative Return (×)", row=1, col=1)
            fig.update_yaxes(title_text="Annualized Volatility (×)", row=1, col=2)

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("No performance data available.")


    # Static Stats Table
    def generate_stats_table(self, bm_df, model_df) -> None:
        """
        Generate and display a comparative table of return statistics for different strategies.
        
        Args:
        - bm_df (pd.DataFrame): Benchmark statistics DataFrame.
        - model_df (pd.DataFrame): Model statistics DataFrame.
        
        Displays:
        - A Streamlit table combining both sets of stats side-by-side.
        """
        if bm_df is not None and model_df is not None:
            common_columns = bm_df.columns.intersection(model_df.columns)
            bm_df = bm_df.drop(columns=common_columns)
            combined_df = pd.concat([bm_df, model_df], axis=1)
            
            # Reorder columns
            desired_order = ['Daily Deep Learning with Sentiment Return', 'Daily Benchmark Return', 'Unscaled Market Return']
            existing_desired = [col for col in desired_order if col in combined_df.columns]
            remaining_cols = [col for col in combined_df.columns if col not in existing_desired]
            combined_df = combined_df[existing_desired + remaining_cols]

            if 'Date' in combined_df.columns:
                combined_df = combined_df.drop(columns=['Date'])

            st.dataframe(combined_df)
        elif bm_df is not None and model_df is None:
             st.dataframe(bm_df)
        else:
            st.error("Error parsing or displaying portfolio statistics.")


    # Sentiment Breakdown Chart
    def generate_sentiment_chart(self, last_day, loc) -> None:
        """
        Visualize the sentiment distribution (positive, neutral, negative) for the latest day using a stacked bar chart.
        
        Args:
        - last_day (pd.DataFrame): Single-row DataFrame containing sentiment probabilities.
        - loc (str): Placeholder for future location-based customizations (currently unused).
        
        Displays:
        - A horizontal stacked bar chart of sentiment probabilities.
        """
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
                go.Bar(y=["Sentiment"], x=[finbert_sentiment_probs["Neutral"]], name="Neutral", marker_color=grey, orientation="h"),
                go.Bar(y=["Sentiment"], x=[finbert_sentiment_probs["Positive"]], name="Positive", marker_color=california_gold, orientation="h"),
            ]
        )
        fig_sentiment.update_layout(barmode="stack", xaxis=dict(title="Probability"), height=180, margin=dict(l=20, r=20, t=40, b=20), showlegend=True)
        st.plotly_chart(fig_sentiment, use_container_width=True)


    # Next Day Return Probability Chart
    def generate_next_day_return_chart(self, last_day) -> None:
        """
        Visualize the predicted probability of a positive return the next day.
        
        Args:
        - last_day (pd.DataFrame): Single-row DataFrame containing the predicted next-day return value.
        
        Displays:
        - A progress bar in Streamlit showing the predicted probability.
        """
        next_day_return_prob = last_day['predicted_continuous_return'].values[0]

        st.markdown(f"""
                    <style>
                    .stProgress > div > div > div {{
                        height: 20px !important;
                        border-radius: 5px !important;
                    }}
                    .stProgress > div > div > div > div {{
                        background-color: {berkeley_blue} !important;
                        height: 20px !important;
                        border-radius: 5px !important;
                    }}
                    </style>
                """, unsafe_allow_html=True)

        st.progress(next_day_return_prob, text=f"📈 Probability of Positive Next-Day Return: {next_day_return_prob * 100:.1f}%")


    # Backtest Chart
    def generate_backtest_chart(self, cum_df, vol_df) -> None:
        """
        Generate a dual-subplot backtest chart showing cumulative return and volatility across time.
        
        Args:
        - cum_df (pd.DataFrame): DataFrame containing cumulative returns with label and datetime columns.
        - vol_df (pd.DataFrame): DataFrame containing rolling volatility with label and datetime columns.
        
        Displays:
        - A Plotly figure comparing backtested strategies and their volatilities.
        """
        fig = make_subplots(rows=1, cols=2, shared_xaxes=True, subplot_titles=("Cumulative Return", "Rolling Annual Volatility"))

        colormap = {
            'Daily DL with Sentiment': berkeley_blue,
            'Daily DL': berkeley_blue_light,
            'Daily Benchmark': california_gold,
            'Unscaled Market': grey,
            'Train vs. Test Division': red,
        }
        
        # Add Cumulative Return lines to the first subplot (left)
        for label in cum_df['label'].unique():
            filtered_df = cum_df[cum_df['label'] == label]
            
            # Assign a color to the label from the color_map
            line = dict(color=colormap.get(label, '#1f77b4'))  # Default color if not in the map

            if "Division" in label:
                line['dash'] = 'dash'  # Apply dashed line for the "Train vs Test Division"

            fig.add_trace(
                go.Scatter(x=filtered_df['datetime'], y=filtered_df['Y'], mode='lines', name=label,
                           line=line),
                row=1, col=1
            )

        # Add Volume Data lines to the second subplot (right)
        for label in vol_df['label'].unique():
            filtered_df = vol_df[vol_df['label'] == label]
            
            # Use the same color for the label as in the cumulative plot
            line = dict(color=colormap.get(label, '#1f77b4'))  # Default color if not in the map

            if "Division" in label:
                line['dash'] = 'dash'  # Apply dashed line for the "Train vs Test Division"

            fig.add_trace(
                go.Scatter(x=filtered_df['datetime'], y=filtered_df['Y'], mode='lines', name=label,
                           line=line, showlegend=False),
                row=1, col=2
            )

        fig.update_layout(
            title_text="Proposed Portfolio (Daily vs. Unscaled) S&P500 Cumulative Return & Rolling Volatility Comparison",
            height=600, width=1000,
            showlegend=True,
            xaxis_title="Date",
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Return (×)", row=1, col=1)
        fig.update_yaxes(title_text="Annualized Volatility (×)", row=1, col=2)

        st.plotly_chart(fig)

