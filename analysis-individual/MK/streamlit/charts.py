from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import streamlit as st
import pandas as pd  # Ensure this is imported for DataFrame handling
from theme import *

class ChartGenerator:
    def __init__(self):
        pass

    # Pie Chart for Portfolio Weights
    def generate_pie_chart(self, scaled_weight_df):
        if not scaled_weight_df.empty:
            # Debug: Print the DataFrame
            # st.write(scaled_weight_df)
            # st.write('sum:', sum(scaled_weight_df['Weight']))

            # Ensure the 'Identifier' and 'Weight' columns are present and valid
            if 'Stock Symbol' in scaled_weight_df.columns and 'Weight' in scaled_weight_df.columns:
                # Check if the Weight column has valid numerical data
                if scaled_weight_df['Weight'].isna().any():
                    st.error("The 'Weight' column contains NaN values.")
                else:
                    scaled_weight_df = scaled_weight_df.iloc[1:]

                    # # Create the pie chart
                    # fig_pie = px.pie(scaled_weight_df, values="Weight", names="Stock Symbol", title="Optimized Portfolio Asset Allocation")
                    # st.plotly_chart(fig_pie, use_container_width=True)

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
    def generate_weights_table(self, model_df):
        """Generate the portfolio weights table."""
        if not model_df.empty:
            # Fill missing weights with 0
            # print(model_df.columns)
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
    def generate_performance_comparisons_chart(self, portf_mkt_rtn_bm, portf_mkt_rtn_dl, last_win_only=False):
        """
        Generates performance comparison charts for cumulative return and rolling volatility.

        Args:
        - portf_mkt_rtn_bm: DataFrame containing benchmark return data
        - portf_mkt_rtn_dl: DataFrame containing deep learning return data
        - last_win_only: Boolean flag to calculate rolling volatility over 60 days or 252 days
        
        Returns:
        - A Plotly chart of cumulative return and rolling volatility comparisons.
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


            portf_mkt_rtn_bm['Date'] = portf_mkt_rtn_dl['Date']

            # Combine all data into one DataFrame
            combined_df = pd.concat([
                portf_mkt_rtn_bm[['Date', 'Cumulative Return', 'Rolling Vol', 'label']],
                portf_mkt_rtn_dl[['Date', 'Cumulative Return', 'Rolling Vol', 'label']],
                unscaled_market[['Date', 'Cumulative Return', 'Rolling Vol', 'label']]
            ])

            # Create the subplots
            fig = make_subplots(
                rows=1, cols=2, shared_xaxes=True, subplot_titles=("Cumulative Return", "Rolling Annual Volatility")
            )

            # Define a color map
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

            # Customize layout
            fig.update_layout(
                title_text="Proposed Portfolio (Daily vs. Unscaled) S&P500 Cumulative Return & Rolling Volatility Comparison",
                height=600, width=1000,
                showlegend=True
            )
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Cumulative Return (×)", row=1, col=1)
            fig.update_yaxes(title_text="Annualized Volatility (×)", row=1, col=2)


            # Display the chart
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("No performance data available.")

    # # Cumulative Returns Comparison
    # def generate_cumulative_returns_chart(self, portf_mkt_rtn_df, bm):
    #     if not portf_mkt_rtn_df.empty:
    #         fig_cumulative = go.Figure()
            
    #         if bm:
    #             portf_mkt_rtn_df['Cumulative Benchmark Return'] = (1 + portf_mkt_rtn_df['Benchmark Return']).cumprod()
    #             portf_mkt_rtn_df['Cumulative Market Return'] = (1 + portf_mkt_rtn_df['Unscaled Market Return']).cumprod()
                
    #             fig_cumulative.add_trace(go.Scatter(
    #                 x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Cumulative Benchmark Return'],
    #                 mode='lines', name='Cumulative Benchmark Return',
    #                 line=dict(color=berkeley_blue, width=2)
    #             ))

    #         else:
    #             portf_mkt_rtn_df['Cumulative Deep Learning Return'] = (1 + portf_mkt_rtn_df['Deep Learning Return']).cumprod()
    #             fig_cumulative.add_trace(go.Scatter(
    #                 x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Cumulative Deep Learning Return'],
    #                 mode='lines', name='Cumulative Deep Learning Return',
    #                 line=dict(color=berkeley_blue, width=2)
    #             ))

    #         # Common unscaled market trace
    #         portf_mkt_rtn_df['Cumulative Market Return'] = (1 + portf_mkt_rtn_df['Unscaled Market Return']).cumprod()
    #         fig_cumulative.add_trace(go.Scatter(
    #             x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Cumulative Market Return'],
    #             mode='lines', name='Cumulative Unscaled Market Return',
    #             line=dict(color=california_gold, width=2)
    #         ))

    #         # division_date = portf_mkt_rtn_df.loc[portf_mkt_rtn_df['label'] == "Train vs. Test Division", "Date"]
    #         # if not division_date.empty:
    #         #     fig_cumulative.add_vline(
    #         #         x=division_date.iloc[0],
    #         #         line=dict(color="black", width=2, dash="dash"),
    #         #         annotation_text="Train vs. Test Division",
    #         #         annotation_position="top right"
    #         #     )

    #         fig_cumulative.update_layout(
    #             # title="Cumulative Return Comparison",
    #             xaxis_title="Date", yaxis_title="Cumulative Return",
    #             margin=dict(l=40, r=40, t=40, b=40)
    #         )
    #         st.plotly_chart(fig_cumulative, use_container_width=True)
    #     else:
    #         st.error("No cumulative return data available.")


    # # Rolling Annual Volatility Comparison
    # def generate_volatility_chart(self, portf_mkt_rtn_df, bm, last_win_only=False):
    #     if not portf_mkt_rtn_df.empty:
    #         window_size = 60 if last_win_only else 252
    #         fig_rolling_vol = go.Figure()

    #         # Always compute unscaled market volatility
    #         portf_mkt_rtn_df['Rolling Vol Unscaled Market'] = (
    #             portf_mkt_rtn_df['Unscaled Market Return'].rolling(window=window_size).std() * np.sqrt(252)
    #         )

    #         if bm:
    #             portf_mkt_rtn_df['Rolling Vol Benchmark'] = (
    #                 portf_mkt_rtn_df['Benchmark Return'].rolling(window=window_size).std() * np.sqrt(252)
    #             )
    #             fig_rolling_vol.add_trace(go.Scatter(
    #                 x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Rolling Vol Benchmark'],
    #                 mode='lines', name='Rolling Volatility (Benchmark)',
    #                 line=dict(color=berkeley_blue, width=2)
    #             ))
    #         else:
    #             portf_mkt_rtn_df['Rolling Vol Deep Learning'] = (
    #                 portf_mkt_rtn_df['Deep Learning Return'].rolling(window=window_size).std() * np.sqrt(252)
    #             )
    #             fig_rolling_vol.add_trace(go.Scatter(
    #                 x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Rolling Vol Deep Learning'],
    #                 mode='lines', name='Rolling Volatility (Deep Learning)',
    #                 line=dict(color=berkeley_blue, width=2)
    #             ))

    #         # This is now safe to include
    #         fig_rolling_vol.add_trace(go.Scatter(
    #             x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Rolling Vol Unscaled Market'],
    #             mode='lines', name='Rolling Volatility (Unscaled Market)',
    #             line=dict(color=california_gold, width=2)
    #         ))

    #         # division_date = portf_mkt_rtn_df.loc[portf_mkt_rtn_df['label'] == "Train vs. Test Division", "Date"]
    #         # if not division_date.empty:
    #         #     fig_rolling_vol.add_vline(
    #         #         x=division_date.iloc[0],
    #         #         line=dict(color="black", width=2, dash="dash"),
    #         #         annotation_text="Train vs. Test Division",
    #         #         annotation_position="top right"
    #         #     )


    #         fig_rolling_vol.update_layout(
    #             # title="Rolling Annual Volatility Comparison",
    #             xaxis_title="Date", yaxis_title="Annualized Volatility",
    #             margin=dict(l=40, r=40, t=40, b=40)
    #         )
    #         st.plotly_chart(fig_rolling_vol, use_container_width=True)
    #     else:
    #         st.error("No rolling volatility data available.")


    # Static Stats Table
    def generate_stats_table(self, bm_df, model_df):
        print('generate_stats_table()')
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
    def generate_sentiment_chart(self, last_day, loc):
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
    def generate_next_day_return_chart(self, last_day):
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
        color_map = {
            'Daily DL with Sentiment': berkeley_blue,
            'Daily DL': berkeley_blue_light,
            'Daily Benchmark': california_gold,
            'Unscaled Market': grey,
            'Train vs. Test Division': red,
        }
        
        colors = list(color_map.values())  # Extract the list of colors for each label
        color_idx = len(colors)  # Starting index for the color assignment

        # Add Cumulative Return lines to the first subplot (left)
        for label in cum_df['label'].unique():
            filtered_df = cum_df[cum_df['label'] == label]
            
            # Assign a color to the label from the color_map
            line = dict(color=color_map.get(label, '#1f77b4'))  # Default color if not in the map

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
            line = dict(color=color_map.get(label, '#1f77b4'))  # Default color if not in the map

            if "Division" in label:
                line['dash'] = 'dash'  # Apply dashed line for the "Train vs Test Division"

            fig.add_trace(
                go.Scatter(x=filtered_df['datetime'], y=filtered_df['Y'], mode='lines', name=label,
                           line=line, showlegend=False),
                row=1, col=2
            )

        # Update layout for the figure
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




