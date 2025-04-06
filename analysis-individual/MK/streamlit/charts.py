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
    # def generate_pie_chart(self, scaled_weight_df):
    #     if not scaled_weight_df.empty:
    #         # Debug: Print the DataFrame
    #         st.write(scaled_weight_df)
    #         st.write('sum:', sum(scaled_weight_df['Weight']))

    #         # Ensure the 'Identifier' and 'Weight' columns are present and valid
    #         if 'Identifier' in scaled_weight_df.columns and 'Weight' in scaled_weight_df.columns:
    #             # Check if the Weight column has valid numerical data
    #             if scaled_weight_df['Weight'].isna().any():
    #                 st.error("The 'Weight' column contains NaN values.")
    #             else:
    #                 # Create the pie chart
    #                 fig_pie = px.pie(scaled_weight_df, values="Weight", names="Identifier", title="Optimized Portfolio Asset Allocation")
    #                 st.plotly_chart(fig_pie, use_container_width=True)
    #         else:
    #             st.error("The DataFrame is missing 'Identifier' or 'Weight' columns.")
    #     else:
    #         st.error("No weight data available.")
    def generate_pie_chart(self, scaled_weight_df):
        if not scaled_weight_df.empty:
            # st.write(scaled_weight_df)
            # st.write('sum:', sum(scaled_weight_df['Weight']))

            if 'Stock Symbol' in scaled_weight_df.columns and 'Weight' in scaled_weight_df.columns:
                if scaled_weight_df['Weight'].isna().any():
                    st.error("The 'Weight' column contains NaN values.")
                    return

                # Set threshold for grouping
                threshold_value = 0.03  # 3%

                # Separate large and small weights
                large_slices = scaled_weight_df[scaled_weight_df["Weight"] > threshold_value]
                small_slices = scaled_weight_df[scaled_weight_df["Weight"] <= threshold_value]

                # Combine small slices into "Other"
                if not small_slices.empty:
                    other_total = small_slices["Weight"].sum()
                    other_row = pd.DataFrame({"Stock Symbol": ["Stocks < 3%"], "Weight": [other_total]})
                    updated_df = pd.concat([large_slices, other_row], ignore_index=True)
                else:
                    updated_df = scaled_weight_df

                # Plot pie chart
                fig_pie = px.pie(updated_df, values="Weight", names="Stock Symbol", title="Optimized Portfolio Asset Allocation")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.error("The DataFrame is missing 'Stock Symbol' or 'Weight' columns.")
        else:
            st.error("No weight data available.")

    # Cumulative Returns Comparison
    def generate_cumulative_returns_chart(self, portf_mkt_rtn_df, bm):
        if not portf_mkt_rtn_df.empty:
            fig_cumulative = go.Figure()
            
            if bm:
                portf_mkt_rtn_df['Cumulative Benchmark Return'] = (1 + portf_mkt_rtn_df['Benchmark Return']).cumprod()
                portf_mkt_rtn_df['Cumulative Market Return'] = (1 + portf_mkt_rtn_df['Unscaled Market Return']).cumprod()
                
                fig_cumulative.add_trace(go.Scatter(
                    x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Cumulative Benchmark Return'],
                    mode='lines', name='Cumulative Benchmark Return',
                    line=dict(color=berkeley_blue, width=2)
                ))

            else:
                portf_mkt_rtn_df['Cumulative Deep Learning Return'] = (1 + portf_mkt_rtn_df['Deep Learning Return']).cumprod()
                fig_cumulative.add_trace(go.Scatter(
                    x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Cumulative Deep Learning Return'],
                    mode='lines', name='Cumulative Deep Learning Return',
                    line=dict(color=berkeley_blue, width=2)
                ))

            # Common unscaled market trace
            portf_mkt_rtn_df['Cumulative Market Return'] = (1 + portf_mkt_rtn_df['Unscaled Market Return']).cumprod()
            fig_cumulative.add_trace(go.Scatter(
                x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Cumulative Market Return'],
                mode='lines', name='Cumulative Unscaled Market Return',
                line=dict(color=california_gold, width=2)
            ))

            # division_date = portf_mkt_rtn_df.loc[portf_mkt_rtn_df['label'] == "Train vs. Test Division", "Date"]
            # if not division_date.empty:
            #     fig_cumulative.add_vline(
            #         x=division_date.iloc[0],
            #         line=dict(color="black", width=2, dash="dash"),
            #         annotation_text="Train vs. Test Division",
            #         annotation_position="top right"
            #     )

            fig_cumulative.update_layout(
                title="Cumulative Return Comparison",
                xaxis_title="Date", yaxis_title="Cumulative Return",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)
        else:
            st.error("No cumulative return data available.")


    # Rolling Annual Volatility Comparison
    def generate_volatility_chart(self, portf_mkt_rtn_df, bm, last_win_only=False):
        if not portf_mkt_rtn_df.empty:
            window_size = 60 if last_win_only else 252
            fig_rolling_vol = go.Figure()

            # Always compute unscaled market volatility
            portf_mkt_rtn_df['Rolling Vol Unscaled Market'] = (
                portf_mkt_rtn_df['Unscaled Market Return'].rolling(window=window_size).std() * np.sqrt(252)
            )

            if bm:
                portf_mkt_rtn_df['Rolling Vol Benchmark'] = (
                    portf_mkt_rtn_df['Benchmark Return'].rolling(window=window_size).std() * np.sqrt(252)
                )
                fig_rolling_vol.add_trace(go.Scatter(
                    x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Rolling Vol Benchmark'],
                    mode='lines', name='Rolling Volatility (Benchmark)',
                    line=dict(color=berkeley_blue, width=2)
                ))
            else:
                portf_mkt_rtn_df['Rolling Vol Deep Learning'] = (
                    portf_mkt_rtn_df['Deep Learning Return'].rolling(window=window_size).std() * np.sqrt(252)
                )
                fig_rolling_vol.add_trace(go.Scatter(
                    x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Rolling Vol Deep Learning'],
                    mode='lines', name='Rolling Volatility (Deep Learning)',
                    line=dict(color=berkeley_blue, width=2)
                ))

            # This is now safe to include
            fig_rolling_vol.add_trace(go.Scatter(
                x=portf_mkt_rtn_df['Date'], y=portf_mkt_rtn_df['Rolling Vol Unscaled Market'],
                mode='lines', name='Rolling Volatility (Unscaled Market)',
                line=dict(color=california_gold, width=2)
            ))

            # division_date = portf_mkt_rtn_df.loc[portf_mkt_rtn_df['label'] == "Train vs. Test Division", "Date"]
            # if not division_date.empty:
            #     fig_rolling_vol.add_vline(
            #         x=division_date.iloc[0],
            #         line=dict(color="black", width=2, dash="dash"),
            #         annotation_text="Train vs. Test Division",
            #         annotation_position="top right"
            #     )


            fig_rolling_vol.update_layout(
                title="Rolling Annual Volatility Comparison",
                xaxis_title="Date", yaxis_title="Annualized Volatility",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_rolling_vol, use_container_width=True)
        else:
            st.error("No rolling volatility data available.")



    # Weights Table
    def generate_weights_table(self, bm_df, model_df):
        """Generate the portfolio weights table."""
        if not bm_df.empty and not model_df.empty:
            # Merge while keeping the order of bm_df
            combined_df = pd.merge(
                bm_df, model_df, on="Stock Symbol", how="left", suffixes=("_Benchmark", "_Model")
            )

            # Fill missing weights with 0
            combined_df["Weight_Model"] = combined_df["Weight_Model"].fillna(0)

            # Format weights as percentages
            combined_df["Benchmark Weight (%)"] = (combined_df["Weight_Benchmark"] * 100).round(2)
            combined_df["Deep Learning Weight(%)"] = (combined_df["Weight_Model"] * 100).round(2)

            # Prepare display dataframe
            display_df = combined_df[["Stock Symbol", "Benchmark Weight (%)", "Deep Learning Weight(%)"]]
            display_df.index = range(1, len(display_df) + 1)

            st.dataframe(display_df)

        else:
            st.error("Weights data unavailable or invalid.")

    # Static Stats Table
    def generate_stats_table(self, bm_df, model_df):
        print('generate_stats_table()')
        if bm_df is not None and model_df is not None:
            common_columns = bm_df.columns.intersection(model_df.columns)
            bm_df = bm_df.drop(columns=common_columns)
            combined_df = pd.concat([bm_df, model_df], axis=1)
            # combined_df.columns = ["Benchmark", "Model"]



            st.dataframe(combined_df)
        elif bm_df is not None:
            st.dataframe(bm_df)
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




