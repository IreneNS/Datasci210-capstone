import streamlit as st
import requests
import pandas as pd
# import matplotlib.pyplot as plt
from io import BytesIO
import pickle
import json

# -------------------- CALL S3 URLS -------------------- #
def fetch_s3_data(data_type="tickers", stock_count = None):
    """
    Fetch data from S3 based on the provided data type. 
    For tickers, return the DataFrame with tickers based on selected stock_count (100, 200, or 300).
    """
    # Fetch the S3 URL based on the data_type
    if data_type == "tickers":
        # Construct the S3 URL based on the stock count (100, 200, or 300)
        if stock_count == 100:
            s3_url = st.secrets["s3"]["ticker_100"]
        elif stock_count == 200:
            s3_url = st.secrets["s3"]["ticker_200"]
        elif stock_count == 300:
            s3_url = st.secrets["s3"]["ticker_300"]
        elif stock_count == 0:
            # s3_url = st.secrets["s3"]["full"]
            return None
        else:
            st.error(f"Invalid stock count: {stock_count}. Choose 100, 200, 300, or full universe.")
            return None
        return_type = "dataframe"
    elif data_type == "stats":
        s3_url = st.secrets["s3"]["stats_url"]
        return_type = "dataframe"  # Expecting DataFrame
    elif data_type == "backtest":
        s3_url = st.secrets["s3"]["backtest_url"]
        return_type = "figure"  # Expecting Matplotlib figure
    else:
        st.error(f"Unknown data type: {data_type}")
        return None
    
    # Fetch and return the data from S3
    try:
        # Using the appropriate method based on data type
        if return_type == "dataframe":
            # Fetch the DataFrame (assuming it's pickled)
            df = pd.read_pickle(s3_url)
            return df
        elif return_type == "figure":
            response = requests.get(s3_url) #TODO
            response.raise_for_status()  # Ensure successful request
            
            # Load figure using BytesIO
            fig = pickle.load(BytesIO(response.content))
            axes = fig.get_axes()  # Get all axes in the figure

            # Extract data from each axis (assuming line plot)
            cumulative_df = []
            vol_df = []
            dfs = []
            # Step 3: Extract the data from the figure's axes
            for ax in fig.axes:
                axis_label = "left" if ax.get_ylabel() else "right"  # Check which axis is being used
                # st.write(ax.get_ylabel())
                for line in ax.get_lines():
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    label = line.get_label()   
                    df_line = pd.DataFrame({
                        "datetime": x_data,
                        "Y": y_data,
                        "label": label,
                        # "axis_label": ax.get_title()
                    })
                    if ax.get_title() == "Cumulative Return Comparison":
                        cumulative_df.append(df_line)
                    else:
                        vol_df.append(df_line)

            cumulative_df = pd.concat(cumulative_df, ignore_index=True)
            cumulative_df['Y'].iloc[-2] = cumulative_df['Y'][:-2].min() - 0.1
            cumulative_df['Y'].iloc[-1] = cumulative_df['Y'][:-2].max() + 0.1 # For vert line

            vol_df = pd.concat(vol_df, ignore_index=True)
            vol_df['Y'].iloc[-2] = vol_df['Y'][:-2].min() - 0.01
            vol_df['Y'].iloc[-1] = vol_df['Y'][:-2].max() + 0.01 # For vert line


            new_label_names = {"Daily-DL-with-senti": "Daily DL with Sentiment",
                               "Daily-DL": "Daily DL",
                               "Daily-Benchmark": "Daily Benchmark",
                               "Train vs Test Division": "Train vs. Test Division"}
            cumulative_df['label'] = cumulative_df['label'].replace(new_label_names)
            vol_df['label'] = vol_df['label'].replace(new_label_names)

            return cumulative_df, vol_df
        # elif return_type == "png":
        #     st.image(s3_url, caption="Backtest Performance", use_column_width=True)
        #     return None

    except Exception as e:
        st.error(f"Error fetching data from {s3_url}: {e}")
        return None