import json
import pandas as pd
import numpy as np
import streamlit as st

def parse_api_data(data):
    """
    Checks if the input data is a string and tries to parse it as JSON. 
    If it's already a list of dictionaries, it leaves the data as is.
    
    Args:
    data: The data to be parsed (can be a string or a list of dictionaries).
    
    Returns:
    Parsed data as a list of dictionaries or the original data if already parsed.
    """
    # If data is a string, try to parse it as JSON
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON data: {e}")
            return None  # Return None or handle the error as needed
    return data


def convert_to_dataframe(data, comb):
    """
    Converts the parsed data (e.g., portf_rtn_test or portf_mkt_rtn_test) into a pandas DataFrame.
    
    Args:
    data: List of dictionaries containing either portfolio returns or market returns data.
    
    Returns:
    A pandas DataFrame with appropriate columns based on the type of data.
    """
    if data is None or not isinstance(data, list):
        print("Invalid data format. Could not convert to DataFrame.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    print('line40 - ', df.columns)

    # Convert 'date' field to datetime if it exists
    if 'index' in df.columns: #'date'
        df['Date'] = pd.to_datetime(df['index'], unit='ms')
        df.drop('index', axis=1, inplace=True)
    
    # Convert 'date' field to datetime if it exists
    if 'date' in df.columns:
        if comb:
            df['Date'] = pd.to_datetime(df['date'], unit='ms')
            df.drop('date', axis=1, inplace=True)
        else:
            df['Date'] = pd.to_datetime(df['date'])
            df.drop('date', axis=1, inplace=True)

    # Rename '0' column if it exists (used for returns in benchmark JSON)
    if '0' in df.columns:
        df.rename(columns={'0': 'Benchmark Return'}, inplace=True)
        print(df.columns)

    # Rename columns if market data
    if 'Unscaled Market' in df.columns:
        print('had Unscaled Market')
        df.rename(columns={
            'Daily-Benchmark-max_sharpe-test': 'Benchmark Return',
            'Daily-DeepLearning': 'Deep Learning Return',
            'Daily-Benchmark': 'Benchmark Return',
            'Unscaled Market': 'Unscaled Market Return',
        }, inplace=True)

    if 'level_0' in df.columns:
        df.drop('level_0', axis=1, inplace=True)
    # # Set 'Date' as index if it exists
    # if 'Date' in df.columns:
    #     df.set_index('Date', inplace=True)

    return df


def convert_to_weight_dataframe(data):
    """
    Converts the parsed scaled weight data into a pandas DataFrame.
    
    Args:
    data: List of dictionaries containing the scaled weight data.
    
    Returns:
    A pandas DataFrame where keys are identifiers and values are the weights (including nulls).
    """
    if data is None or not isinstance(data, list):
        print("Invalid data format. Could not convert to DataFrame.")
        return None
    
    # Convert list of dictionaries into a single dictionary
    combined_data = {k: v for d in data for k, v in d.items()}
    
    return pd.DataFrame(list(combined_data.items()), columns=['Stock Symbol', 'Weight'])[1:]

def convert_to_stats_dataframe(data, pkl, bm):
    """
    Converts the stats data into a pandas DataFrame.
    
    Args:
    data: List of dictionaries containing the scaled weight data.
    
    Returns:
    A pandas DataFrame where keys are identifiers and values are the weights (including nulls).
    """
    new_idx_names = {"avg_rtn_ann": "Average Annual Return",
                     "vol_ann": "Annual Volatility",
                     "sharpe_ann": "Annual Max Sharpe",
                     "max_drawdown": "Max Drawdown"}

    if pkl:
        new_col_names = {"Daily-DL-Max-Sharpe": "Daily Deep Learning Return",
                         "Daily-DL-Max-Sharpe w senti": "Daily Deep Learning with Sentiment Return",
                         "Daily-Benchmark": "Daily Benchmark Return",
                         "Unscaled Market": "Unscaled Market Return Return"}

        
        df = data.rename(columns=new_col_names, index=new_idx_names)
    else:
        if bm:
            print('yes bm')
            new_col_names = {"Daily-Benchmark-max_sharpe-test": "Daily Benchmark Return",
                             "Daily-Benchmark": "Daily Benchmark Return",
                             "Unscaled Market": "Unscaled Market Return",
                             }
        else:
            new_col_names = {"Daily-DeepLearning": "Daily Deep Learning with Sentiment Return",
                             "Daily-Benchmark": "Daily Benchmark Return",
                             "Unscaled Market": "Unscaled Market Return"}
        df = pd.DataFrame(data)
        
        if 'level_0' in df.columns:
            df.drop('level_0', axis=1, inplace=True)
        
        df.set_index('index', inplace=True)
        df.index.name = None
        df.rename(columns=new_col_names, index=new_idx_names, inplace=True)
    
    return df


def mmd_cal(df, return_col_name):
    """
    Helper function for create_stats_dataframe()

    Args:
    df: Uncalculated dataframe
    return_col_name: The name of the new column
    
    Returns:
    A DataFrame containing new columns, 'cum_rtn','drawdown', 'maxdraw_down'
    """
    df_1 = df.copy() 
    df_1['cum_rtn']=(1+df_1[return_col_name]).cumprod()
    df_1['drawdown'] = (df_1['cum_rtn']-df_1['cum_rtn'].cummax())/df_1['cum_rtn'].cummax()
    df_1['max_drawdown'] =  df_1['drawdown'].cummin()
    return df_1['max_drawdown']

def create_stats_dataframe(data):
    """
    Creates a new stats dataframe from basic benchmark dataframe
    
    Args:
    data: Dataframe of the uncalculated data.
    
    Returns:
    A pandas DataFrame of statistics
    """
    benchmark_stats_parsed_df = pd.DataFrame(columns=data.columns)
    benchmark_stats_parsed_df.loc['avg_rtn_ann', :] = data.select_dtypes(include='number').mean() * 252
    benchmark_stats_parsed_df.loc['vol_ann',:] = data.std()*np.sqrt(252)
    benchmark_stats_parsed_df.loc['sharpe_ann',:] = benchmark_stats_parsed_df.loc['avg_rtn_ann',:]/benchmark_stats_parsed_df.loc['vol_ann',:]
    benchmark_stats_parsed_df.loc['max_drawdown', 'Benchmark Return'] = mmd_cal(data, 'Benchmark Return').iloc[-1]

    new_idx_names = {"avg_rtn_ann": "Average Annual Return",
                     "vol_ann": "Annual Volatility",
                     "sharpe_ann": "Annual Max Sharpe",
                     "max_drawdown": "Max Drawdown"}
    new_col_names = {"Daily-DL-Max-Sharpe": "Daily Deep Learning Return",
                     "Daily-DL-Max-Sharpe w senti": "Daily Deep Learning with Sentiment Return",
                     "Daily-Benchmark": "Daily Benchmark Return",
                     "Benchmark Return": "Daily Benchmark Return",
                     "Unscaled Market": "Unscaled Market Return"}

    df = benchmark_stats_parsed_df.rename(columns=new_col_names, index=new_idx_names)
    return df
        