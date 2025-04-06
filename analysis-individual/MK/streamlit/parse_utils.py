import json
import pandas as pd

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

def convert_to_dataframe(data):
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
    
    # Check for the presence of columns to differentiate between portf_rtn_test and portf_mkt_rtn_test
    # Benchmark case
    if '0' in df.columns:
        # This is likely portf_rtn_test data, rename accordingly
        print('had 0')
        df.rename(columns={'0': 'Return'}, inplace=True)
        print(df.columns)

    if 'Unscaled Market' in df.columns:
        # This is likely portf_mkt_rtn_test data, rename accordingly
        print('had Unscaled Market')
        df.rename(columns={
            'Daily-Benchmark-max_sharpe-test': 'Benchmark Return',
            'Daily-DeepLearning': 'Deep Learning Return',
            'Unscaled Market': 'Unscaled Market Return',
        }, inplace=True)
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
        new_col_names = {"Daily-DL-Max-Sharpe": "Daily Deep Learning",
                         "Daily-DL-Max-Sharpe w senti": "Daily Deep Learning with Sentiment",
                         "Daily-Benchmark": "Daily Benchmark",
                         "Unscaled Market": "Unscaled Market Return"}

        
        df = data.rename(columns=new_col_names, index=new_idx_names)
    else:
        if bm:
            print('yes bm')
            new_col_names = {"Daily-Benchmark-max_sharpe-test": "Daily Benchmark",
                             "Unscaled Market": "Unscaled Market Return",
                             }
        else:
            new_col_names = {"Daily-DeepLearning": "Daily Deep Learning with Sentiment",
                             "Unscaled Market": "Unscaled Market Return"}
        df = pd.DataFrame(data)
        df.set_index('index', inplace=True)
        df.index.name = None
        df.rename(columns=new_col_names, index=new_idx_names, inplace=True)
    
    return df