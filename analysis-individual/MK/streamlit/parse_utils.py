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
    
    # Convert 'date' field to datetime if it exists
    if 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'], unit='ms')
        df.drop('date', axis=1, inplace=True)
    
    # Check for the presence of columns to differentiate between portf_rtn_test and portf_mkt_rtn_test
    if '0' in df.columns:
        # This is likely portf_rtn_test data, rename accordingly
        df.rename(columns={'0': 'Return'}, inplace=True)
    
    elif 'Daily-Benchmark-max_sharpe-test-last window' in df.columns:
        # This is likely portf_mkt_rtn_test data, rename accordingly
        df.rename(columns={
            'Daily-Benchmark-max_sharpe-test-last window': 'Benchmark Return',
            'Unscaled Market': 'Unscaled Market Return',
            'cum_rtn': 'Cumulative Return',
            'drawdown': 'Drawdown',
            'max_drawdown': 'Max Drawdown'
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
    
    return pd.DataFrame(list(combined_data.items()), columns=['Identifier', 'Weight'])
