import streamlit as st
import requests

# -------------------- CALL AWS BENCHMARK API -------------------- #
def fetch_benchmark_data(tickers, freq, opt_flag, risk):
    url = st.secrets["api"]["url_1"]  # Fetch URL from secrets.toml

    payload = {
        "ticker_list": tickers,
        "rebal_freq": freq,
        "opt_flag": opt_flag,
        "target_risk": risk / 100  # Convert percentage to decimal
    }
    
    return make_api_request(url, payload)

# -------------------- CALL SECOND API (URL_2) -------------------- #
def fetch_model_data(tickers, freq, opt_flag, risk):
    url = st.secrets["api"]["url_2"]  # Fetch the second URL from secrets.toml

    payload = {
        "ticker_list": tickers,
        "rebal_freq": freq,
        "opt_flag": opt_flag,
        "target_risk": risk / 100  # Convert percentage to decimal
    }
    
    return make_api_request(url, payload)

# -------------------- GENERIC API REQUEST FUNCTION -------------------- #
def make_api_request(url, payload):
    """Handles API requests and error handling."""
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an HTTPError if the response code is 4xx/5xx
        
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as e:
                st.error(f"Error parsing JSON response: {e}")
                return None
        else:
            st.error(f"API request failed with status code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None
