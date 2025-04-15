import streamlit as st
import requests
import pandas as pd
# import matplotlib.pyplot as plt
from io import BytesIO
import pickle
import json


# -------------------- CALL AWS BENCHMARK API -------------------- #
def fetch_benchmark_data(tickers, freq, opt_flag, risk, win):
    url = st.secrets["api"]["url_1"]  # Fetch URL from secrets.toml

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "ticker_list": tickers,
        "rebal_freq": freq,
        "opt_flag": opt_flag,
        "target_risk": risk,  # Convert percentage to decimal
        "last_win_only": win
    }
    
    return make_api_request(url, headers, payload)

# -------------------- CALL SECOND API (URL_2) -------------------- #
def fetch_model_data(tickers, freq, opt_flag, risk, win):
    url = st.secrets["api"]["url_2"]  # Fetch the second URL from secrets.toml

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "ticker_list": tickers,
        "rebal_freq": freq,
        "opt_flag": opt_flag,
        "target_risk": risk,  # Convert percentage to decimal
        "last_win_only": win
    }
    
    return make_api_request(url, headers, payload)

# -------------------- GENERIC API REQUEST FUNCTION -------------------- #
def make_api_request(url, headers, payload):
    """Handles API requests and error handling, including fetching data from URLs."""
    try:
        # st.write("Sending payload to API:", json.dumps(payload, indent=2)) #del

        # Send POST request to API
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an HTTPError if the response code is 4xx/5xx
        
        if response.status_code == 200:
            try:
                # Check if the response is a URL or JSON data
                response_json = response.json()
                # st.write(response_json)

                # Check if the response contains a URL to fetch JSON data from
                if isinstance(response_json, str) and response_json.startswith('http'):
                    # If the response is a URL (string), fetch the JSON from the URL
                    return fetch_json_from_url(response_json)
                else:
                    # Return the JSON response directly
                    return response_json

            except ValueError as e:
                st.error(f"Error parsing JSON response: {e}")
                return None
        else:
            st.error(f"API request failed with status code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Request for API failed: {e}")
        if e.response is not None:
            st.error(f"Response content: {e.response.text}")
        return None

def fetch_json_from_url(url):
    """Fetch and parse JSON data from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure successful request
        return response.json()  # Return the parsed JSON
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching JSON from URL {url}: {e}")
        return None

