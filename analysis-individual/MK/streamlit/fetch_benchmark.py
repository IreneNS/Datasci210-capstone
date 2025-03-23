import requests

# -------------------- CALL AWS BENCHMARK API -------------------- #
def fetch_benchmark_data(tickers, freq, opt_flag, risk):
    url = "https://qu5qwnx3i7.execute-api.us-west-2.amazonaws.com/model/benchmark" #"http://52.41.158.44:8000/benchmark" 
    payload = {
        "ticker_list": tickers,
        "rebal_freq": freq,
        "opt_flag": opt_flag,
        "target_risk": risk / 100  # Convert percentage to decimal
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an HTTPError if the response code is 4xx/5xx
        
        if response.status_code == 200:
            # Try parsing the response as JSON
            try:
                response_json = response.json()
                # st.write("API Response (Raw):", response.text)  # Print raw response text for debugging
                return response_json
            except ValueError as e:
                st.error(f"Error parsing JSON response: {e}")
                # st.write("Raw Response:", response.text)  # Print raw response content
                return None
        else:
            st.error(f"API request failed with status code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        # Handle other request errors like network issues
        st.error(f"Request failed: {e}")
        return None
