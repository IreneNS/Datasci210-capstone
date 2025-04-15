# styles.py

import streamlit as st
from theme import berkeley_blue, california_gold, light_grey, white

# Header container styling
HEADER_CONTAINER = f"""
    <div class='header-container' style='background-color:{berkeley_blue}; padding:20px; border-radius:10px; text-align:center; margin-bottom:30px;'>
        <h1 style='color:{california_gold};'>📊 AI-Powered Portfolio Optimizer</h1>
        <p style='color:{white}; font-size:18px;'>Enter your stock universe, specify risk tolerance, and get an optimized portfolio to minimize drawdowns during regime changes.</p>
    </div>
"""

# Sidebar and footer base styles
BASE_STYLE = f"""
    <style>
        .sidebar .sidebar-content {{
            background-color: {light_grey};
            padding: 15px;
        }}
        .footer {{
            background-color: {berkeley_blue};
            padding: 10px;
            text-align: center;
            color: {california_gold};
            font-size: 12px;
            border-radius: 10px;
        }}
        .footer a {{
            color: {white};
            text-decoration: none;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}
    </style>
"""

# Button style
BUTTON_STYLE = """
    button {
        background-color: white;
        color: #003B5C !important;
        border: 2px solid #003B5C !important;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    button:hover {
        background-color: #003B5C;
        color: #FDB515 !important;
        border: 2px solid #003B5C;
    }
    button:active {
        background-color: #FDB515 !important;
        color: #003B5C !important;
        border: 2px solid #FDB515 !important;
    }
    button:focus {
        outline: none !important;
    }
"""

# Selectbox style
SELECTBOX_STYLE = """
    div.stSelectbox > select {
        border: 2px solid #003B5C !important;
        border-radius: 5px !important;
    }
    div.stSelectbox > select:focus {
        border: 2px solid #003B5C !important;
    }
"""

# Footer content
FOOTER_HTML = f"""    
    <div class="footer">
        &copy; 2025 AI-Powered Portfolio Optimizer
        <br>
        Developed by <strong>Deep Learning Sentiment Portfolio Capstone Team</strong> - UC Berkeley, MIDS School of Information
        <br>
        <a href="https://your-website.com" target="_blank">Visit Project Website</a> | 
        <a href="https://github.com/your-repo" target="_blank">GitHub</a>
    </div>
"""


def apply_base_styles():
    st.markdown(BASE_STYLE, unsafe_allow_html=True)
    st.markdown(f"<style>{BUTTON_STYLE}</style>", unsafe_allow_html=True)
    st.markdown(f"<style>{SELECTBOX_STYLE}</style>", unsafe_allow_html=True)

