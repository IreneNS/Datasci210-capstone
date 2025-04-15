# ui.py

import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from styles import HEADER_CONTAINER, BASE_STYLE, BUTTON_STYLE, SELECTBOX_STYLE
from .config import STOCK_OPTIONS, OPTIMIZATION_DISCLAIMER

__all__ = [
    "render_header",
    "render_sidebar_form"
]

def render_header():
    """Render the app's main header."""
    st.markdown(HEADER_CONTAINER, unsafe_allow_html=True)

def render_sidebar_form():
    """Render the sidebar form and return user inputs."""
    st.sidebar.header("Portfolio Configuration")
    with st.sidebar.form('portfolio_form'):
        selected_stock_count = st.selectbox("Set Optimization Universe:", STOCK_OPTIONS.keys())
        
        risk_tolerance = st.slider(
            "Risk Tolerance (Annual Volatility)",
            0.01, 0.99, 0.3,
            help="Select a risk tolerance between 0 and 1."
        )

        with stylable_container("submit_button_container", css_styles=BUTTON_STYLE):
            submit_button = st.form_submit_button("Run Optimization")

        st.sidebar.markdown(OPTIMIZATION_DISCLAIMER)

    return selected_stock_count, risk_tolerance, submit_button
