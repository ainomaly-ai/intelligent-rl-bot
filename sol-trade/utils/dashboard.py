import streamlit as st
from threading import Thread


# Global variables to store the latest state
latest_sol = None
latest_tokens = None
latest_token_value = None
latest_usd_value = None

def update_dashboard(sol, tokens, token_value, usd_value):
    global latest_sol, latest_tokens, latest_token_value, latest_usd_value
    latest_sol = sol
    latest_tokens = tokens
    latest_token_value = token_value
    latest_usd_value = usd_value

def run_streamlit():
    st.title("Solana Trading Portfolio Dashboard")
    
    while True:
        if latest_sol is not None and latest_tokens is not None and latest_token_value is not None and latest_usd_value is not None:
            st.write(f"Current SOL: {latest_sol}")
            st.write(f"Tokens: {latest_tokens}")
            st.write(f"Token Value: {latest_token_value}")
            st.write(f"USD Value: {latest_usd_value}")
            st.experimental_rerun()  # Force rerun to update the dashboard





# Start the Streamlit app in a separate thread
def start_dashboard():
    dashboard_thread = Thread(target=run_streamlit)
    dashboard_thread.daemon = True  # This will allow you to stop the streamlit server without stopping the ray dashboard
    dashboard_thread.start()