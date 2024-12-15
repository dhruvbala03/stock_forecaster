import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go

from pages.resources.helper_functions import ALL_TICKERS, generate_investment_strategy, get_future_dates, get_snp500_forecasts, get_stock_name, get_train_test_data, generate_historical_predictions, generate_forecast

from functools import cache


START = '2010-01-01'
TODAY = datetime.today().strftime("%Y-%m-%d")


# Investment Strategy

st.title('Investment Strategy')

selected_principal = st.number_input('Principal Amount ($)', value=100)
selected_appetite = st.slider('Risk Appetite (1 is very conservative, 10 is very ambitious)', min_value=1, max_value=10, value=5)

# Button click handler
if 'strategy' not in st.session_state:
    st.session_state['strategy'] = ''

def write_forecast():
    snp500_forecasts = get_snp500_forecasts(tuple(get_future_dates(15)))
    csv_plaintext = snp500_forecasts.to_csv(index=False)
    st.session_state['strategy'] = generate_investment_strategy(csv_plaintext, selected_principal, selected_appetite)

# Button and output
st.button('Go', on_click=write_forecast)
if st.session_state['strategy']:
    st.subheader('Recommended Strategy:')
st.write(st.session_state['strategy'])
