import streamlit as st

pg = st.navigation([st.Page('pages/stock_price_forecasts.py', title='Stock Price Forecasts'), st.Page('pages/investment_strategy.py', title='Investment Strategy')])
pg.run()
