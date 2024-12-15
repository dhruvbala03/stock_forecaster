import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go

from pages.resources.helper_functions import * #ALL_TICKERS, get_stock_name, get_train_test_data, generate_historical_predictions, generate_forecast

st.title('Stock Price Forecasts')
selected_ticker = st.selectbox('Select Stock', ALL_TICKERS, 39, format_func = lambda ticker : f"{get_stock_name(ticker)} ({ticker})")

try:
    train_df, test_df = get_train_test_data(selected_ticker)
    hist_test_y, hist_pred_y = generate_historical_predictions(train_df, test_df)
    forecast = generate_forecast(train_df, test_df)
    pred_y = np.concat((hist_pred_y, forecast))

    historical_dates = test_df.index
    future_dates = pd.date_range(start=datetime.now()+pd.Timedelta(days=1), periods=15)
    extended_dates = test_df.index.union(future_dates)

    st.subheader(f'Historical Trend for {get_stock_name(selected_ticker)}')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=hist_test_y.flatten(),
        mode='lines',
        name='Closing Price ($)',
        line=dict(color='blue')  # Optionally specify color for original price
    ))
    fig.add_trace(go.Scatter(
        x=extended_dates,
        y=pred_y.flatten(),
        mode='lines',
        name='Predicted Price ($)',
        line=dict(color='orange') 
    ))
    st.plotly_chart(fig)

    forecast_df = pd.DataFrame(
        data={'Predicted Price ($)': pred_y.flatten()[-15:]},  # Last 15 predictions
        index=extended_dates[-15:]  # Set the last 15 extended dates as the index
    )
    forecast_df['Predicted Price ($)'] = forecast_df['Predicted Price ($)'].apply(lambda x: f"{x:.2f}")
    forecast_df.index = pd.to_datetime(forecast_df.index).strftime('%b %d, %Y')
    forecast_df.index.name = 'Date'
    st.subheader('15-Day Forecast')

    st.write(forecast_df)
except:
    st.write(f'Sorry, no information could be found on {selected_ticker}.')

