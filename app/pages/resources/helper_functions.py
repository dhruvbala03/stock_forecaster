
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from openai import OpenAI
import os

from functools import cache


st.set_page_config('Stock Price Forecasts')

START = '2010-01-01'
TODAY = datetime.today().strftime("%Y-%m-%d")

stock_info_df = pd.read_csv('data/sp500.csv')
ALL_TICKERS = stock_info_df['Symbol'].to_list()
MOST_ACTIVE_TICKERS = ['NVDA', 'WBA', 'LCID', 'TSLA', 'SOUN', 'PLTR', 'SMCI', 'GSAT', 'INTC', 'NIO', 'AI', 'GRAB', 'GOOGL', 'BBD', 'PLUG', 'SOFI', 'RIOT', 'MARA', 'F', 'PFE', 'AMD', 'RIVN', 'GOOG', 'AAL', 'RIG', 'AAPL', 'CLSK', 'IONQ', 'BAC', 'HOOD', 'ORCL', 'ITUB', 'CMCSA', 'AMZN', 'T', 'IQ', 'VALE', 'MU', 'JBLU', 'SNAP', 'UBER', 'WULF', 'BTG', 'NU', 'WBD', 'ACHR', 'MSTR', 'IREN', 'AMCR', 'AVGO']

# openai.api_key = os.environ.get('OPENAI_API_KEY')
openai_client = OpenAI()

# Load model
single_day_model = load_model('models/aapl_keras_model.keras')
fifteen_day_model = load_model('models/aapl_15_day_forecast.keras')

@st.cache_data
def get_train_test_data(ticker):

    @st.cache_data
    def get_historic_data(ticker):
        df = yf.download(ticker, START, TODAY)
        df.columns = df.columns.droplevel(1)
        df.columns.name = None
        return df
    
    df = get_historic_data(ticker)
    if len(df) == 0:
        raise ValueError

    # Get train and test data
    split = int(0.7 * len(df))
    train_df, test_df = pd.DataFrame(df[:split]), pd.DataFrame(df[split:])
    return train_df, test_df 


def get_stock_name(ticker):
    result = stock_info_df.loc[stock_info_df['Symbol'] == ticker, 'Security']
    if not result.empty:
        return result.iloc[0]
    else:
        return "Stock"


def generate_historical_predictions(train_df, test_df):

    past_days = 100  # Number of past days for input
    future_days = 1  # Number of future days for output

    test_df = pd.concat([train_df.tail(past_days), test_df], ignore_index=True)

    scaler = MinMaxScaler(feature_range=(0,1))
    test_scaled = scaler.fit_transform(np.array(test_df['Close']).reshape(-1,1))

    # Prepare test_x, test_y
    test_x = np.array([test_scaled[i-past_days:i] for i in range(past_days, len(test_scaled)-future_days)])
    test_y = np.array([test_scaled[i:i+future_days] for i in range(past_days, len(test_scaled)-future_days)])

    pred_y = single_day_model.predict(test_x)

    hist_test_y = test_y[:,0]
    hist_pred_y = pred_y[:,0]

    scale_factor = 1 / scaler.scale_[0]

    hist_test_y *= scale_factor
    hist_pred_y *= scale_factor

    return hist_test_y, hist_pred_y

def generate_forecast(train_df, test_df):

    past_days = 150  
    future_days = 15 

    test_df = pd.concat([train_df.tail(past_days), test_df], ignore_index=True)

    scaler = MinMaxScaler(feature_range=(0,1))
    test_scaled = scaler.fit_transform(np.array(test_df['Close']).reshape(-1,1))

    recent_x = np.array([test_scaled[-past_days:]])
    forecast = fifteen_day_model.predict(recent_x).flatten()

    scale_factor = 1/scaler.scale_[0]

    forecast *= scale_factor

    return forecast


@cache
def get_snp500_forecasts(dates):
    try:
        return pd.read_csv(f'cache/cached_forecasts_{TODAY}.csv')
    except:
        pass
    forecasts = pd.DataFrame(index=dates)
    for ticker in MOST_ACTIVE_TICKERS:
        try:
            train_df, test_df = get_train_test_data(ticker)
            forecasts[ticker] = generate_forecast(train_df, test_df)
        except:
            continue
    forecasts.to_csv(f'cached_forecasts_{TODAY}.csv')
    return forecasts


def generate_investment_strategy(predictions, principal, appetite):
    prompt = f"""{predictions}\n
    I am a beginner to stock investment. Based on the stock predictions above (csv format), and a user investment amount of ${principal} with a {appetite} risk appetite (where 1 is extremely conservative, 10 is extremely ambitious),
    create a detailed investment strategy for me -- including allocations, buy/sell decisions, and rationale. 
    Make your response very very concise. Simply make a list of companies and dollar amounts. Also indicate briefly when I should sell. Note down the principal and risk accurately; make sure the investment dollar amounts add up to the principal. Format as follows:
    - Company 1 -- dollar amount
      - Brief rationale and when to sell
    - Company 2 -- dollar amount
      - Brief rationale and when to sell
    ...
    Don't apply any special formatting.
    """
    print(prompt)

    # Call the OpenAI Completion API
    response = openai_client.chat.completions.create(
        model="gpt-4o",  # Use "gpt-4" if available
        messages=[
            {"role": "system", "content": "You are a financial expert specializing in portfolio strategies."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,  # Adjust creativity
        max_tokens=300  # Adjust for desired response length
    )
    
    # Extract and return the generated text
    print(response.choices[0].message)
    return response.choices[0].message.content # [choice['message']['JSON']['content'] for choice in response['choices']]


def get_future_dates(window):
    return pd.date_range(start=datetime.now()+pd.Timedelta(days=1), periods=window)

