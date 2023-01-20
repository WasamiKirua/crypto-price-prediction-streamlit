import streamlit as st
import pandas as pd
import yfinance as yf
import cufflinks as cf
from dotenv import load_dotenv
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import os

# App title
st.markdown('''
# Crypto Price and Prediction App
**Credits**
- App built by [CryptoShepherd](https://nothaveoneyet.com) (aka [Lello](http://mpthaveoneyet.com))
- Built in `Python` using `streamlit`,`yfinance`, `cufflinks`, `pandas`, `dotenv`, `datetime`, and `prophet`
''')
st.write('---')

# Pandas Options
pd.options.display.float_format = '${:,.2f}'.format

# Load from .env
load_dotenv()
base_url = os.getenv("BASE_URL")

# Binance ticker's list DataFrame
df = pd.read_json('https://api.binance.com/api/v3/ticker/24hr')

# Function for Binance URL builder
def make_klines_url(symbol, **kwargs):
    url = base_url + f"?symbol={symbol}"

    for key, value in kwargs.items():
        url += f"&{key}={value}"
    
    return url

# Custom function for rounding values
def round_value(input_value):
    if input_value.values > 1:
        a = float(round(input_value, 2))
    else:
        a = float(round(input_value, 8))
    return a

# STREAMLIT Sidebar Price
st.sidebar.header('Query Parameters Price')
price_ticker = st.sidebar.selectbox('Ticker', ('BTCUSDT', 'ETHUSDT', 'ATOMUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT', 
                                               'MATICUSDT', 'AVAXUSDT', 'NEARUSDT', 'AAVEUSDT', 'FTMUSDT', 'RUNEUSDT'))
interval_selectbox = st.sidebar.selectbox('Interval', ("1d", "4h", "1h", "30m", "15m"))

# Retrive Ticker Price
selected_crypto_index = list(df.symbol).index(price_ticker)
col_df = df[df.symbol == price_ticker]
col_price = round_value(col_df.weightedAvgPrice)
col_percent = f'{float(col_df.priceChangePercent)}%'

# STREAMLIT Price metric
st.metric(label=price_ticker, value=col_price, delta=col_percent)

# Binance klines DataFrame Preparation
pd.options.display.float_format = '${:,.2f}'.format
klines_url = make_klines_url(price_ticker, interval=interval_selectbox)
klines_ticker_price = pd.read_json(klines_url)
klines_ticker_price.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
                               'Quote Asset Volume', 'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']
klines_ticker_price.drop(['Close Time', 'Quote Asset Volume', 'Number of Trades', 'TB Base Volume', 'TB Quote Volume','Ignore'], axis=1, inplace=True)
klines_ticker_price['Date'] = pd.to_datetime(klines_ticker_price['Date']/1000, unit='s')
klines_ticker_price.set_index(pd.DatetimeIndex(klines_ticker_price['Date']), inplace=True)

# STREAMLIT kline DataFrame Preview
st.subheader(f'{price_ticker} Klines Dataframe Preview')
st.write(klines_ticker_price.tail())

# STREAMLIT functions klines Dataframe Plotting
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=klines_ticker_price['Date'], y=klines_ticker_price['Close'], name='Close'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_raw_data_log():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=klines_ticker_price['Date'], y=klines_ticker_price['Close'], name="Close"))
	fig.update_yaxes(type="log")
	fig.layout.update(xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

def plot_bb_data():
    qf=cf.QuantFig(klines_ticker_price,legend='top',name='GS')
    qf.add_bollinger_bands()
    qf.add_ema(periods=[12,26,200], color=['red', 'orange', 'black'])
    qf.add_volume()
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)

# STREAMLIT Multi Option for Plot 
options_klines = st.multiselect('Customize your Dashboard with Charts', ['log', 'raw', 'bb_ema'])
if len(options_klines) == 0:
    st.subheader(f'{price_ticker} Klines Range Express Data')
    express = px.area(klines_ticker_price, x='Date', y='Close')
    st.write(express)

# STREAMLIT for loop to check plot choice selected
for choice in options_klines:
    if choice == 'log':
        st.subheader(f'{price_ticker} Klines Range Slider Log Data')
        plot_raw_data_log()
    if choice == 'raw':
        st.subheader(f'{price_ticker} Klines Range Slider Raw Data')
        plot_raw_data()
    if choice == 'bb_ema':
        st.subheader(f'{price_ticker} Klines Range BB, EMA Data')
        plot_bb_data()

# STREAMLIT Sidebar Prediction
st.sidebar.header('Query Parameters Prediction')
prediction_ticker = st.sidebar.selectbox('Ticker',('BTC-USD', 'ETH-USD', 'ATOM-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD', 'MATIC-USD', 'AVAX-USD', 'NEAR-USD', 
                                                   'AAVE-USD', 'FTM-USD', 'RUNE-USD') )
start_date = st.sidebar.date_input("Start date", date(2016, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.today())

# STREAMLIT years/days prediction slicer
n_years = st.sidebar.slider("Years of predition:", min_value=1, max_value=10, step=1)
n_days = st.sidebar.slider("Days of prediction:", min_value=7, max_value=90)
years_period = n_years * 365

# Yahoo Finance DataFrame
df_yf = yf.download(prediction_ticker, start_date, end_date) #get the historical prices for this ticker

# Yahoo Finance DataFrame Preparation
df_yf.reset_index(inplace=True)

# Forecasting DataFrame Preparation
df_train = df_yf[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Forecasting
m = Prophet(seasonality_mode="multiplicative")
m.fit(df_train)
future_years = m.make_future_dataframe(periods=years_period)
future_days = m.make_future_dataframe(periods=n_days)
forecast_years = m.predict(future_years)
forecast_days = m.predict(future_days)

# STREAMLIT Yahoo Finance DataFrame Preview
def df_yf_preview():
    st.subheader(f'{prediction_ticker} Yahoo Finance DataFrame Preview')
    st.write(df_yf.tail())

# STREAMLIT functions Yahoo Finance Dataframe Plotting
def plot_yf_raw_date():
    st.subheader(f'{prediction_ticker} Facebook Prophet Forecasting Plot')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_yf['Date'], y=df_yf['Open'], name=f"{prediction_ticker} Open"))
    fig.add_trace(go.Scatter(x=df_yf['Date'], y=df_yf['Close'], name=f"{prediction_ticker} Close"))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# STREAMLIT Forecasting DataFrame and Components Plotting
def plot_year_prediction():
    st.subheader(f'{prediction_ticker} Forecasting Plot for {n_years} Years')
    fig1 = plot_plotly(m, forecast_years)
    st.plotly_chart(fig1)

def plot_year_components():
    st.subheader(f'{prediction_ticker} Years Components')
    fig2 = m.plot_components(forecast_years)
    st.write(fig2)

def plot_day_prediction():
    st.subheader(f'{prediction_ticker} Forecasting Plot for {n_days} Days')
    fig3 = plot_plotly(m, forecast_days)
    st.plotly_chart(fig3)

def plot_day_components():
    st.subheader(f'{prediction_ticker} Days Components')
    fig4 = m.plot_components(forecast_days)
    st.write(fig4)


if st.button('Year Prediction Plot'):
    plot_year_prediction()
    plot_year_components()

if st.button('Days Prediction Plot'):
    plot_day_prediction()
    plot_day_components()
