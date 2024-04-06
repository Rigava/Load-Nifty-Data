import streamlit as st
import requests
import os
import io
# import sys
# import subprocess

# # check if the library folder already exists, to avoid building everytime you load the pahe
# if not os.path.isdir("/tmp/ta-lib"):

#     # Download ta-lib to disk
#     with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
#         response = requests.get(
#             "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
#         )
#         file.write(response.content)
#     # get our current dir, to configure it back again. Just house keeping
#     default_cwd = os.getcwd()
#     os.chdir("/tmp")
#     # untar
#     os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
#     os.chdir("/tmp/ta-lib")
#     os.system("ls -la /app/equity/")
#     # build
#     os.system("./configure --prefix=/home/appuser")
#     os.system("make")
#     # install
#     os.system("make install")
#     # back to the cwd
#     os.chdir(default_cwd)
#     sys.stdout.flush()

# # add the library to our current environment
# from ctypes import *

# lib = CDLL("/home/appuser/lib/libta_lib.so.0.0.0")
# # import library
# try:
#     import talib_bin
# except ImportError:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "--global-option=build_ext", "--global-option=-L/home/appuser/lib/", "--global-option=-I/home/appuser/include/", "ta-lib-bin"])
# finally:
#     import talib_bin

import pandas as pd
import plotly.graph_objects as go
import pickle
from unidecode import unidecode
from patterns import candlestick_patterns
import requests


st.title('NIFTY STOCK DASHBOARD')

with open("nifty50tickers.pickle",'rb') as f:
    tickers=pickle.load(f)

ticker_choice = tickers
symbol = st.sidebar.selectbox("Select a stock",ticker_choice)
st.write(f"This is the chart of {symbol}")
dashboard = st.sidebar.selectbox("select analysis",["Squeeze","Breakouts","Crossovers"])
#the below file settings is for plotting the chart only
url = "https://raw.githubusercontent.com/Rigava/Load-Nifty-Data/main/stock_dfs_updated/{}.csv".format(symbol)
download = requests.get(url).content
chart_df = pd.read_csv(io.StringIO(download.decode('utf-8')))
print(chart_df.head())
df = chart_df[['Date','OpenPrice','HighPrice','LowPrice','ClosePrice','TotalTradedQuantity']]
df.rename(columns={df.columns[0]:"Date",df.columns[1]:"Open",df.columns[2]:"High",df.columns[3]:"Low",df.columns[4]:"Close",df.columns[5]:"Volume"},inplace=True)
fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
fig.update_xaxes(type='category')
fig.update_layout(height=800)
st.plotly_chart(fig,use_container_width=True)


#below setting is for identifying the breakouts based on squeezing and consolidation/Breakout functions
if dashboard == "Squeeze":
    squeeze=[]
    for files in tickers:
        url = "https://raw.githubusercontent.com/Rigava/Load-Nifty-Data/main/stock_dfs_updated/{}.csv".format(files)
        download = requests.get(url).content
        data = pd.read_csv(io.StringIO(download.decode('utf-8')))
        #selecting the relevant columns
        df = data[['Date','OpenPrice','HighPrice','LowPrice','ClosePrice','TotalTradedQuantity']]
        df = df.drop_duplicates(subset=['Date'],keep='first')
        df.rename(columns={df.columns[0]:"Date",df.columns[1]:"Open",df.columns[2]:"High",df.columns[3]:"Low",df.columns[4]:"Close",df.columns[5]:"Volume"},inplace=True)
        
        cols = df.select_dtypes(exclude=['float']).columns
        df['Date']=pd.to_datetime(df['Date'])
        for col in cols:
            if col == 'Date':
                pass
            else:
                df[col] = df[col].apply(lambda x: (unidecode(x).replace(',',''))).astype(float)

        df['20sma'] = df['Close'].rolling(window=20).mean()
        df['stddev'] = df['Close'].rolling(window=20).std()
        df['lower_band'] = df['20sma'] - (2 * df['stddev'])
        df['upper_band'] = df['20sma'] + (2 * df['stddev'])

        df['TR'] = abs(df['High'] - df['Low'])
        df['ATR'] = df['TR'].rolling(window=20).mean()
        df['lower_keltner'] = df['20sma'] - (df['ATR'] * 1.5)
        df['upper_keltner'] = df['20sma'] + (df['ATR'] * 1.5)

        def in_squeeze(df):
            return df['lower_band'] > df['lower_keltner'] and df['upper_band'] < df['upper_keltner']
        df['squeeze_on'] = df.apply(in_squeeze, axis=1)

        if df.iloc[-3]['squeeze_on'] and not df.iloc[-1]['squeeze_on']:
            squeeze.append(files)
    st.write("List of stock coming out of squeeze phase",squeeze)

def is_consolidating(data):
    recent_candles = data[-15:]
    max_close = recent_candles['Close'].max()
    min_close = recent_candles['Close'].min()
    if min_close > (max_close * 0.95):
        return True
    return False


def is_breakingout(data):
    last_close = data[-1:]['Close'].values[0]
    if is_consolidating(data[:-1]):
        recent_candles = data[-16:-1]
        if last_close > recent_candles['Close'].max():
            return True
    return False


def is_breakdown(data):
    last_close = data[-1:]['Close'].values[0]
    if is_consolidating(data[:-1]):
        recent_candles = data[-16:-1]
        if last_close < recent_candles['Close'].min():
            return True
    return False
if dashboard == "Breakouts":
    consolidation = []
    breakup=[]
    breakdown = []
    for files in tickers:
        url = "https://raw.githubusercontent.com/Rigava/Load-Nifty-Data/main/stock_dfs_updated/{}.csv".format(files)
        download = requests.get(url).content
        data = pd.read_csv(io.StringIO(download.decode('utf-8')))
        
        #selecting the relevant columns
        df = data[['Date','OpenPrice','HighPrice','LowPrice','ClosePrice','TotalTradedQuantity']]
        df = df.drop_duplicates(subset=['Date'],keep='first')
        df.rename(columns={df.columns[0]:"Date",df.columns[1]:"Open",df.columns[2]:"High",df.columns[3]:"Low",df.columns[4]:"Close",df.columns[5]:"Volume"},inplace=True)
        
        cols = df.select_dtypes(exclude=['float']).columns
        df['Date']=pd.to_datetime(df['Date'])
        for col in cols:
            if col == 'Date':
                pass
            else:
                df[col] = df[col].apply(lambda x: (unidecode(x).replace(',',''))).astype(float)        
        if is_consolidating(df):
            consolidation.append(files)
        if is_breakingout(df):
            breakup.append(files)
        if is_breakdown(df):
            breakdown.append(files)
    st.write("List of stock in consolidation",consolidation)
    st.write("List of stock trying to break up",breakup)
    st.write("List of stock trying to break down",breakdown)

import pandas_ta as ta 
if dashboard == "Crossovers":
    # User input for strategy parameters
    fast_period = st.slider("Fast Period", min_value=5, max_value=50, value=12, step=1)
    slow_period = st.slider("Slow Period", min_value=10, max_value=200, value=26, step=1)
    rsi_period = st.slider("RSI Period", min_value=5, max_value=50, value=14, step=1)
    if len(df) > 0:
        # Calculate crossover, MACD, and RSI indicators
        df["MA_fast"] = ta.sma(df["Close"], timeperiod=fast_period)
        df["MA_slow"] = ta.sma(df["Close"], timeperiod=slow_period)
        # df["MACD"],_,_ = ta.macd(df["Close"], fastperiod=fast_period, slowperiod=slow_period, signalperiod=9)
        df["RSI"] = ta.rsi(df["Close"], timeperiod=rsi_period)
    st.write(df)