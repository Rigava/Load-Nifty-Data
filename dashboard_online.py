import streamlit as st
import requests
import io
import pandas as pd
import plotly.graph_objects as go
import pickle
from unidecode import unidecode
from patterns import candlestick_patterns
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance
import pandas_ta as ta

st.title('NIFTY 50 STOCK DASHBOARD')

with open("nifty50tickers.pickle",'rb') as f:
    tickers=pickle.load(f)

dashboard = st.sidebar.selectbox("select analysis",["Data","Squeeze","Breakouts","Crossover & RSI Shortlist","RSI Strategy","Moving Average Strategy","RSI SMA Strategy"])

## Dashboard 0
if dashboard == "Data":
    symbol = st.sidebar.selectbox("Select stock to pull data", tickers)
    st.title(symbol+" Stocks Price Update")
    if symbol:
        try:
            ticker = symbol+'.NS'
            stock_data = yfinance.Ticker(ticker).history(period="1y")
            latest_price = stock_data['Close'].iloc[-1].round(1)
            print(stock_data,latest_price)
            
            stock_data["RSI"] = ta.rsi(stock_data["Close"], lentgh =14).round(1)
            # stock_data["ADX"] = stock_data.ta.adx()/round(1)
            latest_rsi = stock_data['RSI'].iloc[-1]
            st.success(f"The latest price is: {latest_price} and the rsi is {latest_rsi}")
            # Plotting historical price movement
            st.subheader("Historical Price Movement in Line chart")
            plt.figure(figsize=(10, 6))
            plt.plot(stock_data.index, stock_data['Close'])
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title('Price Movement')
            plt.xticks(rotation=45)
            st.pyplot(plt)
            st.dataframe(stock_data.tail(10))
            # Export data as CSV
            st.subheader("Export Data")
            if st.button("Export as CSV"):
                st.write("Exporting stock data as CSV...")
                stock_data.to_csv(f"{symbol}_data.csv", index=False)
                st.success("Stock data exported successfully!")    
        except Exception as e:
            st.error("Error occurred while fetching stock data.")
            st.error(e)

## Dashboard 1 SQUEEZE
#below setting is for identifying the breakouts based on squeezing and consolidation/Breakout functions
if dashboard == "Squeeze":
    squeeze=[]
    for files in tickers:
        url = "https://raw.githubusercontent.com/Rigava/Load-Nifty-Data/main/stock_dfs_updated/{}.csv".format(files)
        download = requests.get(url).content
        data = pd.read_csv(io.StringIO(download.decode('utf-8')))
        df=data.copy()
        # #selecting the relevant columns
        # df = data[['Date','OpenPrice','HighPrice','LowPrice','ClosePrice','TotalTradedQuantity']]
        # df = df.drop_duplicates(subset=['Date'],keep='first')
        # df.rename(columns={df.columns[0]:"Date",df.columns[1]:"Open",df.columns[2]:"High",df.columns[3]:"Low",df.columns[4]:"Close",df.columns[5]:"Volume"},inplace=True)
        
        # cols = df.select_dtypes(exclude=['float']).columns
        # df['Date']=pd.to_datetime(df['Date'])
        # for col in cols:
        #     if col == 'Date':
        #         pass
        #     else:
        #         df[col] = df[col].apply(lambda x: (unidecode(x).replace(',',''))).astype(float)

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
## Dashboard 2 Functions
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
## Dashboard 2 BREAKOUTS
if dashboard == "Breakouts":
    consolidation = []
    breakup=[]
    breakdown = []
    for files in tickers:
        url = "https://raw.githubusercontent.com/Rigava/Load-Nifty-Data/main/stock_dfs_updated/{}.csv".format(files)
        download = requests.get(url).content
        data = pd.read_csv(io.StringIO(download.decode('utf-8')))
        df=data.copy()

        if is_consolidating(df):
            consolidation.append(files)
        if is_breakingout(df):
            breakup.append(files)
        if is_breakdown(df):
            breakdown.append(files)
    st.write("List of stock in consolidation",consolidation)
    st.write("List of stock trying to break up",breakup)
    st.write("List of stock trying to break down",breakdown)
## Dashboard 3 CROSSOVERS
if dashboard == "Crossover & RSI Shortlist":
    # User input for strategy parameters
    fast = st.sidebar.slider("Fast Period", min_value=5, max_value=50, value=12, step=1)
    slow = st.sidebar.slider("Slow Period", min_value=10, max_value=200, value=26, step=1)
    rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=50, value=14, step=1)
    rsi_low = st.sidebar.slider("RSI low", min_value=1, max_value=100, value=30, step=1)
    rsi_high = st.sidebar.slider("RSI high", min_value=1, max_value=100, value=70, step=1)
    Buy = []
    Sell = []
    Hold = []
    for files in tickers:
        url = "https://raw.githubusercontent.com/Rigava/Load-Nifty-Data/main/stock_dfs_updated/{}.csv".format(files)
        download = requests.get(url).content
        data = pd.read_csv(io.StringIO(download.decode('utf-8')))   
        df=data.copy()
 
        if len(df) > 0:
            # Calculate crossover, MACD, and RSI indicators
            df["MA_fast"] = ta.sma(df["Close"], length =fast).round(1)
            df["MA_slow"] = ta.sma(df["Close"], length =slow).round(1)
            # df["MACD"],_,_ = ta.macd(df["Close"], fast=fast, slow=slow, signal=9)
            df["RSI"] = ta.rsi(df["Close"], lentgh =rsi_period).round(1)
            # st.write(files)
            # st.dataframe(df.tail(5))
            # Determine buy or sell recommendation based on strategy
            if df["MA_fast"].iloc[-1] > df["MA_slow"].iloc[-1] and df["RSI"].iloc[-1] < rsi_low:
                # and df["MACD"].iloc[-1] > 0
                Buy.append(files)
            elif df["MA_fast"].iloc[-1] < df["MA_slow"].iloc[-1] and df["RSI"].iloc[-1] > rsi_high:
                # and df["MACD"].iloc[-1] < 0
                Sell.append(files)
            else:
                Hold.append(files)            
    # Display stock data and recommendation
    st.write("List of stock recommended for Buy",Buy)
    st.write("List of stock recommended for Sell",Sell)
    st.write("Lets plot the moving averages with closing price")
    # Plotly graph for visualization
    ticker_choice = tickers
    symbol = st.selectbox("Select a stock to view moving average crossover",ticker_choice)
    ticker = symbol+'.NS'
    df = yfinance.Ticker(ticker).history(period="1y")
    df["MA_fast"] = ta.sma(df["Close"], length =fast).round(1)
    df["MA_slow"] = ta.sma(df["Close"], length =slow).round(1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_fast'], name='fast', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_slow'], name='slow', line=dict(color='blue')))
    fig.update_xaxes(type='category')
    fig.update_layout(height=800)
    st.plotly_chart(fig,use_container_width=True)

## Dashboard 4 STRATEGY----BUY ABOVE RSI 30 AND SELL BELOW 70
if dashboard == "RSI Strategy":
    ticker_choice = tickers
    symbol = st.selectbox("Select a stock for the strategy",ticker_choice)
    # Download historical data
    ticker = symbol+'.NS'
    df = yfinance.download(ticker, start="2020-01-01", end=None)
    # Calculate RSI
    df['RSI'] = ta.rsi(df['Close'],length=14)
    # Calculate SMA 200
    df['ma'] = ta.sma(df['Close'], length=10)
    # Initialize variables and DataFrame
    position = None
    buy_price = 0
    cumulative_profit = 0
    winning_trades = 0
    total_trades = 0
    trades_df = pd.DataFrame(columns=['Date', 'Price', 'Action'])
    #Parameter settings by user
    lower_bound=st.sidebar.slider("RSI low", min_value=1, max_value=100, value=30, step=1)
    upper_bound=st.sidebar.slider("RSI high", min_value=1, max_value=100, value=70, step=1)
    # Iterate over the data
    for i in range(1, len(df)):
        if df['RSI'][i - 1] < lower_bound and df['RSI'][i] > lower_bound :
            # Buy signal
            if position is None:
                position = 'buy'
                buy_price = df['Close'][i]
                trades_df = trades_df.append({'Date': df.index[i], 'Price': buy_price, 'Action': 'Buy'}, ignore_index=True)
    #             print("Buy at:", buy_price)
        elif df['RSI'][i - 1] > upper_bound and df['RSI'][i] < upper_bound:
            # Sell signal
            if position == 'buy':
                sell_price = df['Close'][i]
                profit = sell_price - buy_price
                cumulative_profit += profit
                total_trades += 1
                if profit > 0:
                    winning_trades += 1
                trades_df = trades_df.append({'Date': df.index[i], 'Price': sell_price, 'Action': 'Sell'}, ignore_index=True)
                position = None
    #             print("Sell at:", sell_price)
    #             print("Profit:", profit)
    # Calculate metrics
    winning_ratio = winning_trades / total_trades if total_trades > 0 else 0
    # Print metrics
    st.write("Cumulative Profit:", cumulative_profit)
    st.write("Total Trades:", total_trades)
    st.write("Winning Ratio:", winning_ratio)
    # Plotting the trades
    fig = go.Figure(data=[go.Scatter(x=df.index, y=df['Close'], name='Close'),
                        go.Scatter(x=trades_df['Date'], y=trades_df['Price'], mode='markers',
                                    marker=dict(color=trades_df['Action'].map({'Buy': 'green', 'Sell': 'red'}),
                                                size=8),
                                    name='Trades')])
    fig.update_layout(height=800)
    st.plotly_chart(fig,use_container_width=True)
    st.write(trades_df)

## Dashboard 5 STRATEGY----BUY Closing price ABOVE MA10 AND SELL BELOW MA50
if dashboard == "Moving Average Strategy":
    ticker_choice = tickers
    symbol = st.selectbox("Select a stock for the MA strategy",ticker_choice)
    # Download historical data
    ticker = symbol+'.NS'
    # df = yfinance.Ticker(ticker).history(period="1y")
    df = yfinance.download(ticker, start="2020-01-01", end=None)
    #Parameter settings by user
    fast=st.sidebar.slider("MA fast", min_value=1, max_value=100, value=10, step=1)
    slow=st.sidebar.slider("MA slow", min_value=1, max_value=200, value=50, step=1)
    # Calculate RSI
    df['RSI'] = ta.rsi(df['Close'],length=14)
    # Calculate SMA 10 and SMA 50
    df['fast'] = ta.sma(df['Close'], length=fast)
    df['slow'] = ta.sma(df['Close'], length=slow)
    # Initialize variables and DataFrame
    position = None
    buy_price = 0
    cumulative_profit = 0
    winning_trades = 0
    total_trades = 0
    trades_df = pd.DataFrame(columns=['Date', 'Price', 'Action'])

    # Iterate over the data
    for i in range(1, len(df)):
        if df['Close'][i]>df['fast'][i] and df['fast'][i] < df['slow'][i] :
            # Buy signal
            if position is None:
                position = 'buy'
                buy_price = df['Open'][i+1]
                trades_df = trades_df.append({'Date': df.index[i], 'Price': buy_price, 'Action': 'Buy'}, ignore_index=True)
        elif df['fast'][i - 1] > df['slow'][i-1] and df['fast'][i] < df['slow'][i] :
            # Sell signal
            if position == 'buy':
                sell_price = df['Open'][i+1]
                profit = sell_price - buy_price
                cumulative_profit += profit
                total_trades += 1
                if profit > 0:
                    winning_trades += 1
                trades_df = trades_df.append({'Date': df.index[i], 'Price': sell_price, 'Action': 'Sell'}, ignore_index=True)
                position = None
    # Calculate metrics
    winning_ratio = winning_trades / total_trades if total_trades > 0 else 0
    # Print metrics
    st.write("Cumulative Profit:", cumulative_profit)
    st.write("Total Trades:", total_trades)
    st.write("Winning Ratio:", winning_ratio)
    # Plotting the trades
    fig = go.Figure(data=[go.Scatter(x=df.index, y=df['Close'], name='Close'),
                          go.Scatter(x=df.index, y=df['slow'], name='slow', line=dict(color='black')),
                          go.Scatter(x=df.index, y=df['fast'], name='fast', line=dict(color='red')),
                          go.Scatter(x=trades_df['Date'], y=trades_df['Price'], mode='markers',
                                    marker=dict(color=trades_df['Action'].map({'Buy': 'green', 'Sell': 'red'}),
                                                size=8),name='Trades')])
    fig.update_layout(height=800)
    st.plotly_chart(fig,use_container_width=True)
    st.write(trades_df)

## Dashboard 6 STRATEGY----BUY Closing price ABOVE MA200 & RSI below 30 ; SELL RSI below 40
import numpy as np
def taCalc(df):
    df['RSI'] = ta.rsi(df['Close'],length=14)
    df['SMA200'] = ta.sma(df['Close'], length=200)
    df['Signal'] = np.where((df['Close']>df.SMA200) & (df['RSI']<30),1,0)
    return df
def getactualTrades(df):
    Buy_dates=[]
    Sell_dates=[]
    Buy_price=[]
    Sell_price=[]
    for i in range(len(df) - 11):
        # if the signal=1
        if df.Signal.iloc[i]: 
            # buy on the next date of signal=1
            Buy_dates.append(df.iloc[i+1].name) 
            Buy_price.append(df.iloc[i+1].Open)
            # Start looping in the subsequent 10 rows to check if the rsi is above 40, else we sell above 10 days
            for j in range(1,11):
                # if rsi is above 40 anytime in next 10 days
                if df['RSI'].iloc[i+j]>40:
                    #sell on the very next date of the signal
                    Sell_dates.append(df.iloc[i+j+1].name)
                    Sell_price.append(df.iloc[i+j+1].Open)
                    #if the rsi is greater than 40 we break from this loop
                    break
                # else if rsi was never above 40 then we sell above 10 days
                elif j == 10:
                    Sell_dates.append(df.iloc[i+j+1].name)
                    Sell_price.append(df.iloc[i+j+1].Open)
    frame = pd.DataFrame({'Buying_Dates':Buy_dates,'Selling_Dates':Sell_dates,'EntryPrice':Buy_price,'ExitPrice':Sell_price})
    # To Remove the overlapping trades we are shifting the selling dates by 1 row in orginal frame and filtering them out
    actualTrades=frame[frame.Buying_Dates>frame.Selling_Dates.shift(1)]
    #Taking the first datapoint from the frame and appending to actual Trades
    actualTrades = frame[:1].append(actualTrades)
    return actualTrades

if dashboard == "RSI SMA Strategy":
    ticker_choice = tickers
    symbol = st.selectbox("Select a stock for the MA strategy",ticker_choice)
    # Download historical data
    ticker = symbol+'.NS'
    df = yfinance.download(ticker, start="2015-01-01", end=None)
    df = taCalc(df)
    st.write(df)
    #To store the buy and sell dates
    actualTrades = getactualTrades(df)
    actualTrades['profit'] = actualTrades['ExitPrice'] - actualTrades['EntryPrice']
    actualTrades['winTrade'] = actualTrades['profit'].apply(lambda x: 1 if x>0 else 0)
    actualTrades['Trades'] = 1      
    st.write(actualTrades)
    # Calculate metrics
    TotalTrades = actualTrades['Trades'].sum()
    WinTrades = actualTrades['winTrade'].sum()
    winning_ratio = WinTrades / TotalTrades
    TotalProfit = actualTrades['profit'].sum()
    # Print metrics
    st.write("Profit Generated:", TotalProfit)
    st.write("Total Trades:", TotalTrades)
    st.write("Winning Ratio:", winning_ratio)
    #Here we are implementing the strategy to all Nifty50 stocks
    matrixProfits = []
    avgsMatrixProfits=[]
    TickerIndex =pd.DataFrame()
    tickers_yahoo = [i +'.NS' for i in tickers]
    for files in tickers:
       url = "https://raw.githubusercontent.com/Rigava/Load-Nifty-Data/main/stock_dfs_updated/{}.csv".format(files)
       download = requests.get(url).content
       df = pd.read_csv(io.StringIO(download.decode('utf-8')))    
       df = taCalc(df)
       actualTrades = getactualTrades(df)
       relProfits = (df.loc[actualTrades.Selling_Dates].Open.values - df.loc[actualTrades.Buying_Dates].Open.values)/df.loc[actualTrades.Buying_Dates].Open.values
       matrixProfits.append(relProfits)
    for i in matrixProfits:
        if len(i)>0:
            avgsMatrixProfits.append(i.mean())
    stockwithmaximumprofit = tickers[avgsMatrixProfits.index(max(avgsMatrixProfits))]
    st.write(stockwithmaximumprofit)
            
    
## CANDLE VIEW FOR ALL DASHBOARD....#Finally the below file settings is for plotting the candle stick chart as a good to have in the analysis
ticker_choice = tickers
candle_symbol = st.sidebar.selectbox("Select a stock to view in candle stick format",ticker_choice)
st.write(f"Below is the candle stick chart of {candle_symbol}")
ticker = candle_symbol+'.NS'
df = yfinance.download(ticker, start="2020-01-01", end=None)
# st.write(df.dtypes)
# st.write(df)
# df = chart_df[['Date','OpenPrice','HighPrice','LowPrice','ClosePrice','TotalTradedQuantity']]
# df.rename(columns={df.columns[0]:"Date",df.columns[1]:"Open",df.columns[2]:"High",df.columns[3]:"Low",df.columns[4]:"Close",df.columns[5]:"Volume"},inplace=True)
fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
fig.update_xaxes(type='category')
fig.update_layout(height=800)
st.plotly_chart(fig,use_container_width=True)
   