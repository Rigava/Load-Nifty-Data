import streamlit as st
import io
import pandas as pd
import plotly.graph_objects as go
import pickle
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance
import pandas_ta as ta

st.title('NIFTY 50 STOCK DASHBOARD')

with open("nifty50tickers.pickle",'rb') as f:
    tickers=pickle.load(f)

dashboard = st.sidebar.selectbox("select analysis",["Data","Crossover & RSI Shortlist"])

## Dashboard 0
if dashboard == "Data":
    symbol = st.sidebar.selectbox("Select stock to pull data", tickers)
    st.title(symbol+" Stocks Price Update")
    if symbol:
        try:
            ticker = symbol+'.NS'
            stock_data = yfinance.Ticker(ticker).history(period="1y")
            latest_price = stock_data['Close'].iloc[-1].round(1)
            # print(stock_data,latest_price)
            
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



## Dashboard 1 CROSSOVERS
if dashboard == "Crossover & RSI Shortlist":
    # User input for strategy parameters
    fast = st.sidebar.slider("Fast Period", min_value=5, max_value=50, value=10, step=1)
    slow = st.sidebar.slider("Slow Period", min_value=10, max_value=200, value=50, step=1)
    rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=50, value=14, step=1)
    rsi_low = st.sidebar.slider("RSI low for buy", min_value=1, max_value=100, value=30, step=1)
    rsi_high = st.sidebar.slider("RSI high for sell", min_value=1, max_value=100, value=70, step=1)
    Buy = []
    Sell = []
    Hold = []
    # Iterate over stock data to find stock with crossover and rsi signal
    for files in tickers:
        url = "https://raw.githubusercontent.com/Rigava/Load-Nifty-Data/main/stock_dfs_updated/{}.csv".format(files)
        download = requests.get(url).content
        data = pd.read_csv(io.StringIO(download.decode('utf-8')))   
        df=data.copy()
        if len(df) > 0:
            # Calculate crossover, MACD, and RSI indicators
            df["MA_fast"] = ta.sma(df["Close"], length =fast).round(1)
            df["MA_slow"] = ta.sma(df["Close"], length =slow).round(1)
            df["RSI"] = ta.rsi(df["Close"], lentgh =rsi_period).round(1)
            # Determine buy or sell recommendation based on last row of the data to provide buy signal
            if df["MA_fast"].iloc[-1] > df["MA_slow"].iloc[-1] and df["RSI"].iloc[-1] < rsi_low:
                # and df["MACD"].iloc[-1] > 0
                Buy.append(files)
            elif df["MA_fast"].iloc[-1] < df["MA_slow"].iloc[-1] and df["RSI"].iloc[-1] > rsi_high:
                # and df["MACD"].iloc[-1] < 0
                Sell.append(files)
            else:
                Hold.append(files)            
    # Display stock data and recommendation
    st.write("List of stock for with buy signal",Buy)
    st.write("List of stock with sell signal",Sell)
    st.write("The below plot shows the moving averages with closing price")
    # Select Stock for Backtesting the crossover strategy
    ticker_choice = tickers
    symbol = st.selectbox("Select a stock to view moving average crossover",ticker_choice)
    ticker = symbol+'.NS'
    df = yfinance.Ticker(ticker).history(period="5y")
    # Add MA indicators
    df["SMA10"] = ta.sma(df["Close"], length =fast).round(1)
    df["SMA50"] = ta.sma(df["Close"], length =slow).round(1)
    #Below crossover function is used to backtest the trade
    def tradedf(df1):
        df1.reset_index(inplace=True)
        Flag = False
        buyframe=pd.DataFrame(columns=['BuyDate','BuyPrice'])
        sellframe=pd.DataFrame(columns=['SellDate','SellPrice'])
        marker_df = pd.DataFrame(columns=['Date','Action','Price'])
        for i,row in df1.iterrows():
            if not Flag:
                if df1.SMA10.iloc[i] > df1.SMA50.iloc[i]  and df1.SMA10.iloc[i-1] < df1.SMA50.iloc[i-1]:
                    buyprice = row.Close
                    Flag=True
                    buyframe= buyframe.append({'BuyDate': row.Date,'BuyPrice': buyprice}, ignore_index=True)
                    marker_df = marker_df.append({'Date': row.Date, 'Action': 'Buy','Price':row.Close}, ignore_index=True)
            if Flag:
                if df1.SMA10.iloc[i] < df1.SMA50.iloc[i]  and df1.SMA10.iloc[i-1] > df1.SMA50.iloc[i-1]:
                    sellprice = row.Close              
                    sellframe= sellframe.append({'SellDate': row.Date,'SellPrice': sellprice}, ignore_index=True)
                    marker_df = marker_df.append({'Date': row.Date, 'Action': 'Sell','Price':row.Close}, ignore_index=True)
                    Flag=False 
        # st.dataframe(marker_df)

        tradedf=pd.concat([buyframe,sellframe],axis=1)
        tradedf.dropna()
        tradedf['profit']=tradedf['SellPrice']-tradedf['BuyPrice']
        totalProfit = tradedf['profit'].sum().round(2)

        st.write(f"Total profit from the moving average strategy is {totalProfit}")
        # st.dataframe(tradedf)
        
        return marker_df
    marker_df = tradedf(df)
    # Plotly graph for visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.Date, y=df['Close'], name='Close', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df.Date, y=df['SMA10'], name='fast', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.Date, y=df['SMA50'], name='slow', line=dict(color='blue')))

    fig.add_trace(go.Scatter(x=marker_df.Date, y=marker_df.Price, name='Trades',mode='markers' ,marker=dict(color=marker_df.Action.map({'Buy':'green','Sell':'red'}),size=8)))
    fig.update_xaxes(type='category')
    fig.update_layout(height=800)
    st.plotly_chart(fig,use_container_width=True)
    