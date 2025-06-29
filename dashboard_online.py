import streamlit as st
import io
import pandas as pd
import plotly.graph_objects as go
import pickle
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance
# import pandas_ta as ta
import numpy as np

def AddRSIIndicators(df):
    df['priceChange']=df['Close']-df['Close'].shift(1)
    df=df.dropna()
    df['Upmove']=df['priceChange'].apply(lambda x: x if x>0 else 0)
    df['Downmove']=df['priceChange'].apply(lambda x: abs(x) if x<0 else 0)
    df['avgUp']=df['Upmove'].ewm(span=27).mean()
    df['avgDown']=df['Downmove'].ewm(span=27).mean()
    df['RS']= df['avgUp']/df['avgDown']
    df['RSI']= df['RS'].apply(lambda x: 100-(100/(x+1)))
    print('RSI indicators added')
    return df
def AddSMAIndicators(df,fast,slow):
    df['SMA10']=df.Close.rolling(fast).mean()
    df['SMA50']=df.Close.rolling(slow).mean()
    df['buySignal']=np.where(df.SMA10>df.SMA50,1,0)
    df['sellSignal']=np.where(df.SMA10<df.SMA50,1,0)
    df['Decision Buy GC']= df.buySignal.diff()
    df['Decision Sell GC']= df.sellSignal.diff()
    print('SMA indicators added')
    return df
def MACDIndicator(df):
    df['EMA12']= df.Close.ewm(span=12).mean()
    df['EMA26']= df.Close.ewm(span=26).mean()
    df['MACD'] = df.EMA12 - df.EMA26
    df['Signal'] = df.MACD.ewm(span=9).mean()
    df['MACD_diff']=df.MACD - df.Signal
    df.loc[(df['MACD_diff']>0) & (df.MACD_diff.shift(1)<0),'Decision MACD']='Buy'
    df.loc[(df['MACD_diff']<0) & (df.MACD_diff.shift(1)>0),'Decision MACD']='Sell'
    df.dropna()
    print('MACD indicators added')
    return df
#For Breakouts
def is_consolidating(data):
    recent_candles = data[-15:]
    max_close = recent_candles['Close'].max()
    min_close = recent_candles['Close'].min()
    # print('the max close was {} and the min close was {}'.format(max_close,min_close))
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
#For adding the sma1 and sma2
def ma_calc(data,n,m):
    data['sma_1'] = data['Close'].rolling(window=n).mean()
    data['sma_2'] = data['Close'].rolling(window=m).mean()
    data['price'] = data['Close'].shift(-1)
def vectorized(df,n,m):
    ma_calc(df,n,m)
    first_buy = pd.Series(df.index == (df.sma_1>df.sma_2).idxmax(),index=df.index)
    real_signal = first_buy | (df.sma_1>df.sma_2).diff()
    trades = df[real_signal]
    if len(trades)%2!=0:
        mtm = df.tail(1).copy()
        mtm.price = mtm.Close
        trades =pd.concat([trades,mtm])
    profits = trades.price.diff()[1::2] / trades.price[0::2].values
    gain = (profits + 1).prod()
    return gain 
def slice_df(price_df,symbol):
    sliced = price_df.copy()
    sliced = price_df[price_df.columns[price_df.columns.get_level_values(1)==symbol]]
    sliced.columns = sliced.columns.droplevel(1)
    return sliced   

st.set_page_config(page_title="Nifty 50 Universe", page_icon=":bar_chart:", layout="wide")
st.title('NIFTY 50 Universe BY JOSH@I')

with open("nifty50tickers.pickle",'rb') as f:
    tickers=pickle.load(f)

index = ["^NSEI","^NSEBANK","^CNXIT","^CNXPHARMA","^CNXMETAL","^CNXREALTY","^CNXPSUBANK","^CNXINFRA","^CNXENERGY"]
dashboard = st.sidebar.selectbox("select analysis",["Data","Stock Shortlist","Back Testing","Stock Crossover","Index Squeeze","Donchian Channel"])


## Dashboard 0
if dashboard == "Data":
    fast = st.sidebar.slider("Fast Period", min_value=5, max_value=50, value=10, step=1)
    slow = st.sidebar.slider("Slow Period", min_value=10, max_value=200, value=50, step=1)
    symbol = st.sidebar.selectbox("Select stock to pull data", tickers)
    st.subheader(f"Stocks Price Update of {symbol}")
    if symbol:
        try:
            ticker = symbol+'.NS'
            stock_data = yfinance.Ticker(ticker).history(period="5y")
            # stock_data.index = stock_data.index.astype('datetime64[s]')
            latest_price = stock_data['Close'].iloc[-1].round(1)

            # Plotting historical price movement
            st.subheader("Historical Price Movement for Last 5 years")
            plt.figure(figsize=(10, 6))
            plt.plot(stock_data.index, stock_data['Close'])
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title('Price Movement')
            plt.xticks(rotation=45)
            st.pyplot(plt)
            # Add indicators
            stock_data = AddRSIIndicators(stock_data)
            stock_data = AddSMAIndicators(stock_data,fast,slow)
            stock_data = MACDIndicator(stock_data)
            #Filter data for the last 1 years
            stock_data = stock_data[(stock_data.index.year>2023)]
            latest_rsi = stock_data['RSI'].iloc[-1].round(1)
            st.success(f"The latest price is: {latest_price} and the rsi is {latest_rsi}")
            ### MACD PLOT
            st.markdown(f"MACD for {symbol}")
            fig =plt.figure(figsize=(12, 6))
            ax1=fig.add_subplot(1,2,1)
            ax2=fig.add_subplot(1,2,2)
            ax1.plot(stock_data.index, stock_data['Close'])
            ax1.set_xlabel('Date')
            ax1.tick_params(axis='x', rotation=45)
            ax1.set_ylabel('Price')
            ax1.set_title('Price Movement')
            
            ax2.plot(stock_data.Signal,color='red')
            ax2.plot(stock_data.MACD,color='green')
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_title('MACD crossover')
            st.pyplot(plt)

            # RSI PLOT
            fig, ax3 = plt.subplots(figsize=(10, 6)) 
            # Plot the stock price on the left y-axis (primary axis)
            ax3.plot(stock_data.index, stock_data['Close'], label='Price', color='blue')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Price (INR)', color='blue')
            ax3.tick_params(axis='y', labelcolor='blue')  
            # Create a second y-axis for the RSI (secondary axis)
            ax4 = ax3.twinx()
            ax4.plot(stock_data.index, stock_data['RSI'], label='RSI', color='orange')
            ax4.set_ylabel('RSI', color='orange')
            ax4.tick_params(axis='y', labelcolor='orange')  
            # Add RSI overbought/oversold levels
            ax4.axhline(70, color='red', linestyle='--', label='Overbought (70)')
            ax4.axhline(30, color='green', linestyle='--', label='Oversold (30)')
            # Add titles and legends
            plt.title(f'{symbol} Price and RSI')
            fig.tight_layout()  # Adjust layout to make room for both y-axes
            ax3.legend(loc='upper left')
            ax4.legend(loc='upper right')
            st.pyplot(plt)

            # Display stock data
            with st.expander("🔍 Data Preview"):
                st.dataframe(stock_data)
           
            # Export data as CSV
            st.subheader("Export Data")
            if st.button("Export as CSV"):
                st.write("Exporting stock data as CSV...")
                stock_data.to_csv(f"{symbol}_data.csv", index=False)
                st.success("Stock data exported successfully!")    
        except Exception as e:
            st.error("Error occurred while fetching stock data.")
            st.error(e)



## -------------------------------------------------------------------------Dashboard 1 SHORTLIST ------------------------------------------------------------------------------------------------------------------
if dashboard == "Stock Shortlist":
    # User input for strategy parameters
    shortlist_option = st.sidebar.selectbox("select strategy",["MACD","RSI","Breakout"])
    rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=50, value=14, step=1)
    rsi_low = st.sidebar.slider("RSI low for buy", min_value=1, max_value=100, value=30, step=1)
    rsi_high = st.sidebar.slider("RSI high for sell", min_value=1, max_value=100, value=70, step=1) 
    url = "https://raw.githubusercontent.com/Rigava/Load-Nifty-Data/main/stock_dfs_updated/{}.csv".format("RELIANCE")
    download = requests.get(url).content
    data = pd.read_csv(io.StringIO(download.decode('utf-8')))   
    latest_date = data['Date'].iloc[-1]
    st.info(f"The latest data is from {latest_date}")  
    # User Button for starting the Algo
    if st.button("Shortlist", use_container_width=True):
        Buy = []
        Sell = []
        Hold = []
        framelist = []

        # Iterate over stock data to find stock with MACD crossover and rsi signal depending upon the selected strategy
        for files in tickers:
            url = "https://raw.githubusercontent.com/Rigava/Load-Nifty-Data/main/stock_dfs_updated/{}.csv".format(files)
            download = requests.get(url).content
            data = pd.read_csv(io.StringIO(download.decode('utf-8')))   
            df=data.copy()

            if len(df) > 0:
                # Calculate indicators
                df = AddRSIIndicators(df)
                df = MACDIndicator(df)
                framelist.append(df)
                # Determine buy or sell recommendation based on last two rows of the data to provide buy & sell signals
                if shortlist_option=="MACD":                
                    if df['Decision MACD'].iloc[-1]=='Buy':    
                        Buy.append(files)
                    elif df['Decision MACD'].iloc[-1]=='Sell':
                        Sell.append(files)
                    else:
                        Hold.append(files)  
                
                if shortlist_option=="RSI":
                    if df["RSI"].iloc[-1] > rsi_low and df["RSI"].iloc[-2] < rsi_low: 
                        Buy.append(files)
                    elif df["RSI"].iloc[-1] < rsi_high and df["RSI"].iloc[-2] > rsi_high:
                        Sell.append(files)
                    else:
                        Hold.append(files)  
                if shortlist_option=="Breakout":
                    if is_breakingout(df):
                        Buy.append(files)
     
        # Display stock data and recommendation
        st.write(":blue[List of stock with buy signal]",Buy)
        st.write(":blue[List of stock with sell signal]",Sell)
   
        bucket = Buy + Sell
        # Display stock Chart for Buy and Sell using yahoofinance as the source to fetch latest data
        for symbol in bucket:
            try:
                # Fetch the data using yfinance lib
                ticker = symbol+'.NS'
                stock_data = yfinance.Ticker(ticker).history(period="1y")
                
                latest_price = stock_data['Close'].iloc[-1].round(1)
                stock_data = AddRSIIndicators(stock_data)
                stock_data = MACDIndicator(stock_data)
                latest_rsi = stock_data['RSI'].iloc[-1].round(1)
                st.subheader(symbol)
                st.info(f"The latest price is: {latest_price} and the rsi is {latest_rsi}")
                # Plotting historical price movement based on selected strategy
                if shortlist_option=="MACD":     
                    st.markdown(f"Historical price movement of {symbol}")
                    fig =plt.figure(figsize=(12, 6))
                    ax1=fig.add_subplot(1,2,1)
                    ax2=fig.add_subplot(1,2,2)
                    ax1.plot(stock_data.index, stock_data['Close'])
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Price')
                    ax1.set_title('Price Movement')
                    # ax1.set_xticks(rotation=45)
                    ax2.plot(stock_data.Signal,color='red')
                    ax2.plot(stock_data.MACD,color='green')
                    st.pyplot(plt)
                if shortlist_option=="RSI":
                    # Create a figure and axis
                    fig, ax1 = plt.subplots(figsize=(10, 6)) 
                    # Plot the stock price on the left y-axis (primary axis)
                    ax1.plot(stock_data.index, stock_data['Close'], label='Price', color='blue')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Price (INR)', color='blue')
                    ax1.tick_params(axis='y', labelcolor='blue')  
                    # Create a second y-axis for the RSI (secondary axis)
                    ax2 = ax1.twinx()
                    ax2.plot(stock_data.index, stock_data['RSI'], label='RSI', color='orange')
                    ax2.set_ylabel('RSI', color='orange')
                    ax2.tick_params(axis='y', labelcolor='orange')  
                    # Add RSI overbought/oversold levels
                    ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
                    ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
                    # Add titles and legends
                    plt.title(f'{ticker} Price and RSI')
                    fig.tight_layout()  # Adjust layout to make room for both y-axes
                    ax1.legend(loc='upper left')
                    ax2.legend(loc='upper right')
                    st.pyplot(plt)
       

            except Exception as e:
                st.error("Error occurred while fetching stock data.")
                st.error(e)



## -------------------------------------------------------------------------Dashboard 2 BACK TESTING----------------------------------------------------------------------------------------------------------------
if dashboard == "Back Testing":
    # Select Stock for Backtesting the crossover strategy
    fast = st.sidebar.slider("Fast Period", min_value=1, max_value=50, value=10, step=1)
    slow = st.sidebar.slider("Slow Period", min_value=10, max_value=200, value=50, step=1)
    symbol = st.selectbox("Select a stock to view thecumulative profits from trading moving average crossover strategy",tickers)
    ticker = symbol+'.NS'
    df = yfinance.Ticker(ticker).history(period="5y")
    # Add MA indicators
    df = AddSMAIndicators(df,fast,slow)
    df = AddRSIIndicators(df)
    df['price'] = df['Close'].shift(-1)
    st.write(df)
    #Below crossover function is used to backtest the trade
    def tradedf(df1):
        df1.reset_index(inplace=True)
        Flag = False
        bi,si=[],[]
        Buy,Sell =[],[]
        Buyp,Sellp = [],[]
       
        marker_df = pd.DataFrame(columns=['Date','Action','Price'])
        for i,row in df1.iterrows():
            if not Flag:
                if df1.SMA10.iloc[i] > df1.SMA50.iloc[i]  and df1.SMA10.iloc[i-1] < df1.SMA50.iloc[i-1]:
                    buyprice = row.Close
                    buydate =row.Date
                    Flag=True
                    Buy.append(buydate)
                    Buyp.append(buyprice) 
                    bi.append(i)                   
                   
            if Flag:
                if df1.SMA10.iloc[i] < df1.SMA50.iloc[i]  and df1.SMA10.iloc[i-1] > df1.SMA50.iloc[i-1]:
                    sellprice = row.Close 
                    selldate =row.Date
                    Sell.append(selldate)
                    Sellp.append(sellprice)       
                    si.append(i)      
                    Flag=False 
        buyframe = pd.DataFrame({'BuyDate':Buy,'BuyPrice':Buyp})
        sellframe = pd.DataFrame({'SellDate':Sell,'SellPrice':Sellp})
        tradedf=pd.concat([buyframe,sellframe],axis=1)
        tradedf.dropna()
        tradedf['profit']=tradedf['SellPrice']-tradedf['BuyPrice']
        totalProfit = tradedf['profit'].sum().round(2)
        st.write(f"Total profit from the moving average strategy is {totalProfit}")

        # Below method uses vectorized approach to find the trades and calculate the profits return
        first_buy = pd.Series(df.index == (df.SMA10>df.SMA50).idxmax(),index=df.index)
        real_signal = first_buy | (df.SMA10>df.SMA50).diff()
        trades = df[real_signal]
        if len(trades)%2 != 0:
            mtm = df.tail(1).copy()
            mtm.price = mtm.Close
            trades = pd.concat([trades,mtm])
        profits = trades.price.diff()[1::2] / trades.price[0::2].values
        gain = (profits + 1).prod()
        st.dataframe(trades)
        st.write(f"Strategu return from the moving average strategy is {gain}")

        st.dataframe(tradedf)

        return tradedf,bi,si
    marker_df,bi,si = tradedf(df)
    # Plotly graph for visualization
    st.write("The below plot shows the moving averages with closing price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.Date, y=df['Close'], name='Close', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.Date, y=df['SMA10'], name='fast', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.Date, y=df['SMA50'], name='slow', line=dict(color='white')))

    fig.add_trace(go.Scatter(x=df.iloc[bi].Date, y=df.iloc[bi].Close, name='buySignal',mode='markers' ,marker=dict(color='green',size=8))) 
    fig.add_trace(go.Scatter(x=df.iloc[si].Date, y=df.iloc[si].Close, name='sellSignal',mode='markers' ,marker=dict(color='red',size=8)))
    fig.update_xaxes(type='category')
    fig.update_layout(height=800)
    st.plotly_chart(fig,use_container_width=True)

# ## Dashboard 4
if dashboard == "Stock Crossover":
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-07-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-06-29"))
    st.write("SMA crossover (SMA50 & SMA100) best return algo Stock - WIP")
    results=[]
    yf_tickers= []
    try:
        for stok in tickers:
            ticker = stok+'.NS'
            yf_tickers.append(ticker)
    except Exception as e:
        st.error(f"Could not fetch data for {stok} from Yahoo Finance. {e}")
    
    stock_data = yfinance.download(yf_tickers ,start=start_date, end=end_date)
    for sym in yf_tickers:
            subdf = slice_df(stock_data,sym)
            results.append(vectorized(subdf,50,100))

    df_profits=pd.DataFrame({'profits':results},index=tickers)
    n_df = df_profits.sort_values(by='profits',ascending=False)
    st.dataframe(n_df)

## Dashboard 5
def in_squeeze(df):
    return df['lower_band'] > df['lower_keltner'] and df['upper_band'] < df['upper_keltner']
# Vectorized function to detect changes in interaction
def detect_individual_signals(interactions, interaction_type):
    """Identifies signal points where a specific interaction changes."""
    prev_interaction = np.roll(interactions, shift=1)
    signals = np.where((interactions == interaction_type) & (prev_interaction != interaction_type), 1, 0)
    signals[0] = 0  # First row has no previous value to compare
    return signals
if dashboard == "Index Squeeze":
    symbol = st.sidebar.selectbox("Select an Index",index)
    # df = yfinance.Ticker(symbol).history(period="5y")
 
    df = yfinance.download(symbol,group_by="Ticker",start="2010-01-01", end=None)
    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    df.index = df.index.astype('datetime64[s]')

    df['20sma'] = df['Close'].rolling(window=20).mean()
    df['stddev'] = df['Close'].rolling(window=20).std()
    df['lower_band'] = df['20sma'] - (2 * df['stddev'])
    df['upper_band'] = df['20sma'] + (2 * df['stddev'])

    df['TR'] = abs(df['High'] - df['Low'])
    df['ATR'] = df['TR'].rolling(window=20).mean()
    df['lower_keltner'] = df['20sma'] - (df['ATR'] * 1.5)
    df['upper_keltner'] = df['20sma'] + (df['ATR'] * 1.5)
    squeeze = (df['lower_band'] > df['lower_keltner']) & (df['upper_band'] < df['upper_keltner'])
    df['interaction'] = np.where(squeeze, "Squeeze", "No Squeeze")
    df['squeeze_on'] = detect_individual_signals(df['interaction'].values, "Squeeze")
    squeeze_data = df[df['squeeze_on'] == 1]

    st.write(squeeze_data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price',line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df.lower_band, mode='lines', name='lower band',line=dict(color="red")))
    fig.add_trace(go.Scatter(x=df.index, y=df.upper_band, mode='lines', name='upper band',line=dict(color="green")))
    fig.add_trace(go.Scatter(x=df.index, y=df.lower_keltner, mode='lines', name='lower ketler',line=dict(color="grey")))
    fig.add_trace(go.Scatter(x=df.index, y=df.upper_keltner, mode='lines', name='upper ketler',line=dict(color="grey")))
    fig.add_trace(go.Scatter(x=squeeze_data.index, y=squeeze_data.Close, mode='markers', name='Signal', marker=dict(color="pink", size=10)))
    fig.update_layout(title='Squeeze Strategy', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
#---------------------------------------------------------------NEUROTRADER---------------------------------------------------
# Donchian Channel Breakout Strategy
# Dashboard 6
def donchian_breakout_data(ohlc: pd.DataFrame, lookback: int):
    # input df is assumed to have a 'close' column
    ohlc['Upper'] = ohlc['Close'].rolling(lookback - 1).max().shift(1)
    ohlc['Lower'] = ohlc['Close'].rolling(lookback - 1).min().shift(1)
    penetration_upper = ohlc['Close'] > ohlc['Upper']
    penetration_lower = ohlc['Close'] < ohlc['Lower']
    ohlc['signal'] = np.nan
    ohlc.loc[penetration_upper, 'signal'] = 1 # for long entry
    ohlc.loc[penetration_lower, 'signal'] = -1 # for short entry
    ohlc['signal'] = ohlc['signal'].ffill()
    ohlc['price'] = ohlc['Open'].shift(-1)
    ohlc['pos_change'] = ohlc['signal'].diff()
    ohlc['benchmark_return']=ohlc['Close'].pct_change()
    ohlc['benchmark_euity'] = (1+ohlc['benchmark_return']).cumprod()
    #signal for optimisation
    upper = ohlc['Close'].rolling(lookback - 1).max().shift(1)
    lower = ohlc['Close'].rolling(lookback - 1).min().shift(1)
    signal = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
    signal.loc[ohlc['Close'] > upper] = 1
    signal.loc[ohlc['Close'] < lower] = -1
    signal = signal.ffill()
    return ohlc,signal
def optimize_donchian(ohlc: pd.DataFrame):

    best_pf = 0
    best_lookback = -1
    r = np.log(ohlc['Close']).diff().shift(-1)
    for lookback in range(12, 169):
        ohlc_date,signal = donchian_breakout_data(ohlc, lookback)
        sig_rets = signal * r
        sig_pf = sig_rets[sig_rets > 0].sum() / sig_rets[sig_rets < 0].abs().sum()

        if sig_pf > best_pf:
            best_pf = sig_pf
            best_lookback = lookback

    return best_lookback, best_pf
if dashboard == "Donchian Channel":
    st.write("Donchian Channel is a trend-following strategy that uses the highest high and lowest low over a specified period to identify breakout points. It generates buy and sell signals based on the price crossing these levels.")
    symbol = st.sidebar.selectbox("Select an Index",index) 
    df = yfinance.download(symbol,group_by="Ticker",start="2010-01-01", end=None)
    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    df.index = df.index.astype('datetime64[s]')

    best_lookback, best_real_pf = optimize_donchian(df)
     # Best lookback = 19, best_real_pf = 1.08
    st.write(f"Best lookback = {best_lookback}, best_real_pf = {best_real_pf}")
        #Donchian breakout data
    donchian_data,signals = donchian_breakout_data(df, best_lookback)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=donchian_data.index, y=donchian_data['Close'], mode='lines', name='Close Price',line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=donchian_data.index, y=donchian_data['Upper'], mode='lines', name='Upper Band',line=dict(color="green")))
    fig.add_trace(go.Scatter(x=donchian_data.index, y=donchian_data['Lower'], mode='lines', name='Lower Band',line=dict(color="red")))   
    fig.update_layout(title='Donchian Strategy', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    df['r'] = np.log(df['Close']).diff().shift(-1)
    df['donch_r'] = df['r'] * signals

    plt.style.use("dark_background")
    df['donch_r'].cumsum().plot(color='red')
    plt.title("In-Sample Donchian Breakout")
    plt.ylabel('Cumulative Log Return')
    st.pyplot(plt)