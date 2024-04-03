import os
import streamlit as st
import pandas as pd
import io
import requests

from pandasai import SmartDataframe
from pandasai.llm import GooglePalm

GOOGLE_API_KEY = st.secrets.API_KEY

llm = GooglePalm(api_key=GOOGLE_API_KEY)

dashboard = st.sidebar.selectbox("select analysis",["Prompting","NSE"])

if dashboard=="Prompting":
    st.title("Your Data Analysis Dashboard")
    choice = st.selectbox("Select a default files",["Titanic","Country","Upload my csv"])
    if choice =="Titanic":
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        download = requests.get(url).content
        df = pd.read_csv(io.StringIO(download.decode('utf-8')))      
    if choice =="Country":
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/WorldDBTables/CountryTable.csv"
        download = requests.get(url).content
        df = pd.read_csv(io.StringIO(download.decode('utf-8')))            
    else:
        upload_csv = st.file_uploader("Upload a csv file for analysis", type =['csv'])
        df = pd.read_csv(upload_csv)
    # upload_csv = st.file_uploader("Upload a csv file for analysis", type =['csv'])
    if df is not None:
        # df = pd.read_csv(upload_csv)
        sdf = SmartDataframe(df,config={"llm":llm})
        st.dataframe(df.tail(3))

        prompt = st.text_area("Enter your query")
        if st.button("Generate"):
            if prompt:
                with st.spinner("Generating response..."):
                    response = sdf.chat(prompt)
                    st.success(response)

                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
            else:
                st.warning("Please enter another query")

import requests
from datetime import datetime
import matplotlib.pyplot as plt
from urllib.parse import quote
import pandas_ta as ta
import pickle
from nselib import capital_market
from unidecode import unidecode

with open("nifty50tickers.pickle",'rb') as f:
    tickers=pickle.load(f)
if dashboard=="NSE":
    st.title("Your NSE stock dashboard")
    # symbol_list = ["RELIANCE", "SBIN","TCS","INFY","HDFC","ITC","ASIANPAINT","AXISBANK","ADANIPORTS","BAJAJFINSV"]
    symbol = st.sidebar.selectbox("Select stock symbol", tickers)
    encoded_symbol=quote(symbol)

    st.title(symbol+" Stocks Price Update")

    # symbol = st.text_input("Enter stock symbol (e.g., SBIN, RELIANCE)")

    if symbol:

        try:
            data = capital_market.price_volume_and_deliverable_position_data(
                symbol="ITC",period='1M')
            # from_date='01-01-2023', to_date='01-04-2024'
            # stock_url='https://www.nseindia.com/api/historical/cm/equity?symbol={}'.format(encoded_symbol)
            # headers= {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36' ,
            # "accept-encoding": "gzip, deflate, br", "accept-language": "en-US,en;q=0.9"}
            # r = requests.get(stock_url, headers=headers).json()
            # data_values=[data for data in r['data']]
            # df=pd.DataFrame(data_values)
            st.dataframe(data)
            df = data[['Date','OpenPrice','HighPrice','LowPrice','ClosePrice','TotalTradedQuantity']]
            df = df.drop_duplicates(subset=['Date'],keep='first')
            df.rename(columns={df.columns[0]:"Date",df.columns[1]:"Open",df.columns[2]:"High",df.columns[3]:"Low",df.columns[4]:"Close",df.columns[5]:"Volume"},inplace=True)
            st.write("before datatype",df.dtypes)
            st.write(df)
            cols = df.select_dtypes(exclude=['float']).columns
            st.write("cols which are not float",cols)
            df['Date']=pd.to_datetime(df['Date'])
            for col in cols:
                if col == 'Date':
                    pass
                else:
                    df[col] = df[col].apply(lambda x: (unidecode(x).replace(',',''))).astype(float)
            df.set_index('Date',inplace=True)
            st.write("final data types",df.dtypes)
            latest_price = df['Close'].iloc[-1]
            st.success(f"The latest price is: {latest_price}")
            # Plotting historical price movement
            st.subheader("Historical Price Movement")
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['Close'])
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title('Price Movement')
            plt.xticks(rotation=45)
            st.pyplot(plt)
            

            # Export data as CSV
            st.subheader("Export Data")
            if st.button("Export as CSV"):
                st.write("Exporting stock data as CSV...")
                df.to_csv(f"{symbol}_data.csv", index=False)
                st.success("Stock data exported successfully!")     
            #Fetch the recommendation
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

                # Determine buy or sell recommendation based on strategy
                if df["MA_fast"].iloc[-1] > df["MA_slow"].iloc[-1] and df["RSI"].iloc[-1] < 45:
                    # and df["MACD"].iloc[-1] > 0
                    recommendation = "Buy"
                elif df["MA_fast"].iloc[-1] < df["MA_slow"].iloc[-1] and df["RSI"].iloc[-1] > 70:
                    # and df["MACD"].iloc[-1] < 0
                    recommendation = "Sell"
                else:
                    recommendation = "Hold"
                st.dataframe(df.tail(5))
                # Display stock data and recommendation
                st.subheader("Recommendation")
                st.write(f"The recommendation for {symbol} is: {recommendation}")       
        except Exception as e:
            st.error("Error occurred while fetching stock data.")
            st.error(e)
