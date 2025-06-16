import os
import streamlit as st
import pandas as pd
import io
import requests

from pandasai import SmartDataframe
from pandasai.llm import GooglePalm

GOOGLE_API_KEY = st.secrets.API_KEY

llm = GooglePalm(api_key=GOOGLE_API_KEY)

dashboard = st.sidebar.selectbox("select analysis",["Prompting","Sensex"])

if dashboard=="Prompting":
    st.title("Your Data Analysis Dashboard")
    choice = st.selectbox("Select a default files",["Finance","Country","Upload my csv"])
    if choice =="Finance":
        url = "https://raw.githubusercontent.com/Rigava/DataRepo/main/yesbank.csv"
        download = requests.get(url).content
        df = pd.read_csv(io.StringIO(download.decode('utf-8')))  
    elif choice =="Country":
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
        st.write("Some sample questions- Describe the data, dtypes of variables, shape of the data, any missing value, are there any duplicate rows, plot the graph, group the data by and calculate average ")
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
# ---------------------------------------------------Portfolio Theory-------------------------------------                
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

wiki ='https://en.wikipedia.org/wiki/BSE_SENSEX'
ticker =pd.read_html(wiki)[1].Symbol.to_list()
df =yf.download(ticker, start = '2023-01-01')['Close']

if dashboard=="Sensex":
    
    ret_df = np.log(df/df.shift(1))
    ret_df.dropna(inplace=True)
    st.write(ret_df)

    st.write("Correlation of the stocks return")
    plt.figure(figsize=(20, 10))
    plot = sns.heatmap(ret_df.corr(),annot=True)
    st.pyplot(plot.get_figure())
 
    #Prompting on stock returns dataframe
    sample_question = ['']

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

    # symbol = st.sidebar.selectbox("Select a stock to view its return",ticker)
    # st.write(f"Below is the stock return chart of {symbol}")
    # plt.figure(figsize=(20, 10))
    # plt.plot(ret_df.Date, ret_df.symbol)
    # st.pyplt(plt)
    


