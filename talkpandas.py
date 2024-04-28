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
    choice = st.selectbox("Select a default files",["Finance","Country","Upload my csv"])
    if choice =="Finance":
        url = "https://raw.githubusercontent.com/Rigava/DataRepo/main/yesbank.csv"
        download = requests.get(url).content
        df = pd.read_csv(io.StringIO(download.decode('utf-8')))
        st.write(df)   
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


