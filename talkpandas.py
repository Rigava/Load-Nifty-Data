import os
import streamlit as st
import pandas as pd

from pandasai import SmartDataframe
from pandasai.llm import GooglePalm

GOOGLE_API_KEY = "AIzaSyAKEaaM7fWIErN3VbikjP_T5m0UfhBy5iE"

llm = GooglePalm(api_key=GOOGLE_API_KEY)

st.title("Your Data Analysis Dashboard")

upload_csv = st.file_uploader("Upload a csv file for analysis", type =['csv'])
if upload_csv is not None:
    df = pd.read_csv(upload_csv)
    sdf = SmartDataframe(df,config={"llm":llm})
    # st.write (sdf.head(3))

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
