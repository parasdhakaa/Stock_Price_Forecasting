# app.py
# Simple Streamlit app to visualize historical prices and show prediction.
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import subprocess, sys

st.title('Stock Price Forecasting - Demo')
uploaded = st.file_uploader('Upload CSV (Date, Open, High, Low, Close, Volume)', type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=['Date'])
    st.subheader('Preview')
    st.dataframe(df.tail())
    st.subheader('Closing Price Chart')
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Close'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Close')
    st.pyplot(fig)
    st.info('Use the included train.py and predict.py locally to train models and produce predictions.')
else:
    st.info('Upload a historical CSV (or run data_fetch.py to download one).')
