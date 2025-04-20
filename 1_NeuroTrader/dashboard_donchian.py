import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from donchian import optimize_donchian,donchian_breakout_data,donchian_breakout
from bar_permute import get_permutation
from moving_average import optimize_moving_average,moving_average
from tqdm import tqdm
from io import BytesIO

dashboard = st.sidebar.selectbox("Select Strategy",["InSample","Simulation","Trend_MA","Animation"])
st.title('QUANT TRADER')
with open("nifty50tickers.pickle",'rb') as f:
    tickers=pickle.load(f)
symbol = st.sidebar.selectbox("Select stock to pull data", tickers)

if dashboard == "InSample":
    df = yf.download(symbol+'.NS',group_by="Ticker",start="2020-01-01", end=None)
    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    df.index = df.index.astype('datetime64[s]')
    # st.write(df)
    # TO find the best lookback for the donchian breakout based on profit factor for last 5 years
    train_df = df[(df.index.year >= 2020) & (df.index.year < 2026)]
    best_lookback, best_real_pf = optimize_donchian(train_df)
    st.write("In-sample PF", best_real_pf, "Best Lookback", best_lookback)

    df = df[(df.index.year >= 2020) & (df.index.year < 2026)]
    #Donchian breakout data
    donchian_data = donchian_breakout_data(df, best_lookback)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=donchian_data.index, y=donchian_data['Close'], mode='lines', name='Close Price',line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=donchian_data.index, y=donchian_data['Upper'], mode='lines', name='Upper Band',line=dict(color="green")))
    fig.add_trace(go.Scatter(x=donchian_data.index, y=donchian_data['Lower'], mode='lines', name='Lower Band',line=dict(color="red")))   
    
    #Visual to check on returns
    signal = donchian_breakout(df, best_lookback) 

    df['r'] = np.log(df['Close']).diff().shift(-1)
    df['donch_r'] = df['r'] * signal
    equity_retrun_donc = (df['donch_r']).cumsum()
    plt.style.use("dark_background")
    # bechmark_returns_pct = (donchian_data['return']+1).cumprod()
    
    plt.plot(equity_retrun_donc,color = "red" ,label='Donchian Strategy Returns')
    # plt.plot(bechmark_returns_pct, label='Benchmark Returns')
    plt.title("In-Sample Donchian Breakout")
    plt.ylabel('Cumulative Return')
    st.pyplot(plt)
    # User Button for starting the Algo
if dashboard == "Simulation":
    st.write("Simulation started")
    df = yf.download(symbol+'.NS',group_by="Ticker",start="2010-01-01", end=None)
    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    df.index = df.index.astype('datetime64[s]')

    train_df = df[(df.index.year >= 2016) & (df.index.year < 2020)]
    best_lookback, best_real_pf = optimize_donchian(train_df)
    print("In-sample PF", best_real_pf, "Best Lookback", best_lookback)


    n_permutations = 100
    perm_better_count = 1
    permuted_pfs = []
    print("In-Sample MCPT")
    for perm_i in tqdm(range(1, n_permutations)):
        train_perm = get_permutation(train_df)
        _, best_perm_pf = optimize_donchian(train_perm)

        if best_perm_pf >= best_real_pf:
            perm_better_count += 1

        permuted_pfs.append(best_perm_pf)

    insample_mcpt_pval = perm_better_count / n_permutations
    print(f"In-sample MCPT P-Value: {insample_mcpt_pval}")

    plt.style.use('dark_background')
    pd.Series(permuted_pfs).hist(color='blue', label='Permutations')
    plt.axvline(best_real_pf, color='red', label='Real')
    plt.xlabel("Profit Factor")
    plt.title(f"In-sample MCPT. P-Value: {insample_mcpt_pval}")
    plt.grid(False)
    plt.legend()
    st.pyplot(plt)


if dashboard == "Trend_MA":
    st.write("Moving average trend of last 5 years")
    df = yf.download(symbol+'.NS',group_by="Ticker",start="2017-01-01", end=None)
    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    df.index = df.index.astype('datetime64[s]')
    
    # Select fast and slow for the crossover strategy
    fast = st.sidebar.slider("Fast Period", min_value=5, max_value=50, value=30, step=5)
    slow = st.sidebar.slider("Slow Period", min_value=51, max_value=200, value=100, step=5)
    # #Visuals
    df = df[(df.index.year >= 2020) & (df.index.year < 2026)]
    df['ret'] =df.Close.pct_change()
    #Buy and Hold returns
    ret = (df.ret+1).cumprod()
    moving_average = moving_average(df,fast,slow)
    
    st.dataframe(moving_average)
    trades = moving_average[moving_average['Long_signal'] == 1]
    if len(trades)%2!=0:
        mtm = df.tail(1).copy()
        mtm.price = mtm.Close
        trades =pd.concat([trades,mtm])
    profits = trades.price.diff()[1::2] / trades.price[0::2].values # Sell - Buy / Buy
    gain = (profits + 1).prod()
    st.write("Benchmark Return",ret[-1],"Total Profit with the MA strategy", gain)
    st.write("Below is the Trade frame")
    st.write(trades)
    # st.write(profits)
    #Visual to check on returns
    plt.style.use("dark_background")
    pd.Series(profits).plot(color='blue',label="profit trade")
    (pd.Series(profits)+1).cumprod().plot(color="red",label="MA Strategy")
    
    plt.title("In-Sample Moving Average Returns")
    plt.ylabel('Cumulative Return')
    st.pyplot(plt)
    #Plotly To see the trades
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=moving_average.index, y=moving_average['Close'], mode='lines', name='Close Price',line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=moving_average.index, y=moving_average['sma_fast'], mode='lines', name='Fast MA',line=dict(color="red")))
    fig.add_trace(go.Scatter(x=moving_average.index, y=moving_average['sma_slow'], mode='lines', name='Slow MA',line=dict(color="black")))
    fig.add_trace(go.Scatter(x=trades.index[0::2], y=trades['price'][0::2], mode='markers', name='Buy Signal', marker=dict(color="green", size=10)))
    fig.add_trace(go.Scatter(x=trades.index[1::2], y=trades['price'][1::2], mode='markers', name='Sell Signal', marker=dict(color="red", size=10)))
    fig.update_layout(title='Moving Average Strategy', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # TO find the best lookback for the donchian breakout based on profit factor
    if st.button("Optimize the MA"):
        train_df = df[(df.index.year >= 2020) & (df.index.year < 2026)]
        best_lookback_fast,best_lookback_slow ,best_real_pf = optimize_moving_average(train_df)
        st.write("In-sample MA PF", best_real_pf, "Best Lookback Fast:", best_lookback_fast, "Bese lookback slow",best_lookback_slow)

if dashboard == "Animation":
    # Streamlit app title
    st.title("Stock Comparison Dashboard")

    with open("nifty50tickers.pickle",'rb') as f:
        tickers=pickle.load(f)

    # Sidebar for user input
    selected_tickers = st.sidebar.multiselect(
        "Select Stocks to Plot",tickers, default=['RELIANCE', 'ITC']
    )
    # Convert tickers to Yahoo Finance format
    yf_tickers = [f"{stok}.NS" for stok in selected_tickers]
    # Fetch stock data
    @st.cache_data
    def fetch_data(tickers):
        return yf.download(tickers,period='5y')['Close']

    if selected_tickers:
        df = fetch_data(yf_tickers)
        # Resample data to monthly frequency
        df_monthly = df.resample('M').last()  # Take the mean for each month
        st.write("Data Preview (Monthly Data)", df_monthly.head())

        # Create a static plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Stock returns History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

        for ticker, yf_ticker in zip(selected_tickers, yf_tickers):
            ax.plot(df.index, np.exp(np.log(df[yf_ticker]/df[yf_ticker].shift(1)).cumsum()), label=ticker)

        ax.legend()
        st.pyplot(fig)
        
        # Prepare data for animation
        df_reset = df_monthly.reset_index()  # Reset index to use Date as a column
        df_reset['Date'] = pd.to_datetime(df_reset['Date'])  # Ensure Date is in datetime format
        df_reset['Date'] = df_reset['Date'].dt.strftime('%Y-%m-%d')  # Convert Date to string format

        # Create the base figure
        fig = go.Figure()

        # Add traces for each stock
        for ticker in yf_tickers:
            fig.add_trace(go.Scatter(
                x=df_reset['Date'],
                y=df_reset[ticker],
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))

        # Create frames for animation (one frame per day)
        frames = []
        for i, date in enumerate(df_reset['Date']):
            frame_data = []
            for ticker in yf_tickers:
                frame_data.append(go.Scatter(
                    x=df_reset['Date'][:i+1],  # Show data up to the current frame
                    y=df_reset[ticker][:i+1],
                    mode='lines',
                    name=ticker,
                    line=dict(width=2)
                ))
            frames.append(go.Frame(data=frame_data, name=str(date)))

        fig.frames = frames

        # Update layout for animation
        fig.update_layout(
            title="Stock Price History",
            xaxis=dict(title="Date", showgrid=True),
            yaxis=dict(title="Price", showgrid=True),
            template="plotly_white",
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                            method="animate",
                            args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                        dict(label="Pause",
                            method="animate",
                            args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                    ]
                )
            ]
        )

        # Add slider for animation
        fig.update_layout(
            sliders=[{
                "steps": [
                    {
                        "args": [[frame.name], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}],
                        "label": frame.name,
                        "method": "animate",
                    } for frame in fig.frames
                ],
                "transition": {"duration": 0},
                "x": 0.1,
                "len": 0.9
            }]
        )


        # Add slider for animation
        fig.update_layout(
            sliders=[{
                "steps": [
                    {
                        "args": [[frame.name], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}],
                        "label": frame.name,
                        "method": "animate",
                    } for frame in fig.frames
                ],
                "transition": {"duration": 0},
                "x": 0.1,
                "len": 0.9
            }]
        )

        # Save the animated chart as an HTML file
        html_str = fig.to_html(include_plotlyjs='cdn')  # Save the chart as a string
        html_buffer = BytesIO(html_str.encode('utf-8'))  # Encode the string as bytes

        # Add a download button for the HTML file
        st.download_button(
            label="Download Animated Chart as HTML",
            data=html_buffer,
            file_name="animated_stock_chart.html",
            mime="text/html"
        )

        # Display the animated chart
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Please select at least one stock to plot.")