# https://github.com/neurotrader888/mcpt/blob/main/donchian.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## STRATEGY 1: Donchian Breakout
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
    return ohlc
def trades(donchian_breakout_data):
    trades = donchian_breakout_data[donchian_breakout_data['pos_change'] != 0]
    if len(trades)%2!=0:
        mtm = donchian_breakout_data.tail(1).copy()
        mtm.price = mtm.Close
        trades =pd.concat([trades,mtm])
    trades['exit'] = trades['price'].shift(-1)
    trades['profit'] = trades['exit'] - trades['price']
    trades['profit'] = trades['profit']*trades['signal']
    trades['strat_return'] = trades['profit']/trades['price'] 
    trades['strat_equity'] = (1+trades['strat_return']).cumprod()
    trades.dropna(inplace=True)
    return trades

def donchian_breakout(ohlc: pd.DataFrame, lookback: int):
    # input df is assumed to have a 'close' column
    upper = ohlc['Close'].rolling(lookback - 1).max().shift(1)
    lower = ohlc['Close'].rolling(lookback - 1).min().shift(1)
    signal = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
    signal.loc[ohlc['Close'] > upper] = 1
    signal.loc[ohlc['Close'] < lower] = -1
    signal = signal.ffill()
    return signal

def optimize_donchian(ohlc: pd.DataFrame):

    best_pf = 0
    best_lookback = -1
    r = np.log(ohlc['Close']).diff().shift(-1)
    for lookback in range(12, 169):
        signal = donchian_breakout(ohlc, lookback)
        sig_rets = signal * r
        sig_pf = sig_rets[sig_rets > 0].sum() / sig_rets[sig_rets < 0].abs().sum()

        if sig_pf > best_pf:
            best_pf = sig_pf
            best_lookback = lookback

    return best_lookback, best_pf

def walkforward_donch(ohlc: pd.DataFrame, train_lookback: int = 24 * 365 * 4, train_step: int = 24 * 30):

    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None
    
    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            best_lookback, _ = optimize_donchian(ohlc.iloc[i-train_lookback:i])
            tmp_signal = donchian_breakout(ohlc, best_lookback)
            next_train += train_step
        
        wf_signal[i] = tmp_signal.iloc[i]
    
    return wf_signal

import yfinance as yf
if __name__ == '__main__':

    # Load data
    df = yf.download('ITC.NS',group_by="Ticker",start="2020-01-01", end=None)
    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    df.index = df.index.astype('datetime64[s]')

    # df = df[(df.index.year >= 2010) & (df.index.year < 2020)] 
    best_lookback, best_real_pf = optimize_donchian(df)

    # Best lookback = 19, best_real_pf = 1.08
    print(f"Best lookback = {best_lookback}, best_real_pf = {best_real_pf}")
    signal = donchian_breakout(df, best_lookback) 

    df['r'] = np.log(df['Close']).diff().shift(-1)
    df['donch_r'] = df['r'] * signal

    plt.style.use("dark_background")
    df['donch_r'].cumsum().plot(color='red')
    plt.title("In-Sample Donchian Breakout")
    plt.ylabel('Cumulative Log Return')
    plt.show()

