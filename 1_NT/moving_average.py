import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Vectorized function to detect changes in interaction
def detect_individual_signals(interactions, interaction_type):
    """Identifies signal points where a specific interaction changes."""
    prev_interaction = np.roll(interactions, shift=1)
    signals = np.where((interactions == interaction_type) & (prev_interaction != interaction_type), 1, 0)
    signals[0] = 0  # First row has no previous value to compare
    return signals
def moving_average(ohlc: pd.DataFrame, lookback1: int, lookback2: int):
    ohlc['sma_fast'] = ohlc['Close'].rolling(lookback1).mean()
    ohlc['sma_slow'] = ohlc['Close'].rolling(lookback2).mean()
    ohlc['price'] = ohlc['Open'].shift(-1)
    crossoverUp = ohlc['sma_fast'] > ohlc['sma_slow']
    crossoverDown = ohlc['sma_fast'] < ohlc['sma_slow']
    ohlc['interaction'] = np.where(crossoverUp, "Long", np.where(crossoverDown,"Short", "None"))
    ohlc['Long_signal'] = detect_individual_signals(ohlc['interaction'].values, "Long")
    return ohlc
def vectorized(df,n,m):
    data = moving_average(df,n,m)
    first_buy = pd.Series(data.index == (data.sma_fast>data.sma_slow).idxmax(),index=data.index)
    real_signal = first_buy | (data.sma_fast>data.sma_slow).diff()
    trades = data[real_signal]
    if len(trades)%2!=0:
        mtm = data.tail(1).copy()
        mtm.price = mtm.Close
        trades =pd.concat([trades,mtm])
    profits = trades.price.diff()[1::2] / trades.price[0::2].values
    gain = (profits + 1).prod()
    return gain  
def optimize_moving_average(ohlc: pd.DataFrame):
    best_pf = 0
    best_lookback1 = -1
    best_lookback2 = -1
    # r = np.log(ohlc['Open']).diff().shift(-1)
    for lookback1 in range(5, 50):
        for lookback2 in range(51, 200):
            sig_pf = vectorized(ohlc,lookback1,lookback2)
        if sig_pf > best_pf:
            best_pf = sig_pf
            best_lookback1 = lookback1
            best_lookback2 = lookback2
    return best_lookback1, best_lookback2, best_pf