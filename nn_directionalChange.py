import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf

def directional_change(Close: np.array, High: np.array, Low: np.array, sigma: float):
    
    up_zig = True # Last extreme is a bottom. Next is a top. 
    tmp_max = High[0]
    tmp_min = Low[0]
    tmp_max_i = 0
    tmp_min_i = 0

    tops = []
    bottoms = []

    for i in range(len(Close)):
        if up_zig: # Last extreme is a bottom
            if High[i] > tmp_max:
                # New High, update 
                tmp_max = High[i]
                tmp_max_i = i
            elif Close[i] < tmp_max - tmp_max * sigma: 
                # Price retraced by sigma %. Top confirmed, record it
                # top[0] = confirmation index
                # top[1] = index of top
                # top[2] = price of top
                top = [i, tmp_max_i, tmp_max]
                tops.append(top)

                # Setup for next bottom
                up_zig = False
                tmp_min = Low[i]
                tmp_min_i = i
        else: # Last extreme is a top
            if Low[i] < tmp_min:
                # New Low, update 
                tmp_min = Low[i]
                tmp_min_i = i
            elif Close[i] > tmp_min + tmp_min * sigma: 
                # Price retraced by sigma %. Bottom confirmed, record it
                # bottom[0] = confirmation index
                # bottom[1] = index of bottom
                # bottom[2] = price of bottom
                bottom = [i, tmp_min_i, tmp_min]
                bottoms.append(bottom)

                # Setup for next top
                up_zig = True
                tmp_max = High[i]
                tmp_max_i = i

    return tops, bottoms

def get_extremes(ohlc: pd.DataFrame, sigma: float):
    tops, bottoms = directional_change(ohlc['Close'], ohlc['High'], ohlc['Low'], sigma)
    tops = pd.DataFrame(tops, columns=['conf_i', 'ext_i', 'ext_p'])
    bottoms = pd.DataFrame(bottoms, columns=['conf_i', 'ext_i', 'ext_p'])
    tops['type'] = 1
    bottoms['type'] = -1
    extremes = pd.concat([tops, bottoms])
    extremes = extremes.set_index('conf_i')
    extremes = extremes.sort_index()
    return extremes




if __name__ == '__main__':
    # Load data
    data = yf.download('^NSEI',group_by="Ticker",start="2020-01-01", end=None)
    data = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    # data = data.set_index('date')
    tops, bottoms = directional_change(data['Close'].to_numpy(), data['High'].to_numpy(), data['Low'].to_numpy(), 0.02)

    pd.Series(data['Close'].to_numpy()).plot()
    idx = data.index
    for top in tops:
        plt.plot(top[1], top[2], marker='o', color='green', markersize=4)

    plt.show()













