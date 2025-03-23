import numpy as np
import pandas as pd
from typing import List, Union

def get_permutation(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]], start_index: int = 0, seed=None
):
    assert start_index >= 0

    np.random.seed(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc]

    n_bars = len(ohlc[0])

    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[['Open', 'High', 'Low', 'Close']])

        # Get start bar
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        # Open relative to last close
        r_o = (log_bars['Open'] - log_bars['Close'].shift()).to_numpy()
        
        # Get prices relative to this bars open
        r_h = (log_bars['High'] - log_bars['Open']).to_numpy()
        r_l = (log_bars['Low'] - log_bars['Open']).to_numpy()
        r_c = (log_bars['Close'] - log_bars['Open']).to_numpy()

        relative_open[mkt_i] = r_o[perm_index:]
        relative_high[mkt_i] = r_h[perm_index:]
        relative_low[mkt_i] = r_l[perm_index:]
        relative_close[mkt_i] = r_c[perm_index:]

    idx = np.arange(perm_n)

    # Shuffle intrabar relative values (high/low/close)
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    # Shuffle last close to open (gaps) seprately
    perm2 = np.random.permutation(idx)
    relative_open = relative_open[:, perm2]

    # Create permutation from relative prices
    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))

        # Copy over real data before start index 
        log_bars = np.log(reg_bars[['Open', 'High', 'Low', 'Close']]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]
        
        # Copy start bar
        perm_bars[start_index] = start_bar[mkt_i]

        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(perm_bars, index=time_index, columns=['Open', 'High', 'Low', 'Close'])

        perm_ohlc.append(perm_bars)

    if n_markets > 1:
        return perm_ohlc
    else:
        return perm_ohlc[0]
import yfinance as yf
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    df = yf.download('ITC.NS',group_by="Ticker",start="2018-01-01", end=None)
    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    df.index = df.index.astype('datetime64[s]')
    btc_real = df[(df.index.year >= 2018) & (df.index.year < 2025)]

    btc_perm = get_permutation(btc_real)

    btc_real_r = np.log(btc_real['Close']).diff() 
    btc_perm_r = np.log(btc_perm['Close']).diff()

    print(f"Mean. REAL: {btc_real_r.mean():14.6f} PERM: {btc_perm_r.mean():14.6f}")
    print(f"Stdd. REAL: {btc_real_r.std():14.6f} PERM: {btc_perm_r.std():14.6f}")
    print(f"Skew. REAL: {btc_real_r.skew():14.6f} PERM: {btc_perm_r.skew():14.6f}")
    print(f"Kurt. REAL: {btc_real_r.kurt():14.6f} PERM: {btc_perm_r.kurt():14.6f}")

    df1 = yf.download('RELIANCE.NS',group_by="Ticker",start="2018-01-01", end=None)
    df1 = df1.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    df1.index = df1.index.astype('datetime64[s]')
    eth_real = df1[(df1.index.year >= 2018) & (df1.index.year < 2025)]
    eth_real_r = np.log(eth_real['Close']).diff()
    
    print("") 
    # plt.style.use("dark_background")    
    np.log(btc_real['Close']).diff().cumsum().plot(color='purple', label='realNifty')
    np.log(btc_perm['Close']).diff().cumsum().plot(color='orange', label='permNifty')
    plt.title("Permuted NIFTY and REal nifty")
    plt.ylabel("Cumulative Log Return")
    plt.legend()
    plt.show()

    permed = get_permutation([btc_real, eth_real])
    btc_perm = permed[0]
    eth_perm = permed[1]
    
    btc_perm_r = np.log(btc_perm['Close']).diff()
    eth_perm_r = np.log(eth_perm['Close']).diff()
    print(f"BTC&ETH Correlation REAL: {btc_real_r.corr(eth_real_r):5.3f} PERM: {btc_perm_r.corr(eth_perm_r):5.3f}")


    np.log(eth_real['Close']).diff().cumsum().plot(color='purple', label='ETHUSD')
    np.log(eth_perm['Close']).diff().cumsum().plot(color='orange', label='reliance')
    
    plt.ylabel("Cumulative Log Return")
    plt.title("Real BTCUSD and ETHUSD")
    plt.legend()
    plt.show()

  
 
  


