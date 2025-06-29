# https://www.youtube.com/watch?v=3zI_l_P-lF8
##SMA as support and resistance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import binomtest
# Download data
data= yf.download('^NSEI',group_by='Ticker',start="2008-09-01" ,end=None)
data = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
# Calculate 50-day SMA
data['sma_1']=data['Close'].rolling(window=200).mean()
#Calcultaing the average true range
high = data["High"]
low = data["Low"]
close = data["Close"]
data['tr0'] = abs(high - low)
data['tr1'] = abs(high - close.shift(1))
data['tr2'] = abs(low - close.shift(1))
data['TR']=np.maximum(data['tr0'],data['tr1'],data['tr2'])
data['ATR']=data['TR'].rolling(window=14).mean()
data['ATR_wilder']=data['ATR'].ewm(span=14,adjust=False).mean()
#Keltner Channels
data['upper_keltner'] = data['sma_1'] + (data['ATR_wilder'] * 0.5)
data['lower_keltner'] = data['sma_1'] - (data['ATR_wilder'] * 0.5)
# print(data[["Close","sma_1","upper_keltner","lower_keltner"]].tail(15))

# Define interaction classification

penetration_upper = data['Close'] > data['upper_keltner']
penetration_lower = data['Close'] < data['lower_keltner']
bounce_upper = (data['Close'] <= data['upper_keltner']) & (data['Close'] > data['sma_1']) & ~penetration_upper & ~penetration_lower
bounce_lower = (data['Close'] >= data['lower_keltner']) & (data['Close'] < data['sma_1']) & ~penetration_lower & ~penetration_upper
within_bands = (data['Close'] <= data['upper_keltner']) & (data['Close'] >= data['lower_keltner'])

data["interaction"] = np.where(penetration_upper, "Penetration_Up",
                 np.where(penetration_lower, "Penetration_Down",
                 np.where(bounce_upper, "Bounce_Up",
                 np.where(bounce_lower, "Bounce_Down", "Within Bands"))))
print(data[["Close","ATR_wilder" ,"sma_1", "upper_keltner", "lower_keltner", "interaction"]].tail(15))

# Vectorized function to detect changes in interaction
def detect_individual_signals(interactions, interaction_type):
    """Identifies signal points where a specific interaction changes."""
    prev_interaction = np.roll(interactions, shift=1)
    signals = np.where((interactions == interaction_type) & (prev_interaction != interaction_type), 1, 0)
    signals[0] = 0  # First row has no previous value to compare
    return signals

# Apply the vectorized function for separate signals
data["signal_bounce_up"] = detect_individual_signals(data["interaction"].values, "Bounce_Up")
data["signal_bounce_down"] = detect_individual_signals(data["interaction"].values, "Bounce_Down")
data["signal_penup"] = detect_individual_signals(data["interaction"].values, "Penetration_Up")
data["signal_pendown"] = detect_individual_signals(data["interaction"].values, "Penetration_Down")
data["signal_within_bands"] = detect_individual_signals(data["interaction"].values, "Within Bands")
print(data[["Close","ATR_wilder" ,"sma_1", "upper_keltner", "lower_keltner", "interaction", "signal_bounce_up", "signal_bounce_down", "signal_penup","signal_pendown"]].tail(15))


# # Print rows where any signal occurs
# print(data[(data["signal_bounce_up"] == 1) | (data["signal_bounce_down"] == 1) | (data["signal_within_bands"] == 1)]
#       [["Close", "sma_1", "upper_keltner", "lower_keltner", "interaction", "signal_bounce_up", "signal_bounce_down", "signal_within_bands"]])

# Count observed bounces
observed_bounces_up = data['signal_bounce_up'].sum()
observed_bounces_down = data['signal_bounce_down'].sum()
observed_penup = data['signal_penup'].sum()
observed_pendown = data['signal_pendown'].sum()
support_penetration = observed_penup + observed_pendown
support_bounces = observed_bounces_up + observed_bounces_down
Bounce_pct = support_bounces/(support_bounces+support_penetration)
print(f"Observed Penetration: {observed_penup} Penetration Up, {observed_pendown} Penetration Down,{support_penetration} Support Penetration")
print(f"Observed Bounces: {observed_bounces_up} Bounce Up, {observed_bounces_down} Bounce Down,{support_bounces} Support Bounce")
print(f"Percentage of Bounce Signals: {Bounce_pct:.2%}")
# # Define threshold for a bounce (e.g., 0.5% deviation from SMA)
# threshold = 0.01  # 1% range
# # Expected bounces (assuming random price movements)
# expected_bounce_rate = 2 * threshold  # Assuming a uniform random distribution
# expected_bounces = expected_bounce_rate * len(data)
# # Perform binomial test to check statistical significance
# p_value = binomtest(support_bounces, len(data), expected_bounce_rate, alternative="greater").pvalue
# print(f"Observed Bounces: {support_bounces}")
# print(f"Expected Bounces (Random Chance): {expected_bounces:.2f}")
# print(f"P-Value: {p_value}")
# # Conclusion
# alpha = 0.05  # Significance level
# if p_value < alpha:
#     print("Reject H₀: SMA acts as a significant support/resistance level.")
# else:
#     print("Fail to Reject H₀: No significant evidence that SMA acts as a support/resistance level.")
# Visualization
fig = go.Figure()
# Add price trace
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close Price"))
# Add SMA trace
fig.add_trace(go.Scatter(x=data.index, y=data["sma_1"], mode="lines", name="50-day SMA", line=dict(color="blue")))
# Add upper and lower Keltner bands
fig.add_trace(go.Scatter(x=data.index, y=data["upper_keltner"], mode="lines", name="Upper Keltner Band", line=dict(color="green")))
fig.add_trace(go.Scatter(x=data.index, y=data["lower_keltner"], mode="lines", name="Lower Keltner Band", line=dict(color="red")))
# Highlight Bounce Up Signals
signals_up = data[data["signal_bounce_up"] == 1]
fig.add_trace(go.Scatter(x=signals_up.index, y=signals_up["Close"], mode="markers", 
                         name="Bounce Up", marker=dict(color="yellow", size=8, symbol="triangle-up")))
# Highlight Bounce Down Signals
signals_down = data[data["signal_bounce_down"] == 1]
fig.add_trace(go.Scatter(x=signals_down.index, y=signals_down["Close"], mode="markers", 
                         name="Bounce Down", marker=dict(color="orange", size=8, symbol="triangle-down")))
# Highlight Within Bands Signals
signals_within = data[data["signal_within_bands"] == 1]
fig.add_trace(go.Scatter(x=signals_within.index, y=signals_within["Close"], mode="markers", 
                         name="Within Bands", marker=dict(color="cyan", size=8, symbol="circle")))
# Highlight Penup Signals
signals_penup = data[data["signal_penup"] == 1]
fig.add_trace(go.Scatter(x=signals_penup.index, y=signals_penup["Close"], mode="markers", 
                         name="PenUp", marker=dict(color="Green", size=8, symbol="triangle-up")))
# Highlight PenDown Signals
signals_pendown = data[data["signal_pendown"] == 1]
fig.add_trace(go.Scatter(x=signals_pendown.index, y=signals_pendown["Close"], mode="markers", 
                         name="PenDown", marker=dict(color="Red", size=8, symbol="triangle-down")))
fig.update_layout(title="Nifty 50 Moving Average and Keltner Channel Analysis with Separate Signals",
                  xaxis_title="Date", yaxis_title="Price", template="plotly_dark")

fig.write_html("nifty_keltner_signals.html",auto_open=True)