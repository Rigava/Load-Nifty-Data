import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import binomtest
import plotly.graph_objects as go

# Download data
data = yf.download('^NSEI', group_by='Ticker', start="2010-01-01", end=None)
data = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
# Calculate 50-day SMA
data['sma_50'] = data['Close'].rolling(window=50).mean()

# Define threshold for a bounce (e.g., 0.5% deviation from SMA)
threshold = 0.005  # 5% range

# Identify bounces (price gets close to SMA and then reverses)
data['sma_diff'] = abs((data['Close'] - data['sma_50']) / data['sma_50'])
data['bounce'] = (data['sma_diff'] <= threshold) & (data['Close'].shift(-1) > data['Close']) | (data['Close'].shift(-1) < data['Close'])

# Count observed bounces
observed_bounces = data['bounce'].sum()

# Expected bounces (assuming random price movements)
expected_bounce_rate = 2 * threshold  # Assuming a uniform random distribution
expected_bounces = expected_bounce_rate * len(data)

# Perform binomial test to check statistical significance
p_value = binomtest(observed_bounces, len(data), expected_bounce_rate, alternative="greater").pvalue

print(f"Observed Bounces: {observed_bounces}")
print(f"Expected Bounces (Random Chance): {expected_bounces:.2f}")
print(f"P-Value: {p_value}")

# Conclusion
alpha = 0.05  # Significance level
if p_value < alpha:
    print("Reject H₀: SMA acts as a significant support/resistance level.")
else:
    print("Fail to Reject H₀: No significant evidence that SMA acts as a support/resistance level.")

# Visualization
fig = go.Figure()
# Add price trace
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close Price"))
# Add SMA trace
fig.add_trace(go.Scatter(x=data.index, y=data["sma_50"], mode="lines", name="50-day SMA", line=dict(color="blue")))
#  Highlight Bounce Up Signals
signals = data[data["bounce"] == 1]
fig.add_trace(go.Scatter(x=signals.index, y=signals["Close"], mode="markers", 
                         name="Bounce Up", marker=dict(color="yellow", size=8, symbol="triangle-up")))
fig.write_html("nifty_keltner_signals.html",auto_open=True)
