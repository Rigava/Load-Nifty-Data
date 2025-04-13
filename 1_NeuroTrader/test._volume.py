import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Fetch S&P 500 data (intraday for better granularity)
data= yf.download('^NSEI',group_by='Ticker',start="2025-03-01" ,end=None)
data = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)

# df=pd.read_csv(r'D:\Archives\backtest_ursell\NIFTY 50 - Minute data.csv')
# df.rename(columns={df.columns[0]:"Date",df.columns[1]:"Open",df.columns[2]:"High",df.columns[3]:"Low",df.columns[4]:"Close"},inplace=True)
# df.set_index(df.date,inplace=True)
# df.index=pd.to_datetime(df.index)

sp500 = data.reset_index()
print(sp500.head())
# Convert timestamp to datetime format
sp500['Datetime'] = pd.to_datetime(sp500['Date'])
sp500['Date'] = sp500['Datetime'].dt.date
sp500['DayOfWeek'] = sp500['Datetime'].dt.day_name()
sp500['Hour'] = sp500['Datetime'].dt.hour
sp500['Minute'] = sp500['Datetime'].dt.minute

# Daily average volume by day of the week
daily_avg_volume = sp500.groupby('DayOfWeek')['Volume'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
)

# Hourly average volume
hourly_avg_volume = sp500.groupby('Hour')['Volume'].mean()

# First-minute volume spikes
first_minute_volume = sp500[sp500['Minute'] == 0].groupby('Hour')['Volume'].mean()

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

axes[0].bar(daily_avg_volume.index, daily_avg_volume.values, color='blue', alpha=0.7)
axes[0].set_title("Average Volume by Day of the Week")
axes[0].set_ylabel("Volume")

axes[1].plot(hourly_avg_volume.index, hourly_avg_volume.values, marker='o', linestyle='-', color='green')
axes[1].set_title("Average Volume by Hour of the Day")
axes[1].set_xlabel("Hour (UTC)")
axes[1].set_ylabel("Volume")

axes[2].plot(first_minute_volume.index, first_minute_volume.values, marker='s', linestyle='-', color='red')
axes[2].set_title("First Minute Volume Spikes by Hour")
axes[2].set_xlabel("Hour (UTC)")
axes[2].set_ylabel("Volume")

plt.tight_layout()
plt.show()
