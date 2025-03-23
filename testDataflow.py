
import requests
import json
import pandas as pd
import io

# symbol = ["RELIANCE", "SBIN","TCS","INFY","HDFC","ITC","ASIANPAINT","AXISBANK","ADANIPORTS","BAJAJFINSV"]

# symbol=symbol[0]

# stock_url=f'https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}'
# headers= {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36' ,
# "accept-encoding": "gzip, deflate, br", "accept-language": "en-US,en;q=0.9"}
# r = requests.get(stock_url,headers==headers).json()
# print(r)
# data_values=[data for data in r['data']]
# stock_data=pd.DataFrame(data_values)
# latest_price = stock_data['CH_CLOSING_PRICE'].iloc[-1]
# print(stock_data,latest_price)

url = "https://raw.githubusercontent.com/Rigava/Load-Nifty-Data/main/stock_dfs_updated/{}.csv".format("RELIANCE")
download = requests.get(url).content
data = pd.read_csv(io.StringIO(download.decode('utf-8')))   
print(data.columns)
latest_date = data['Date'].iloc[-1]
print(latest_date)