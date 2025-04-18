from bs4 import BeautifulSoup as bs
import os
import requests
import pickle
import datetime as dt
import yfinance as yf

#Get the ticker list from the wiki site and pickle it to file
def save_nifty50():
  resp=requests.get('https://en.wikipedia.org/wiki/NIFTY_50')
  soup=bs(resp.text,"lxml")
  table=soup.find('table',{'class':'wikitable sortable'})
  ticker_list=[]
  for row in table.findAll('tr')[1:]:
    ticker=row.findAll('td')[0].text.rstrip()
    ticker_list.append(ticker)
  with open("nifty50tickers.pickle",'wb') as f:
    pickle.dump(ticker_list,f)
    print(ticker_list)
  return ticker_list
# save_nifty50()
#Get data for each stock and dunp it to folder stock_df_updated
def get_data(reload_nifty=False):
  if reload_nifty:
    tickers=save_nifty50()
  else:
    with open("nifty50tickers.pickle",'rb') as f:
      tickers=pickle.load(f)
  if not os.path.exists('stock_dfs_updated'):
    os.makedirs('stock_dfs_updated')

  for ticker in tickers:
    ticker_yahoo = ticker+'.NS'
    print(ticker_yahoo)
    if not os.path.exists('stock_dfs_updated/{}.csv'.format(ticker)):
      data = yf.download(ticker_yahoo,group_by="Ticker",start="2018-01-01", end=None)
      data = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
      data.to_csv('stock_dfs_updated/{}.csv'.format(ticker))
    else:
      print('Already have'.format(ticker))

get_data()