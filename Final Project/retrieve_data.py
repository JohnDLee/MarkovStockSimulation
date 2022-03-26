import alpaca_trade_api as api
import pandas as pd
import os
from tqdm import tqdm



if __name__ == '__main__':
    
    # hard coded tickers
    tickers = ['TSLA', 'AAPL', 'NVDA', 'GS', 'JPM', 'V']

    # connect to API
    ALPACA_KEY = 'PKF1UA7IOK2GNIOZJCCE'
    ALPACA_SECRET_KEY = 'qoUq6tjPOwpwDld6DfjTvov5q0bAj0KvvHnUSUDx'
    rest = api.REST(ALPACA_KEY, ALPACA_SECRET_KEY)


    # creates a data folder with a number tag if extras exist
    if not os.path.exists('data'):
        os.mkdir('data')
        
    # create a clean and raw dir
    if not os.path.exists('data/raw_data'):
        os.mkdir('raw_data')

    if not os.path.exists('data/clean_data'):
        os.mkdir('clean_data')
        
    # retrieve data as far back as possible. Constant end date to remain consistent
    for ticker in tqdm( tickers, desc = 'Tickers'):
        
        ohlc = rest.get_bars(ticker, timeframe = api.TimeFrame(1, unit= api.TimeFrameUnit('Hour')), start = '2000-01-01', end = '2022-03-26').df
        
        # save this raw data
        ohlc.to_csv(f'data/raw_data/{ticker}.csv', index = True, header = True)

        # clean Nans
        ohlc.fillna(method = 'ffill', inplace = True)
        ohlc.dropna(0, inplace = True)
        
        # compute returns
        ohlc['returns'] = ohlc.close.pct_change()
        
        # first value is now a nan
        ohlc.dropna(0, inplace = True)
        
        # only include correct time period from 9:00AM - 4:00 PM
        ohlc = ohlc.between_time('09:00', '16:00', include_start = True, include_end = False)
        
        #save this clean data
        ohlc.to_csv(f'data/clean_data/{ticker}.csv', index = True, header = True)
        
        