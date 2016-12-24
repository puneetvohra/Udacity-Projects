from yahoo_finance import Share
import pandas as pd
import caps_common as cc
tickers = cc.get_ticker_list()

ticker_prices = {}
hist_prices = pd.DataFrame()
for ticker in tickers:
	try:
		ticker_info = Share(ticker)
		ticker_data = ticker_info.get_historical('2006-01-01', '2016-11-20')
		for day in ticker_data:
			ticker_prices[day['Date']] = day['Adj_Close']
		df = pd.DataFrame.from_dict(ticker_prices, orient  = 'index')
		df.columns = [ticker]
		if hist_prices.empty:
			hist_prices = df
		else:
			hist_prices = hist_prices.merge(df, how = 'outer', left_index = True, right_index = True)
	except:
		pass
hist_prices.index = hist_prices.index.to_datetime()
hist_prices = hist_prices.sort()
hist_prices = hist_prices.convert_objects(convert_numeric = True)
hist_returns = hist_prices.pct_change(1)
hist_returns.to_csv('c:/returns_data.csv')


