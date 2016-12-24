import pandas as pd

def get_ticker_list():
	'''
	'''
	df = pd.read_csv('companylist.csv')
	df = df[(df['Sector'] != 'n/a') & (df['Industry'] != 'n/a')]
	symbols = df['Symbol'].tolist()
	raw_symbols = [x.strip() for x in symbols]
	clean_symbols = [x for x in raw_symbols if '^' not in x]
	clean_symbols = [x for x in clean_symbols if '.' not in x]
	return clean_symbols

def get_sector_tickers(fund_data):
	'''
	'''
	sector_tickers = {}
	for sector in fund_data['Sector'].unique():
		sector_tickers[sector] = fund_data[fund_data['Sector'] == sector].index.tolist()
	return sector_tickers

def create_sector_returns(fund_data, returns_data):
	'''
	'''
	sector_rtns= pd.DataFrame()
	for sector in fund_data['Sector'].unique():
		sector_tickers = fund_data[fund_data['Sector'] == sector].index.tolist()
		sector_rtns[sector] = returns_data[list(set(sector_tickers) & set(returns_data.columns))].mean(axis = 1)
	return sector_rtns