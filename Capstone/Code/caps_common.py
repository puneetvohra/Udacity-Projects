import pandas as pd

def get_ticker_list():
	'''
	Reads the companylist.csv file (downloaded from NASDAQ website) and returns a list of ticker strings.
	'''
	df = pd.read_csv('data/companylist.csv')
	df = df[(df['Sector'] != 'n/a') & (df['Industry'] != 'n/a')]
	symbols = df['Symbol'].tolist()
	raw_symbols = [x.strip() for x in symbols]
	clean_symbols = [x for x in raw_symbols if '^' not in x]
	clean_symbols = [x for x in clean_symbols if '.' not in x]
	return clean_symbols

def get_sector_tickers(fund_data):
	'''
	Inputs raw fundamental data and returns a dictionary with keys = sector name
	and value = list of tickers belonging to that sector
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

def calc_sector_precision(predictions, sector):
	''' Given a predictions dataframe (with 'Predictions' and 'Actual' column), it calculates the precision of predictions
		for that sector.
	'''
	true_positives = predictions[(predictions['Predictions'] == sector) & (predictions['Actual'] == sector)]
	sector_preds = predictions[predictions['Predictions'] == sector]
	return float(len(true_positives))/len(sector_preds)

def calc_sector_recall(predictions, sector):
	''' Given a predictions dataframe (with 'Predictions' and 'Actual' column), it calculates the recall of predictions
		for that sector.
	'''
	true_positives = predictions[(predictions['Predictions'] == sector) & (predictions['Actual'] == sector)]
	sector_act = predictions[predictions['Actual'] == sector]
	return float(len(true_positives))/len(sector_act)

def create_sector_counts(clean_fund_data):
	'''
	Creates a csv with number of firms belonging to each sector
	'''
	sector_count = {}
	for sector in clean_fund_data['Sector'].unique():
		sector_data = clean_fund_data[clean_fund_data['Sector'] == sector]
		sector_count[sector] = len(sector_data)
	pd.DataFrame.from_dict(sector_count, orient = 'index').to_csv('data/sector_counts.csv')
	return

def create_sector_prec_recall(prediction_df):
	'''
	Creates a csv with precision and recall for each sector
	'''
	sector_prec = {}
	sector_recall = {}
	for sector in prediction_df['Actual'].unique():
		sector_prec[sector] = calc_sector_precision(prediction_df, sector)
		sector_recall[sector] = calc_sector_recall(prediction_df, sector)
	sector_prec_df = pd.DataFrame.from_dict(sector_prec,orient = 'index')
	sector_recall_df = pd.DataFrame.from_dict(sector_recall,orient = 'index')
	sector_prec_recall = pd.merge(sector_prec_df, sector_recall_df, left_index = True, right_index = True)
	sector_prec_recall.columns = ['Precision','Recall']
	sector_prec_recall.to_csv('data/prec_recall.csv')
	return
