import pandas as pd
import numpy as np

def convert_zillions_to_float(pd_ser):
	'''
	'''
	mil_idx = pd_ser.str.contains('Mil', na = False)
	new_ser = pd_ser.str.replace('Mil','')
	new_ser_mil = new_ser[mil_idx].astype(np.float)*1000000

	bil_idx = pd_ser.str.contains('Bil', na = False)
	new_ser = pd_ser.str.replace('Bil','')
	new_ser_bil = new_ser[bil_idx].astype(np.float)*1000000000

	return pd.Series(index = pd_ser.index).combine_first(new_ser_mil).combine_first(new_ser_bil)

def clean_fund_data():
	'''
	'''
	fund_data =  pd.read_csv('fundamental_data.csv', encoding = 'utf-8')
	fund_data = fund_data.set_index(['Unnamed: 0'])
	fund_data.index.rename('Ticker', inplace = True)
	fund_data['Net Income'] = convert_zillions_to_float(fund_data['Net Income'])
	fund_data['Market Cap'] = convert_zillions_to_float(fund_data['Market Cap'])
	fund_data['Sales'] = convert_zillions_to_float(fund_data['Sales'])
	fund_data['Employees'] = fund_data['Employees'].str.replace(',','').astype(np.float)
	fund_data = fund_data.replace([np.inf, -np.inf], np.nan)
	return fund_data

def clean_returns_data():
	'''
	'''
	returns_data = pd.read_csv('returns_data.csv')
	returns_data = returns_data.set_index(['Unnamed: 0'])
	returns_data.index.rename('Date', inplace = True)
	returns_data = returns_data.replace([np.inf, -np.inf], np.nan)
	return returns_data

