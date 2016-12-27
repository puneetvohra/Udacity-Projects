import pandas as pd
import numpy as np

def create_features(clean_fund_data):
	'''
	Creates fundamental data features: 5 financial ratios. Returns data with these 5 columns added.
	'''
	unique_sectors = clean_fund_data['Sector'].unique()
	clean_features = pd.get_dummies(clean_fund_data['Stock Type'])
	clean_features['margin'] = clean_fund_data['Net Income']/clean_fund_data['Sales']
	clean_features['psr'] = clean_fund_data['Market Cap']/clean_fund_data['Sales']
	clean_features['per'] = clean_fund_data['Market Cap']/clean_fund_data['Net Income']
	clean_features['ser'] = clean_fund_data['Sales']/clean_fund_data['Employees']
	clean_features['eer'] = clean_fund_data['Net Income']/clean_fund_data['Employees']
	clean_features['Sector'] = clean_fund_data['Sector']
	clean_features['Summary'] = clean_fund_data['Summary']
	clean_features = clean_features.replace([np.inf, -np.inf], np.nan)
	return clean_features
