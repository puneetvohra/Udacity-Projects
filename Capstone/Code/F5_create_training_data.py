from sklearn.cross_validation import train_test_split
import caps_common as cc
from sklearn import preprocessing
import numpy as np
import pandas as pd
from wordcloud import WordCloud


def create_test_train_data(clean_features, returns_data):
	'''
	'''
	X_all = clean_features.drop(['Sector'], axis =1)
	y_all = clean_features[['Sector']]
	X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, random_state=42)
	sector_tickers = cc.get_sector_tickers(y_train)
	sector_rtns = pd.DataFrame(columns = sector_tickers.keys(), index = returns_data.index)
	for k,v in sector_tickers.iteritems():
		sector_rtns[k] = returns_data[list(set(v) & set(returns_data.columns))].mean(axis = 1)
	
	for k,v in sector_tickers.iteritems():
		sector_summary = ' '.join(map(str,X_all.loc[v,'Summary'].tolist()))
		wordcloud = WordCloud().generate(sector_summary)
		image = wordcloud.to_image()
		try:
			image.save('wordclouds/' + k + '.png', format='png')
		except:
			try:
				image.show()
			except:
				pass
	all_summaries = ' '.join(map(str,X_all['Summary'].tolist()))
	wordcloud = WordCloud().generate(all_summaries)
	image = wordcloud.to_image()
	try:
		image.save('wordclouds/summary.png', format='png')
	except:
		pass
	top_words = [a for a,b in wordcloud.words_[:30]]

	for ticker in X_train.index:
		for word in top_words:
			X_train.loc[ticker, word + ' Count'] = str(X_train.loc[ticker,'Summary']).count(word)
		if ticker in returns_data.columns:
			sector = y_train.loc[ticker, 'Sector']
			for k,v in sector_tickers.iteritems():
				if k == sector:
					sector_returns = returns_data[list((set(v) - set([ticker])) & set(returns_data.columns))].mean(axis = 1)
				else:
					sector_returns = sector_rtns[k]
				X_train.loc[ticker, k + ' Corr'] = sector_returns.corr(returns_data[ticker])


	for ticker in X_test.index:
		for word in top_words:
			X_train.loc[ticker, word + ' Count'] = str(X_train.loc[ticker,'Summary']).count(word)
		if ticker in returns_data.columns:
			sector_corrs = sector_rtns.corrwith(returns_data[ticker])
			for k,v in sector_tickers.iteritems():
				X_test.loc[ticker, k + ' Corr'] = sector_corrs[k]

	X_train = X_train.reset_index()
	y_train = y_train.reset_index()
	del X_train['Ticker']
	del y_train['Ticker']

	X_test = X_test.reset_index()
	y_test = y_test.reset_index()


	X_train = X_train.replace([np.inf, -np.inf], np.nan)
	X_test = X_test.replace([np.inf, -np.inf], np.nan)

	ticker_idx_map = X_test['Ticker'].to_dict()
	del X_test['Ticker']
	del y_test['Ticker']

	del X_train['Summary']
	del X_test['Summary']
	col_list = X_test.columns.tolist()



	return X_train, X_test, y_train, y_test, ticker_idx_map, col_list