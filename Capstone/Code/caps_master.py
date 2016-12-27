import pandas as pd
import numpy as np
import F3_clean_data as cd
import F4_create_features as cf
import F5_create_training_data as ctd
import F6_preprocess_features as pf
import F7_train_models as tm
import config as cfg

if cfg.fetch_raw_data:
	import F1_get_fundamental_data as gfd
	import F2_get_market_data as gmd
	gfd.get_fundamental_data()
	gmd.get_market_data()

clean_fund_data = cd.clean_fund_data()
clean_returns_data = cd.clean_returns_data()
clean_features = cf.create_features(clean_fund_data)
X_train, X_test, y_train, y_test, ticker_idx_map, col_list = ctd.create_test_train_data(clean_features, clean_returns_data)

#X_train = pd.read_csv('X_train.csv')
#X_test = pd.read_csv('X_test.csv')
#y_train = pd.read_csv('y_train.csv')
#y_test = pd.read_csv('y_test.csv')
#col_list = [u'Aggressive', u'Classic', u'Cyclical', u'Distressed', u'Hard', u'High', u'Slow', u'Speculative', 'margin', 'psr', 'per', 'ser', 'eer', 'service Count', 'company Count', 'Inc Count', 'product Count', 'engaged Count', 'provide Count', 'Corp Count', 'gas Count', 'operate Count', 'market Count', 'management Count', 'business Count', 'oil Count', 'real Count', 'subsidiaries Count', 'estate Count', 'energy Count', 'investment Count', 'State Count', 'segment Count', 'Holding Count', 'solution Count', 'United Count', 'natural Count', 'financial Count', 'commercial Count', 'manufacture Count', 'trust Count', 'include Count', 'including Count', u'Communication Services Corr', u'Industrials Corr', u'Consumer Cyclical Corr', u'Consumer Defensive Corr', u'Energy Corr', u'Healthcare Corr', u'Utilities Corr', u'Financial Services Corr', u'Real Estate Corr', u'Basic Materials Corr', u'Technology Corr']

X_train, X_test = pf.preprocess_features(X_train, X_test, col_list)
pred_model, predictions, score = tm.train_classifier(X_train, X_test, y_train, y_test)
prediction_df = pd.Series(predictions).rename(index = ticker_idx_map).to_frame('Predictions')
prediction_df.to_csv('data/predictions.csv')

if cfg.cross_validate:
	from sklearn.metrics import accuracy_score
	X_train, X_test, y_train, y_test, ticker_idx_map, col_list = ctd.create_test_train_data(clean_features, clean_returns_data, random_state = 99)
	X_train, X_test = pf.preprocess_features(X_train, X_test, col_list)
	pred_model.fit(X_train, y_train.values.flatten())
	y_pred = pred_model.predict(X_test)
	cv_score = accuracy_score(y_test.values.flatten(), y_pred)
	print 'Score for test data with random state 99 = %f' %(cv_score)
##X_train.to_csv('X_train.csv')
##X_test.to_csv('X_test.csv')
##y_train.to_csv('y_train.csv')
##y_test.to_csv('y_test.csv')
