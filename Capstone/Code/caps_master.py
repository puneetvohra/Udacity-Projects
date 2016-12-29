import pandas as pd
import F3_clean_data as cd
import F4_create_features as cf
import F5_create_training_data as ctd
import F6_preprocess_features as pf
import F7_train_models as tm
import caps_common as cc
import config as cfg

if cfg.fetch_raw_data:
	import F1_get_fundamental_data as gfd
	import F2_get_market_data as gmd
	gfd.get_fundamental_data()
	gmd.get_market_data()

clean_fund_data = cd.clean_fund_data()
cc.create_sector_counts(clean_fund_data)
clean_returns_data = cd.clean_returns_data()
clean_features = cf.create_features(clean_fund_data)
X_train, X_test, y_train, y_test, ticker_idx_map, col_list = ctd.create_test_train_data(clean_features, clean_returns_data)

X_train, X_test = pf.preprocess_features(X_train, X_test, col_list)
pred_model, predictions, score = tm.train_classifier(X_train, X_test, y_train, y_test)
prediction_df = pd.merge(pd.Series(predictions).to_frame('Predictions'), y_test, left_index = True, right_index = True).rename(index = ticker_idx_map)
prediction_df.columns = ['Predictions', 'Actual']
prediction_df.to_csv('data/predictions.csv')

cc.create_sector_prec_recall(prediction_df)


if cfg.cross_validate:
	from sklearn.metrics import accuracy_score
	X_train_cv, X_test_cv, y_train_cv, y_test_cv, ticker_idx_map_cv, col_list = ctd.create_test_train_data(clean_features, clean_returns_data, random_state = 99)
	X_train_cv, X_test_cv = pf.preprocess_features(X_train_cv, X_test_cv, col_list)
	#pred_model.fit(X_train_cv, y_train_cv.values.flatten())
	y_pred = pred_model.predict(X_test_cv)
	cv_score = accuracy_score(y_test_cv.values.flatten(), y_pred)
	print 'Score for test data with random state 99 = %f' %(cv_score)

