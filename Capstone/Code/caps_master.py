import pandas as pd
import numpy as np
import F3_clean_data as cd
import F4_create_features as cf
import F5_create_training_data as ctd
import F6_preprocess_features as pf
import F7_train_models as tm

clean_fund_data = cd.clean_fund_data()
clean_returns_data = cd.clean_returns_data()
clean_features = cf.create_features(clean_fund_data)
X_train, X_test, y_train, y_test, ticker_idx_map, col_list = ctd.create_test_train_data(clean_features, clean_returns_data)

X_train.to_csv('X_train.csv')
X_test.to_csv('X_test.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')

#X_train = pd.read_csv('X_train.csv')
#X_test = pd.read_csv('X_test.csv')
#y_train = pd.read_csv('y_train.csv')
#y_test = pd.read_csv('y_test.csv')

X_train, X_test = pf.preprocess_features(X_train, X_test, col_list)
result = tm.train_classifier(X_train, X_test, y_train, y_test)

