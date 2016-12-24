from sklearn import preprocessing
import numpy as np
import pandas as pd

def preprocess_features(X_train, X_test,col_list):
	'''
	'''
	imp = preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0)
	imp.fit(X_train)
	X_train = imp.transform(X_train)
	X_test = imp.transform(X_test)
	X_train = pd.DataFrame(preprocessing.scale(X_train), columns = col_list)
	X_test = pd.DataFrame(preprocessing.scale(X_test), columns = col_list)
	return X_train, X_test