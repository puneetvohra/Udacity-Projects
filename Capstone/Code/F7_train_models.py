import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.grid_search import GridSearchCV

def train_classifier(X_train, X_test, y_train, y_test):
	'''
	'''
	scoring_function = make_scorer(accuracy_score,greater_is_better = True)
	parameters = {'C':np.logspace(-1, 1, 50)}
	
	clf_dt = DecisionTreeClassifier()
	clf_svm = LinearSVC()
	clf_nb = GaussianNB()

	a = X_train.iloc[:,-11:]
	a = a.rank(axis = 1)
	c = pd.merge(X_train, a,left_index = True, right_index = True)

	clf_dt.fit(X_train, y_train.values.flatten())
	clf_svm.fit(X_train, y_train.values.flatten())
	clf_nb.fit(X_train, y_train.values.flatten())

	d = X_test.iloc[:,-11:]
	d = d.rank(axis = 1)
	e = pd.merge(X_test, d, left_index = True, right_index = True)

	y_pred_dt = clf_dt.predict(X_test)
	y_pred_svm = clf_svm.predict(X_test)
	y_pred_nb = clf_nb.predict(X_test)

	score_dt = accuracy_score(y_test.values.flatten(), y_pred_dt)
	score_svm = accuracy_score(y_test.values.flatten(), y_pred_svm)
	score_nb = accuracy_score(y_test.values.flatten(), y_pred_nb)

	best_clf = GridSearchCV(clf_svm,param_grid = parameters, scoring=scoring_function)
	best_clf.fit(X_train,y_train.values.flatten())
	y_pred_best = best_clf.predict(X_test)

	best_score = accuracy_score(y_test, y_pred_best)
	return y_pred_best, best_score