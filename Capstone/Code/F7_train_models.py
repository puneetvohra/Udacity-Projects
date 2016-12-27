import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.grid_search import GridSearchCV

def train_classifier(X_train, X_test, y_train, y_test):
	'''
	Traing three classifiers: Decision Tree, Gaussian Naive Bayes and Linear SVM. 
	Tries to refine the data by including correlation ranks.
	Calculates accuracy score for 20 different values of 'C' param in SVC. 
	Returns the fitted SVC model with C = 1, the predictions and the accuracy score of this model.
	'''
	clf_dt = DecisionTreeClassifier()
	clf_svm = LinearSVC()
	clf_nb = GaussianNB()

	clf_dt.fit(X_train, y_train.values.flatten())
	clf_svm.fit(X_train, y_train.values.flatten())
	clf_nb.fit(X_train, y_train.values.flatten())

	y_pred_dt = clf_dt.predict(X_test)
	y_pred_svm = clf_svm.predict(X_test)
	y_pred_nb = clf_nb.predict(X_test)

	score_dt = accuracy_score(y_test.values.flatten(), y_pred_dt)
	score_svm = accuracy_score(y_test.values.flatten(), y_pred_svm)
	score_nb = accuracy_score(y_test.values.flatten(), y_pred_nb)

	print 'Accuracy Score of Decision Tree model without refinement: %f' %(score_dt)
	print 'Accuracy Score of Gaussian NB model without refinement: %f' %(score_nb)
	print 'Accuracy Score of Linear SVM model without refinement: %f' %(score_svm)

	###########################Let's refine our models#######################
	train_corrs = X_train.iloc[:,-11:]
	train_corr_ranks = train_corrs.rank(axis = 1)
	X_train_ranks = pd.merge(X_train.iloc[:,:-11], train_corr_ranks,left_index = True, right_index = True)

	test_corrs = X_test.iloc[:,-11:]
	test_corr_ranks = test_corrs.rank(axis = 1)
	X_test_ranks = pd.merge(X_test.iloc[:,:-11], test_corr_ranks, left_index = True, right_index = True)

	clf_svm_rank = LinearSVC()
	clf_svm_rank.fit(X_train_ranks, y_train.values.flatten())
	y_pred_svm_rank = clf_svm_rank.predict(X_test_ranks)
	score_svm_rank = accuracy_score(y_test.values.flatten(), y_pred_svm_rank)
	print 'Accuracy Score of Linear SVM model with ranks: %f' %(score_svm_rank)


	X_train_ranks_corrs = pd.merge(X_train, train_corr_ranks,left_index = True, right_index = True)
	X_test_ranks_corrs = pd.merge(X_test, test_corr_ranks, left_index = True, right_index = True)
	clf_svm_rank_corr = LinearSVC()
	clf_svm_rank_corr.fit(X_train_ranks_corrs, y_train.values.flatten())
	y_pred_svm_rank_corr = clf_svm_rank_corr.predict(X_test_ranks_corrs)
	score_svm_rank_corr = accuracy_score(y_test.values.flatten(), y_pred_svm_rank_corr)
	print 'Accuracy Score of Linear SVM model with ranks and correlations: %f' %(score_svm_rank_corr)

	###################Let's fine tune our parameter C#####################################
	scoring_function = make_scorer(accuracy_score,greater_is_better = True)
	parameters = {'C':np.logspace(-1, 1, 20)}
	#best_clf = GridSearchCV(clf_svm,param_grid = parameters, scoring=scoring_function)
	#best_clf.fit(X_train,y_train.values.flatten())
	#y_pred_best = best_clf.predict(X_test)

	#best_score = accuracy_score(y_test, y_pred_best)
	#print 'Accuracy Score of Linear SVM model with optimized C: %f' %(best_score)
	#print 'Optimized C: %f' %(best_clf.best_estimator_.C)
	for param in parameters['C']:
		svm_clf = LinearSVC(C = param)
		svm_clf.fit(X_train, y_train.values.flatten())
		y_pred = svm_clf.predict(X_test)
		score = accuracy_score(y_test.values.flatten(), y_pred)
		print 'Accuracy Score of Linear SVM model with C = %f: %f' %(param, score)
	return svm_clf, y_pred_svm, score_svm