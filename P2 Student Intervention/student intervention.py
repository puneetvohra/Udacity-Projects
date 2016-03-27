import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
student_data = pd.read_csv("C:/Users/systemcenter/Documents/Python Scripts/student_intervention/student-data.csv")

n_students = len(student_data)
n_features = len(student_data.columns)-1
n_passed = sum(student_data['passed'] == 'yes')
n_failed = sum(student_data['passed'] == 'no')
grad_rate = float(n_passed)*100/n_students
print "Total number of students: {}".format(n_students)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Number of features: {}".format(n_features)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows

def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)
print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))

# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset

train_rows = np.random.choice(X_all.index.values,num_train,replace = False)
X_train = X_all.ix[train_rows]
y_train = y_all.ix[train_rows]

X_test = X_all.drop(train_rows)
y_test = y_all.drop(train_rows)
print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])
# Note: If you need a validation set, extract it from within training data

import time

def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

# TODO: Choose a model, import it and instantiate an object
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
clf_dt = DecisionTreeClassifier()
clf_svm = LinearSVC()
clf_nb = GaussianNB()

# Fit model to training data
train_classifier(clf_dt, X_train, y_train)  # note: using entire training set here
train_classifier(clf_svm, X_train, y_train)  # note: using entire training set here
train_classifier(clf_nb, X_train, y_train)  # note: using entire training set here
#print clf  # you can inspect the learned model by printing it

from sklearn.metrics import f1_score

def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')

train_f1_score = predict_labels(clf_dt, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score)

test_f1_score = predict_labels(clf_dt, X_test, y_test)
print "F1 score for test set: {}".format(test_f1_score)
test_f1_score = predict_labels(clf_svm, X_test, y_test)
print "F1 score for test set: {}".format(test_f1_score)
test_f1_score = predict_labels(clf_nb, X_test, y_test)
print "F1 score for test set: {}".format(test_f1_score)

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
def fit_model(X, y):
    """ Tunes a decision tree regressor model using GridSearchCV on the input data X 
        and target labels y and returns this optimal model. """

    
    regressor = LinearSVC()

    # Set up the parameters we wish to tune
    parameters = {'C':np.logspace(-6, -1, 10)}

    # Make an appropriate scoring function
    scoring_function = make_scorer(f1_score,greater_is_better = True)

    # Make the GridSearchCV object
    reg = GridSearchCV(regressor,param_grid = parameters, scoring=scoring_function)

    # Fit the learner to the dataset to obtain the optimal model with tuned parameters
    reg.fit(X, y)

    # Return the optimal model
    return reg

print "Final model optimal parameters:", reg.best_params_


#Choose 3 supervised learning models that are available in scikit-learn, and appropriate for this problem. For each model:
#What are the general applications of this model? What are its strengths and weaknesses?
#Given what you know about the data so far, why did you choose this model to apply?
#Fit this model to the training data, try to predict labels (for both training and test sets), and measure the F1 score. Repeat this process with different training set sizes (100, 200, 300), keeping test set constant.
#Produce a table showing training time, prediction time, F1 score on training set and F1 score on test set, for each training set size.


#Based on the experiments you performed earlier, in 1-2 paragraphs explain to the board of supervisors what single model you chose as the best model. Which model is generally the most appropriate based on the available data, limited resources, cost, and performance?
#In 1-2 paragraphs explain to the board of supervisors in layman's terms how the final model chosen is supposed to work (for example if you chose a Decision Tree or Support Vector Machine, how does it make a prediction).
#Fine-tune the model. Use Gridsearch with at least one important parameter tuned and with at least 3 settings. Use the entire training set for this.
#What is the model's final F1 score?

'''We choose the Gaussian Naive Bayes Classifier based on three factors:
1) Reasonable training time (faster than Support Vector machines)
2) Very fast prediction time
3) Easy to understand and explain

Naiver Bayes prediction model predicts the most likely classification given the input data. 

We first train the model using the naive bayes assumption: we assume each feature is independent of 
every other feature given the classification of training data.
Based on this, we determine the probability of each classification for unseen data as follows:
Probability of Pass/Fail is product of two factors:
1) a priori probability of pass/fail; and 
2) product of probabilities of feature values given the classification in training data

The classification chosen for the unseen data input is simply the classification which has the highest
probability from above.
'''