# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:43:29 2017
s
@author: MarioNaia
"""

#http://scikit-learn.org/stable/auto_examples/plot_cv_predict.html#sphx-glr-auto-examples-plot-cv-predict-py

import numpy as np
import pandas as pd
#First, let's load the data, and split it in four. It is the fold used the authors of the original paper.



train = pd.read_csv(".../train.csv")
X1= train.iloc[:,:-1]
y1 = train.iloc[:,-1]

test = pd.read_csv(".../test.csv")
X= test.iloc[:,:-1]
y = test.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.25, random_state=0) #0.25 to divide in 4


#################the models ######################################

from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
sgdreg = linear_model.SGDRegressor()
br=linear_model.BayesianRidge()
lass=linear_model.Lasso()
larsModel=linear_model.Lars(n_nonzero_coefs=1)

from sklearn.neural_network import MLPRegressor
mlpreg=MLPRegressor(hidden_layer_sizes=(10,), max_iter=100000)

from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor()


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
rng = np.random.RandomState(1)
regr_1 = DecisionTreeRegressor(max_depth=4)

#http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html
#http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_regression.html#sphx-glr-auto-examples-ensemble-plot-adaboost-regression-py
#regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300, random_state=rng)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300, random_state=rng)

#http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py
#http://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.linear_model import ElasticNet
enet = ElasticNet(alpha=0.1, l1_ratio=0.7)

#boston = datasets.load_boston()
#y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:

#predicted = cross_val_predict(lr, boston.data, y, cv=10)
predicted = cross_val_predict(lr, X1, y1, cv=10)
predicted2 = cross_val_predict(sgdreg, X1, y1, cv=10)
predicted3 = cross_val_predict(gbr, X1, y1, cv=10)
predicted4 = cross_val_predict(br, X1, y1, cv=10)
predicted5 = cross_val_predict(lass, X1, y1, cv=10)
predicted6 = cross_val_predict(mlpreg, X1, y1, cv=10)
predicted7 = cross_val_predict(regr_1, X1, y1, cv=10)
#predicted8 = cross_val_predict(regr_2, X1, y1, cv=10)
predicted8 = cross_val_predict(enet, X1, y1, cv=10)
predicted9 = cross_val_predict(larsModel, X1, y1, cv=10)


fig, ax = plt.subplots()
ax.scatter(y1, predicted)
ax.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(y1, predicted2)
ax.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(y1, predicted3)
ax.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


fig, ax = plt.subplots()
ax.scatter(y1, predicted4)
ax.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(y1, predicted5)
ax.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(y1, predicted6)
ax.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(y1, predicted7)
ax.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(y1, predicted8)
ax.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(y1, predicted9)
ax.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
####################################################

"""
KNeighborsClassifier(),
SVC(kernel="linear"),
GaussianProcessClassifier(),
DecisionTreeClassifier(),
RandomForestClassifier(),
MLPClassifier(),
AdaBoostClassifier(),
GaussianNB(),
QuadraticDiscriminantAnalysis()
"""    


results = {}
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr, X_train, y_train, cv=5)
results["linear regression"] = scores

scores = cross_val_score(sgdreg, X_train, y_train, cv=5)
results["SGDRegressor"] = scores

scores = cross_val_score(gbr, X_train, y_train, cv=5)
results["Gradient Boosting Regressor"] = scores

scores = cross_val_score(br, X_train, y_train, cv=5)
results["Bayesian Ridge"] = scores

scores = cross_val_score(lass, X_train, y_train, cv=5)
results["Lasso"] = scores
  
scores = cross_val_score(mlpreg, X_train, y_train, cv=5)
results["MLPRegressor"] = scores
  
scores = cross_val_score(regr_1, X_train, y_train, cv=5)
results["Decision Tree Regressor"] = scores
    
scores = cross_val_score(enet, X_train, y_train, cv=5)
results["Elastic Net"] = scores
    
scores = cross_val_score(larsModel, X_train, y_train, cv=5)
results["L A R S"] = scores


#scores = cross_val_score(regr_2, X_train, y_train, cv=5)
#results["AdaBoost Regressor"] = scores

for name, scores in results.items():
    print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (name, 100*scores.mean(), 100*scores.std() * 2))   
    
clf = GradientBoostingRegressor()
clf.fit(X_train, y_train)
y_eval = clf.predict(X)
#print(y_eval)
prediction = pd.DataFrame(y_eval, columns=['predictions']).to_csv('outGradientBoostingScikit.csv')


clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)
y_eval = clf.predict(X)
prediction = pd.DataFrame(y_eval, columns=['predictions']).to_csv('outLinearRegressionScikit.csv')

clf = linear_model.BayesianRidge()
clf.fit(X_train, y_train)
y_eval = clf.predict(X)
prediction = pd.DataFrame(y_eval, columns=['predictions']).to_csv('outBayesianRidgeScikit.csv')


clf = linear_model.Lasso()
clf.fit(X_train, y_train)
y_eval = clf.predict(X)
prediction = pd.DataFrame(y_eval, columns=['predictions']).to_csv('outLasso.csv')


clf = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100000)
clf.fit(X_train, y_train)
y_eval = clf.predict(X)
prediction = pd.DataFrame(y_eval, columns=['predictions']).to_csv('outMLPregressor.csv')


clf = DecisionTreeRegressor(max_depth=4)
clf.fit(X_train, y_train)
y_eval = clf.predict(X)
prediction = pd.DataFrame(y_eval, columns=['predictions']).to_csv('outDecisionTreeRegressor.csv')


clf = ElasticNet(alpha=0.1, l1_ratio=0.7)
clf.fit(X_train, y_train)
y_eval = clf.predict(X)
prediction = pd.DataFrame(y_eval, columns=['predictions']).to_csv('outElasticNet.csv')

regr_2.fit(X_train, y_train)
y_eval = regr_2 .predict(X)
prediction = pd.DataFrame(y_eval, columns=['predictions']).to_csv('outAdaBoostRegressor.csv')

clf = linear_model.Lars(n_nonzero_coefs=1)
clf.fit(X_train, y_train)
y_eval = clf.predict(X)
prediction = pd.DataFrame(y_eval, columns=['predictions']).to_csv('outLARS.csv')

"""
clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300, random_state=rng)
clf.fit(X_train, y_train)
y_eval = clf.predict(X)
prediction = pd.DataFrame(y_eval, columns=['predictions']).to_csv('outAdaBoostRegressor.csv')
"""


from pyearth import Earth
clf = Earth()
clf.fit(X_train, y_train)
y_eval = clf.predict(X)
prediction = pd.DataFrame(y_eval, columns=['predictions']).to_csv('outMARS.csv')

