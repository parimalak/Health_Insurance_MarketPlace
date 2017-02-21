# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 21:31:42 2017

@author: Parimala Killada

"""
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#importing data from csv file
#data = pd.read_csv('I:/Parimala/dropbox/Dropbox/Thesis/final1415.csv')
data = pd.read_csv('I:/Parimala/dropbox/Dropbox/Thesis/final141516.csv')
#data = pd.read_csv('C:/Users/pkillad/Dropbox/Thesis/final1415.csv')
#Making columns categorical type
#data['age'] = data.age.astype('category')
data['State'] = data.State.astype('category')
data['County'] = data.County.astype('category')
data['Metal_level'] = data.Metal_level.astype('category')
data['Plan_Type'] = data.Plan_Type.astype('category')
data['Source'] = data.Source.astype('category')
data = data.dropna()

#
statecodes = data.State.cat.categories
countycodes = data.County.cat.categories
mlevel_codes =data.Metal_level.cat.categories
plan_type_codes = data.Plan_Type.cat.categories
source_codes = data.Source.cat.categories

cat_columns = data.select_dtypes(['category']).columns
data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
data_used = data[data.columns.difference(['Plan_id','value','couple'])]
data_target =data['value']
x_train, x_test, y_train, y_test = \
train_test_split(data_used, data_target, test_size=0.0001)

#min_max_scaler = preprocessing.MinMaxScaler()

#scalerX = StandardScaler().fit(x_train)
#scalery = StandardScaler().fit(y_train)

scalerX = MinMaxScaler().fit(x_train)
scalery = MinMaxScaler().fit(y_train)

x_train = scalerX.transform(x_train)
y_train = scalery.transform(y_train)
x_test = scalerX.transform(x_test)
y_test = scalery.transform(y_test)

print(np.max(x_train), np.min(x_train),\
np.mean(x_train), np.max(y_train), np.min(y_train), np.mean(y_train))

# Create linear regression object
regr = linear_model.LinearRegression(normalize=True)

# Train the model using the training sets
regr.fit(x_train, y_train) 

#Co-eeficients
print(regr.coef_)
#Mean-squared Error
print("Mean squared error: %.4f"
      % np.mean((regr.predict(x_test) - y_test) ** 2))
#    # Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_test, y_test))

acty = scalery.inverse_transform(y_test)
caly = scalery.inverse_transform((regr.predict(x_test)))
plt.figure(1)
#plt.subplot(1, 2, 1)
plt.title('actual vs predicted price using Linear regression')
plt.xlabel('testing data')
plt.xscale('log')
plt.ylabel('Predicted Value')
plt.plot(acty,'rx',label='actual')
plt.plot(caly,'bo',label='predicted')
plt.legend(loc = 'upper right')
#accuracy Score
#from sklearn.metrics import accuracy_score
#from pandas.tools.plotting import parallel_coordinates
#from pandas.tools.plotting import lag_plot

#Decision tree regression
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf = clf.fit(x_train, y_train)
dscaly =clf.predict(x_test)
dcaly = scalery.inverse_transform(dscaly)
sc=clf.score(x_test,y_test)
print(sc)
plt.figure(2)
plt.title('actual vs predicted price using Decision Tree regression')
plt.xlabel('actual value')
plt.ylabel('Predicted Value')
plt.plot(acty,'rx',dcaly,'bo')

   
from sklearn.ensemble import AdaBoostRegressor
regr_2 = AdaBoostRegressor(clf,n_estimators=30, random_state=None)
regr_2.fit(x_train, y_train)
y_2 = regr_2.predict(x_test)   
y_2_act = scalery.inverse_transform(y_2)
mse = mean_squared_error(y_test, y_2)
plt.figure(3)
plt.title('actual vs predicted price using AdaBoost Regressor boosting Decision Tree regression')
plt.xlabel('actual value')
plt.ylabel('Predicted Value')
plt.plot(acty,'rx',y_2_act,'bo')

from sklearn import ensemble

params = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 4,
          'learning_rate': 0.01, 'loss': 'ls'}
clf1 = ensemble.GradientBoostingRegressor(**params)

clf1.fit(x_train, y_train)
mse = mean_squared_error(y_test, clf1.predict(x_test))

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf1.staged_predict(x_test)):
    test_score[i] = clf1.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf1.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

gdcaly = scalery.inverse_transform(y_pred)
plt.figure(4)
plt.title('actual vs predicted price using Graduate boosting regression')
plt.xlabel('actual value')
plt.ylabel('Predicted Value')
plt.plot(acty,'rx',gdcaly,'bo')
plt.legend(loc='upper right')

feature_importance = clf1.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
cols = data_used.columns

plt.yticks(pos, cols[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
