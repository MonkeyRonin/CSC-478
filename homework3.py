#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 20:47:36 2017

@author: Yiyang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor

#1.a
comm_df = pd.read_csv("/Users/Yiyang/Documents/CSC 478/communities.csv", na_values = ['?'])
comm_df.head()

pd.set_option("display.max_rows", 1000)

na = pd.DataFrame(comm_df.isnull().sum(0))
na

OPC_mean = comm_df.OtherPerCap.mean()
comm_df.OtherPerCap.fillna(OPC_mean, axis = 0, inplace = True)

pd.DataFrame(comm_df.corr())

comm_df.describe(include = "all")

comm_df_x = comm_df.drop(['state', 'communityname', 'ViolentCrimesPerPop'], axis = 1, inplace = False)
comm_df_y = comm_df['ViolentCrimesPerPop']

comm_df_x.head()
comm_df_y.head()

comm_df_x.shape
comm_df_y.shape

#1.b
x = np.array(comm_df_x)
x = np.array([np.concatenate((v, [1])) for v in x])
x

y = np.array(comm_df_y)
y

linreg = LinearRegression()
linreg.fit(x, y)

p = linreg.predict(x)
err = abs(p - y)
print(err[:10])

#Compute RMSE Train
total_error = np.dot(err, err)
rmse_train = np.sqrt(total_error/len(p))
print(rmse_train)

#Coefficients
print('Regression Coefficients: \n', linreg.coef_)

#Plot output
plt.plot(p, y, 'ro')
plt.plot([0, 1.5], [0, 1.5], 'g-')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.show()

#RMSE using 10-fold validation
kf = KFold(len(x), n_folds = 10)
xval_err = 0
for train, test in kf:
    linreg.fit(x[train], y[train])
    p = linreg.predict(x[test])
    e = p - y[test]
    xval_err += np.dot(e, e)

rmse_10cv = np.sqrt(xval_err/len(x))

print('RMSE on training: %.4f' %rmse_train)
print('RMSE on 10-fold CV: %.4f' %rmse_10cv)

#1.c
from sklearn import feature_selection
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

x = np.array(comm_df_x)
x = np.array([np.concatenate((v, [1])) for v in x])
y = np.array(comm_df_y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 33)

linreg = LinearRegression()

percentiles = range(1, 100, 5)
results = []
for i in range(1, 100, 5):
    fs = feature_selection.SelectPercentile(feature_selection.f_regression, percentile = i)
    x_train_fs = fs.fit_transform(x_train, y_train)
    scores = abs(cross_validation.cross_val_score(linreg, x_train_fs, y_train, cv=5))
    print(i, scores.mean())
    results = np.append(results, scores.mean())
    
optimal_percentile = np.where(results == results.max())[0]
print("Optimal percentile of feature:{0}".format(percentile[optimal_percentile]), "\n")
optimal_num_features = int(floor(percentiles[optimal_percentile] * len(comm_df_x.columns) / 100))
print ("Optimal number of features:{0}".format(optimal_num_features), "\n")

import pylab as pl
pl.figure()
pl.xlabel("Percentage of features selected")
pl.ylabel("Cross validation accuracy")
pl.plot(percentiles, results)

fs = feature_selection.SelectPercentile(feature_selection.f_regression, percentile = 96)
x_train_fs = fs.fit_transform(x_train, y_train)

for i in range(len(comm_df_x.columns.values)):
    if fs.get_support()[i]:
        print(comm_df_x.columns.values[i],'\t\t', fs.scores_[i])