# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 08:57:34 2019

@author: roma
"""
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y_train=train["SalePrice"]
cols=["Id","SalePrice","PoolQC"]
X_train= train.drop(cols,axis=1)

X_train= X_train.fillna(X_train.mean())
X_train = pd.get_dummies(X_train)

col=["Id","PoolQC"]
X_test= test.drop(col,axis=1)

X_test= X_test.fillna(X_test.mean())
X_test = pd.get_dummies(X_test)


X_train, X_test = X_train.align(X_test, join='inner', axis=1)

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y= train_test_split(X_train,y_train, test_size = 0.1)

from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05,max_depth=4,max_features='sqrt',min_samples_leaf=15,min_samples_split=10,loss='huber',random_state=5)
regressor.fit(train_X,train_y)

y_pred = regressor.predict(X_test)
result = pd.DataFrame({"Id" : list(range(1461,1461+len(y_pred))), "SalePrice" : y_pred})
result.set_index("Id",inplace=True)
result.to_csv("hP.csv")
