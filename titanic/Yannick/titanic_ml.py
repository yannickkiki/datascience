#from collections import Counter
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor
import utils

#read datasets
train_data, test_data  = pd.read_csv("train.csv"), pd.read_csv("test.csv")

#select relevant features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
        'Fare', 'Cabin', 'Embarked']

#subset dataset
#   #X
train_X, test_X = train_data[features], test_data[features]
#v = train_X.isnull().sum()

#Missing values handling
"""
#find columns with missing value
cols_with_missing = [col for col in features if train_X[col].isnull().any() \
                     or test_X[col].isnull().any()]

#method 1: drop these columns
train_X = train_X.drop(cols_with_missing, axis=1)
test_X = test_X.drop(cols_with_missing, axis=1)
"""

#method 2: imputing
train_X = train_X.fillna(train_X.mean())
test_X = test_X.fillna(test_X.mean())
#print(X.dtypes.sample(8))
#   #y
train_y = train_data.Survived

#one hot encoding
train_X, test_X = pd.get_dummies(train_X), pd.get_dummies(test_X)

#align columns to fix eventual columns mix 
train_X, test_X = train_X.align(test_X, join='inner', axis=1)

model = DecisionTreeRegressor(random_state=1)
#model = RandomForestRegressor(random_state=1)
#model = XGBRegressor()
model.fit(train_X,train_y)
predicted_y = [int(round(value)) for value in model.predict(test_X)]
#err = mean_absolute_error(train_y,predicted_y)
utils.create_submit_file("submission0.csv", 892, predicted_y)