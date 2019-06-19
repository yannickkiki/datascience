from pandas import read_csv, get_dummies, DataFrame
df = read_csv("dataset/train.csv")
val_df = read_csv("dataset/test.csv")

features = df.columns.drop(["Id","SalePrice"])
target = ["SalePrice"]

X, y = df[features], df[target]
X= X.fillna(X.mean())
X = get_dummies(X)

vX= val_df[features]
vX= vX.fillna(vX.mean())
vX = get_dummies(vX)

X, vX = X.align(vX, join='inner', axis=1)

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y= train_test_split(X, y, test_size = 0.1)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(train_X,train_y)

pred_y = model.predict(test_X)
from sklearn.metrics import mean_squared_error
import numpy as np
rmse = mean_squared_error(np.log(test_y), np.log(pred_y))**0.5

val_y = model.predict(vX)
result = DataFrame({"Id" : list(range(1461,1461+len(val_y))), "SalePrice" : val_y})
result.set_index("Id",inplace=True)
result.to_csv("v0.csv")
