from funcs import read_data, process_X
data_X, validation_X, data_y = read_data()

X = process_X(data_X)

y = data_y[['surface']]

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y= train_test_split(X, y, test_size = 0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=500, n_jobs = -1)
model.fit(train_X,train_y)
model.score(train_X,train_y)
model.score(test_X,test_y)

val_X = process_X(validation_X)
predicted_y = model.predict(val_X)
from pandas import DataFrame
result = DataFrame({"series_id" : list(range(len(predicted_y))), "surface" : predicted_y})
result.set_index("series_id",inplace=True)
result.to_csv("result.csv")
