import pandas as pd
import numpy as np
from funcs import feature_X

#importing data
data_X = pd.read_csv("X_train.csv")
validation_X = pd.read_csv("X_test.csv")
data_y = pd.read_csv("y_train.csv")

"""
#just for test the code
cut_n = 100
data_X = data_X.iloc[0:128*cut_n]
data_y = data_y.iloc[0:cut_n]
validation_X = validation_X.iloc[0:128*cut_n]
"""

n_samples, n_timesteps = len(data_y), 128
validation_n_samples = len(validation_X)//n_timesteps

y = data_y[['surface']].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y[:,0])
correspondances = list(labelencoder.classes_)
y = y.reshape(n_samples, 1)
onehotencoder = OneHotEncoder(categories='auto')
y = onehotencoder.fit_transform(y).toarray()

X = feature_X(data_X)
features = X.columns
n_features = len(features)
vX = feature_X(validation_X)
X, vX = X.values, vX.values
all_X = np.concatenate((X, vX), axis=0)

#scale values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(all_X)
X = sc.transform(X)
vX = sc.transform(vX)
#structure data
X = X.reshape(n_samples, n_timesteps, n_features)
vX = vX.reshape(validation_n_samples, n_timesteps, n_features)

#split X and y into train and test
#from sklearn.model_selection import train_test_split
#train_X, test_X, train_y, test_y= train_test_split(X, y, test_size = 0.2)

# design network
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(units = 32, return_sequences = True, input_shape=(n_timesteps, n_features)))
model.add(LSTM(units = 32, return_sequences = True))
model.add(LSTM(units = 32))
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dense(units = 9, activation = 'softmax'))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X, y, batch_size = 32, epochs = 200)

ev = model.evaluate(X, y)
#ev_test = model.evaluate(test_X, test_y)
#pd.to_pickle(model, "model0.98.rnn")

get_surfaces = lambda one_hot_encoded_values : [correspondances[list(line).index(max(line))] for line in one_hot_encoded_values]

y_true = get_surfaces(y)
y_pred = get_surfaces(model.predict(X))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred, labels = correspondances)

#make predictions on the validation data
vy_encoded = model.predict(vX)
vy = get_surfaces(vy_encoded)
result = pd.DataFrame({"series_id" : list(range(len(vy))), "surface" : vy})
result.set_index("series_id",inplace=True)
result.to_csv("result"+str(ev)+".csv")
