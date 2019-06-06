from pandas import read_csv
data_X = read_csv("datasets/X_train.csv")
validation_X = read_csv("datasets/X_test.csv")
data_y = read_csv("datasets/y_train.csv")

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

from funcs import feature_X
X = feature_X(data_X)
features = X.columns
n_features = len(features)
vX = feature_X(validation_X)

X, vX = X.values, vX.values
from numpy import concatenate
all_X = concatenate((X, vX), axis=0)

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
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y= train_test_split(X, y, test_size = 0.1)

# design network
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(units = 32, return_sequences = True, input_shape=(n_timesteps, n_features)))
model.add(LSTM(units = 32, return_sequences = True))
model.add(LSTM(units = 32))
model.add(Dense(units = 9, activation = 'softmax'))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_X, train_y, batch_size = 64, epochs = 10)

ev = model.evaluate(train_X, train_y)
ev = model.evaluate(test_X, test_y)
#pd.to_pickle(model, "model0.98.rnn")

get_surfaces = lambda one_hot_encoded_values : [correspondances[list(line).index(max(line))] for line in one_hot_encoded_values]

y_true = get_surfaces(test_y)
y_pred = get_surfaces(model.predict(test_X))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_true, y_pred, labels = correspondances)
accuracy = accuracy_score(y_true, y_pred)

#make predictions on the validation data
vy_encoded = model.predict(vX)
vy = get_surfaces(vy_encoded)
from pandas import DataFrame
result = DataFrame({"series_id" : list(range(len(vy))), "surface" : vy})
result.set_index("series_id",inplace=True)
result.to_csv("result"+str(ev)+".csv")
