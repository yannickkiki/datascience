import pandas as pd

#importing data
X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv")

#feature selection
features = ['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W',
            'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z',
            'linear_acceleration_X', 'linear_acceleration_Y',
            'linear_acceleration_Z']
n_samples, n_timesteps, n_features= len(y), 128, len(features)
X = X[features].values
y = y[['surface']].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y[:,0])
correspondances = list(labelencoder.classes_)
y = y.reshape(n_samples, 1)
onehotencoder = OneHotEncoder(categories='auto')
y = onehotencoder.fit_transform(y).toarray()

#scale values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#structure data
X = X.reshape(n_samples, n_timesteps, n_features)

#split X and y into train and test
#cut_idx = int(n_samples*0.8)
cut_idx = 3712 #total: 3810 , test: 98
train_X, test_X = X[:cut_idx], X[cut_idx:]
train_y, test_y = y[:cut_idx], y[cut_idx:]

# design network
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(units = 10, input_shape=(n_timesteps, n_features)))
model.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])
model.fit(train_X, train_y, epochs = 1)

#make a prediction on the test set
predicted_y_encoded = model.predict(test_X)
get_surfaces = lambda one_hot_encoded_values : [correspondances[list(line).index(max(line))] for line in one_hot_encoded_values]
predicted_y = get_surfaces(predicted_y_encoded)
real_y = get_surfaces(test_y)
accuracy = 0
for idx,prediction in enumerate(predicted_y):
   if prediction==real_y[idx]:
       accuracy+=1
accuracy /= len(predicted_y)
pd.to_pickle(model,"model.rnn")

#make predictions on the validation data
vX = pd.read_csv("X_test.csv")
vX = vX[features].values
sc = StandardScaler()
vX = sc.fit_transform(vX)
n_samples = int(len(vX)/n_timesteps)
vX = vX.reshape(n_samples, n_timesteps, n_features)
vy_encoded = model.predict(vX)
vy = get_surfaces(vy_encoded)
result = pd.DataFrame({"series_id" : list(range(n_samples)), "surface" : vy})
result.set_index("series_id",inplace=True)
result.to_csv("result2.csv")