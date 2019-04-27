import pandas as pd
import numpy as np

#importing data
data_X = pd.read_csv("X_train.csv")
validation_X = pd.read_csv("X_test.csv")
data_y = pd.read_csv("y_train.csv")

n_samples, n_timesteps = len(data_y), 128

#just for test the code
"""cut_n = 200
data_X = data_X.iloc[0:128*cut_n]
data_y = data_y.iloc[0:cut_n]"""

def process_X(data_X):
    #rename columns (just abbreviation)
    data_X.rename(columns={'orientation_X':'oX','orientation_Y':'oY',
                         'orientation_Z':'oZ', 'orientation_W':'oW',
                         'angular_velocity_X':'avX', 'angular_velocity_Y':'avY',
                         'angular_velocity_Z':'avZ', 'linear_acceleration_X':'laX',
                         'linear_acceleration_Y':'laY',
                         'linear_acceleration_Z':'laZ'},inplace=True)
    
    #feature engineering
    ##orientation
    ###roll
    data_X["num_tan_roll"] = data_X.apply(lambda row: 2*(row.oW+row.oX+row.oY+row.oZ),
          axis = 1)
    data_X["den_tan_roll"] = data_X.apply(lambda row: 1-2*(row.oX**2+row.oY**2),
          axis = 1)
    data_X["tan_roll"] = data_X.apply(lambda row: row.num_tan_roll/row.den_tan_roll,
          axis = 1)
    data_X["roll"] = data_X.apply(lambda row: np.arctan2(row.num_tan_roll,row.den_tan_roll),
          axis = 1)
    ###pitch
    data_X["sin_pitch"] = data_X.apply(lambda row: 2*(row.oW*row.oY-row.oZ*row.oX),
          axis = 1)
    data_X["pitch"] = data_X.apply(lambda row: np.arcsin(row.sin_pitch), axis = 1)
    ###yaw
    data_X["num_tan_yaw"] = data_X.apply(lambda row : 2*(row.oW*row.oZ+row.oX*row.oY),
          axis = 1)
    data_X["den_tan_yaw"] = data_X.apply(lambda row : 1-2*(row.oY**2+row.oZ**2),
          axis = 1)
    data_X["tan_yaw"] = data_X.apply(lambda row: row.num_tan_yaw/row.den_tan_yaw,
          axis = 1)
    data_X["yaw"] = data_X.apply(lambda row: np.arctan2(row.num_tan_yaw,row.den_tan_yaw),
          axis = 1)
    ##angular velocity
    ###xy
    data_X["avXY"] = data_X.apply(lambda row: (row.avX**2+row.avY**2)**0.5, axis = 1)
    ###xz
    data_X["avXZ"] = data_X.apply(lambda row: (row.avX**2+row.avZ**2)**0.5, axis = 1)
    ###yz
    data_X["avYZ"] = data_X.apply(lambda row: (row.avY**2+row.avZ**2)**0.5, axis = 1)
    ###xyz
    data_X["avXYZ"] = data_X.apply(lambda row: (row.avX**2+row.avY**2+row.avZ**2)**0.5, axis = 1)
    ##linear acceleration
    ###xy
    data_X["laXY"] = data_X.apply(lambda row: (row.laX**2+row.laY**2)**0.5, axis = 1)
    ###xz
    data_X["laXZ"] = data_X.apply(lambda row: (row.laX**2+row.laZ**2)**0.5, axis = 1)
    ###yz
    data_X["laYZ"] = data_X.apply(lambda row: (row.laY**2+row.laZ**2)**0.5, axis = 1)
    ###xyz
    data_X["laXYZ"] = data_X.apply(lambda row: (row.laX**2+row.laY**2+row.laZ**2)**0.5 , axis = 1)
    
    #feature selection
    """features = ['row_id', 'series_id', 'measurement_number', 'oX', 'oY', 'oZ', 'oW',
           'avX', 'avY', 'avZ', 'laX', 'laY', 'laZ', 'num_tan_roll',
           'den_tan_roll', 'tan_roll', 'roll', 'sin_pitch', 'pitch', 'num_tan_yaw',
           'den_tan_yaw', 'tan_yaw', 'yaw', 'avXY', 'avXZ', 'avYZ', 'avXYZ',
           'laXY', 'laXZ', 'laYZ', 'laXYZ']
    features = ['oX', 'oY', 'oZ', 'oW',
           'avX', 'avY', 'avZ', 'laX', 'laY', 'laZ', 'roll', 'pitch', 'yaw', 
           'avXYZ','laXYZ']"""
    features = ['oX', 'oY', 'oZ', 'oW',
           'avX', 'avY', 'avZ', 'laX', 'laY', 'laZ', 'num_tan_roll',
           'den_tan_roll', 'tan_roll', 'roll', 'sin_pitch', 'pitch', 'num_tan_yaw',
           'den_tan_yaw', 'tan_yaw', 'yaw', 'avXY', 'avXZ', 'avYZ', 'avXYZ',
           'laXY', 'laXZ', 'laYZ', 'laXYZ']
    #features = data_X.columns.drop(['row_id','series_id', 'measurement_number'])
    '''
    import random
    features = random.choices(features, k = 10)
    '''
    X = data_X[features].values
    return X, len(features)

X, n_features = process_X(data_X)
vX, _ = process_X(validation_X)
all_X = np.concatenate((X, vX), axis=0)
#scale values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(all_X)
X = sc.transform(X)
vX = sc.transform(vX)
#structure data
X = X.reshape(n_samples, n_timesteps, n_features)
vX = vX.reshape(3816, n_timesteps, n_features)

y = data_y[['surface']].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y[:,0])
correspondances = list(labelencoder.classes_)
y = y.reshape(n_samples, 1)
onehotencoder = OneHotEncoder(categories='auto')
y = onehotencoder.fit_transform(y).toarray()

#split X and y into train and test
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y= train_test_split(X, y, test_size = 0.2,
                                                   random_state = 0)

# design network
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(units = 28, input_shape=(n_timesteps, n_features)))
model.add(Dense(units = 19, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])
model.fit(train_X, train_y, batch_size = 24, epochs = 200)

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
#pd.to_pickle(model,"Model.rnn")

#make predictions on the validation data
vy_encoded = model.predict(vX)
vy = get_surfaces(vy_encoded)
result = pd.DataFrame({"series_id" : list(range(len(vy))), "surface" : vy})
result.set_index("series_id",inplace=True)
result.to_csv("result67.csv")
