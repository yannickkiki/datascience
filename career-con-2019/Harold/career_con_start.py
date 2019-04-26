# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:13:26 2019

@author: HaroldKS
"""
INPUT_DIR = 'data'


import pandas as pd
import numpy as np
#feature engineering stuffs
import math
def quaternion_to_euler(x, y, z, w):
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll =  np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    print(t2.min())
    #print(t2.max())
    #t2.apply(lambda x : -1.0 if t2 < -1.0 else t2)
    #t2 = +1.0 if t2 > +1.0 else t2
    #t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw =  np.arctan2(t3, t4)
    return [yaw, pitch, roll]




X_dataset =  pd.read_csv(f'{INPUT_DIR}/X_train.csv')
y_dataset =  pd.read_csv(f'{INPUT_DIR}/y_train.csv')

X_dataset.info()

lol = quaternion_to_euler(X_dataset.orientation_X, X_dataset.orientation_Y, X_dataset.orientation_Z, X_dataset.orientation_W)

X_dataset['yaw'] = lol[0]
X_dataset['pitch'] = lol[1]
X_dataset['roll'] = lol[2]

for name in X_dataset.columns:
    print(f'name : {name} \t min value : {X_dataset[name].min()} \t max value : {X_dataset[name].max()}')
    

#Used to get the numbers of serie we have in the dataset. Good good good
group = X_dataset.groupby('series_id').count()

n_samples, timestep = len(group), group['row_id'][0]
#Among the features think the row_id is not necessary so will drop it  'orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W'

X_dataset.drop(['row_id', 'series_id', 'measurement_number'], axis = 1, inplace = True)

features = len(X_dataset.columns)

#Transform Y
y_train = y_dataset.surface
len(y_train.unique())

#Xcale data for better training
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, LabelBinarizer

#min_max_scaler = MinMaxScaler()
X_train = X_dataset.values
#X_train = min_max_scaler.fit_transform(X_train)


label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

bin_encoder = LabelBinarizer()
y_train = bin_encoder.fit_transform(y_train)



classes = list(label_encoder.classes_)

print(classes)


#Reshaping the dataset for time serie purpose
X_train = np.reshape(X_train, (n_samples, timestep, features))


from keras.models import Sequential
from keras.layers import Dense, LSTM

classifier = Sequential()
classifier.add(LSTM(units=13, input_shape =(timestep, features), activation='relu'))
#classifier.add(LSTM(units=10, input_shape =(timestep, features), activation='relu'))
classifier.add(Dense(units = 13, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 13, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units=9,  kernel_initializer = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy', 'categorical_crossentropy'])
classifier.summary()
classifier.fit(X_train, y_train, epochs=10, batch_size=16)



X_test =  pd.read_csv(f'{INPUT_DIR}/X_test.csv')
lol_test = quaternion_to_euler(X_test.orientation_X, X_test.orientation_Y, X_test.orientation_Z, X_test.orientation_W)

X_test['yaw'] = lol_test[0]
X_test['pitch'] = lol_test[1]
X_test['roll'] = lol_test[2]


group_test = X_test.groupby('series_id').count()
X_test.drop(['row_id', 'series_id', 'measurement_number'], axis = 1, inplace = True)
X_test = X_test.values
X_test = min_max_scaler.transform(X_test)
X_test = np.reshape(X_test, (3816, timestep, features))

y_pred = classifier.predict(X_test)

get_encoded_values = list()
for row in y_pred:
    get_encoded_values.append(row.argmax())

surfaces =  list(label_encoder.inverse_transform(get_encoded_values))

series_id = [ i for i in range(3816)]

raw_data = {'series_id' : series_id,
            'surface' : surfaces}

df = pd.DataFrame(raw_data, columns=['series_id', 'surface'])
df.to_csv('hk_submit.csv', index= False)
hk = pd.read_csv('hk_submit.csv', sep=',', decimal=',')

