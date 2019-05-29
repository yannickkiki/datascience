# -*- coding: utf-8 -*-
"""
Created on Wed May 29 08:41:34 2019

@author: Jean-Baptiste
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd






dataset_train = pd.read_csv('train.csv')
X_train = dataset_train.iloc[:,1:]/255.0
Y_train = dataset_train.iloc[:, 0].values

Y_train = Y_train.reshape(Y_train.shape[0], 1)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
Y_train = onehotencoder.fit_transform(Y_train).toarray()


X_train = (X_train.values).reshape(X_train.shape[0], 28, 28, 1)






from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, Y_train, batch_size = 420, epochs = 20)




dataset_test = pd.read_csv('test.csv')
#Y_id=dataset_test.iloc[:,-1]
X_test = dataset_test.iloc[:,0:]/255.0
X_test = (X_test.values).reshape(X_test.shape[0], 28, 28, 1)


Y_pred_softM=model.predict(X_test)

prediction=[]
for s in Y_pred_softM:
    i=-1
    pos=0
    maxi=0
    for p in s:
        i+=1
        if p>=maxi:
            pos=i
            maxi=p
    prediction.append(pos)




i=0
Y_id=[]
while i<28000:
    i+=1
    Y_id.append(i)
raw_data = {'ImageId':Y_id, 
        'Label': prediction}
df = pd.DataFrame(raw_data, columns = ['ImageId', 'Label'])
df.to_csv('submission_CNN-128-10-120-20.csv',index=False)








