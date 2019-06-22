# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 07:35:35 2019

@author: roma
"""

import pandas as pd
import numpy as np
import datetime as dt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


#load data
train = pd.read_csv("../input/sales_train.csv")
test = pd.read_csv("../input/test.csv")

train = train.loc[(train.item_cnt_day < 2000)]

train.date = train.date.apply(lambda x:dt.datetime.strptime(x, '%d.%m.%Y'))
train.date = train.date.apply(lambda x:dt.datetime.strftime(x,'%Y-%m'))
data = train.groupby(['date','item_id','shop_id']).sum().reset_index()

data = data[['date','item_id','shop_id','item_cnt_day']]

table = pd.pivot_table(data, values='item_cnt_day', index=['item_id', 'shop_id'],
                        columns=['date'], aggfunc=np.sum).reset_index()
table = table.fillna(0)

#get rid of those not in test data
data_inc = test.merge(table, on = ['item_id', 'shop_id'], how = 'left')
data_inc = data_inc.fillna(0)
data_inc = data_inc.iloc[:,3:]

# features Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data_inc.values.T)
dataset = dataset.T

X, y = dataset[:,:33], dataset[:,33:]

# reshape input to be [samples, time steps, features]
x_train = X.reshape((214200, 33, 1))
y_train = y.reshape((214200,1))

x_test = dataset[:,1:34]
x_test = x_test.reshape((214200, 33, 1))


batchSize = 30
def templateRnn():
    # Initialising the RNN
    regressor = Sequential()

    # Adding the input layer and the LSTM layer
    regressor.add(LSTM(64,return_sequences = True, batch_input_shape=(batchSize, x_train.shape[1], x_train.shape[2]),
               stateful = False))
  
    # Adding a second LSTM layer
    regressor.add(LSTM(units = 64, return_sequences = True))

    # Adding a third LSTM layer
    regressor.add(LSTM(units = 64, return_sequences = True))

    # Adding a fourth LSTM layer
    regressor.add(LSTM(units = 64))
    
    # Adding the output layer
    regressor.add(Dense(1))

    # Compiling the RNN
    regressor.compile(loss='mse', optimizer='rmsprop', metrics=['mean_squared_error'])

    return regressor

model = templateRnn()

# Fitting the RNN to the Training set
training=model.fit(x_train, y_train, batch_size = batchSize, epochs = 20, shuffle=False)

# make predictions
pred = model.predict(x_test, batch_size = batchSize)


# # creating submission file
submission = pd.DataFrame({'ID': test['ID'], 'item_cnt_month': pred.ravel()})
submission.to_csv('futureSales.csv',index=False)# # creating csv file from dataframe
