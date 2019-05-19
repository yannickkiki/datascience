
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math

def softmax(z):
    z_exp = [math.exp(i) for i in z]
    sum_z_exp = sum(z_exp)
    return [i / sum_z_exp for i in z_exp]

dataset_X_train = pd.read_csv('X_train.csv')
dataset_Y_train = pd.read_csv('Y_train.csv')
dataset_X_test = pd.read_csv('X_test.csv')

X_train=dataset_X_train.iloc[:, [3,4,5,6,7,8,9,10,11,12]].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler,StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

#Manual reshaping instead of using the numpy function (just to try )  ;)
X_train = (np.split(X_train,3810))
temp=[]
for tr in X_train:
    temp.append(tr)
X_train=np.array(temp)

# Getting of Y_train values
dataset_Y_train=dataset_Y_train.iloc[:,[2]].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
dataset_Y_train[:, 0] = labelencoder_X_1.fit_transform(dataset_Y_train[:, 0])


#Getting of text input
test=dataset_X_test
X_test=test.iloc[:, [3,4,5,6,7,8,9,10,11,12]].values
sc = MinMaxScaler()
X_test = sc.fit_transform(X_test)


#Manual reshaping again :)
X_test = (np.split(X_test,3816))
temp=[]
for te in X_test:
    temp.append(te)
X_test=np.array(temp)


onehotencoder = OneHotEncoder(categorical_features = [0])
dataset_Y_train=onehotencoder.fit_transform(dataset_Y_train).toarray()
#print(dataset_Y_train)
labels=list(labelencoder_X_1.classes_)
Y_train=(dataset_Y_train)


#RNN model building
classiffier = Sequential()
classiffier.add(LSTM(units = 10,return_sequences=True,input_shape = (X_train.shape[1], 10)))
classiffier.add(LSTM(units = 10,activation='relu'))
classiffier.add(Dense(units = 9,kernel_initializer='uniform', activation = 'softmax'))
classiffier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classiffier.fit(X_train, Y_train, epochs = 100, batch_size = 50)

softMaxComput=[]
for pred in classiffier.predict(X_test):
    softMaxComput.append(softmax(pred))
prediction=[]
for s in softMaxComput:
    i=-1
    pos=0
    maxi=0
    for p in s:
        i+=1
        if p>=maxi:
            pos=i
            maxi=p
    prediction.append(labels[pos])



i=-1
Y_id=[]
while i<3815:
    i+=1
    Y_id.append(i)
raw_data = {'series_id':Y_id, 
        'surface': prediction}
df = pd.DataFrame(raw_data, columns = ['series_id', 'surface'])
df.to_csv('submission_RNN.csv',index=False)