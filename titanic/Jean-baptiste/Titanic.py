# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:28:30 2019

@author: ASUS
"""





# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd








# Importing the dataset
dataset_train = pd.read_csv('train.csv')
X = dataset_train.iloc[:, [2,4,5,6,7,9]].values




print(X)
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
print(X)


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:6])
X[:, 1:6] = imputer.transform(X[:, 1:6])
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
print(X)

onehotencoder1 = OneHotEncoder(categorical_features = [0,1,3])
X=onehotencoder1.fit_transform(X).toarray()



Y = dataset_train.iloc[:, 1].values






dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, [1,3,4,5,6,8]].values
print(X_test)
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X_test[:, 1] = labelencoder_X_1.fit_transform(X_test[:, 1])
print(X_test)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, 1:6])
X_test[:, 1:6] = imputer.transform(X_test[:, 1:6])
X_test[:, 1] = labelencoder_X_1.fit_transform(X_test[:, 1])
print(X_test)

onehotencoder = OneHotEncoder(categorical_features = [0,1,3])
X_test=onehotencoder.fit_transform(X_test).toarray()





gender_submission = pd.read_csv('gender_submission.csv')
Y_id = gender_submission.iloc[:, 0].values
print(Y_id)

 #Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size = 0.10)


from sklearn import metrics 
from sklearn.linear_model import LogisticRegression 
classifierLR = LogisticRegression()
classifierLR.fit(X_train,Y_train)

Y_pred1=classifierLR.predict(X_test)
from sklearn.metrics import confusion_matrix
cmRL = confusion_matrix(Y_test,Y_pred1)



print("----------------SVM----------------------")
from sklearn.svm import SVC
classifierSVM=SVC(kernel = 'linear')
classifierSVM.fit(X_train,Y_train)

Y_pred2=classifierSVM.predict(X_test)
from sklearn.metrics import confusion_matrix
cmSVM = confusion_matrix(Y_test,Y_pred2)




print("-----------------------------Naive Bayes------------")
from sklearn.naive_bayes import GaussianNB
classifierSVM=GaussianNB()
classifierSVM.fit(X_train,Y_train)

Y_pred3=classifierSVM.predict(X_test)
from sklearn.metrics import confusion_matrix
cmNB = confusion_matrix(Y_test,Y_pred3)




import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X, Y, batch_size = 100, epochs = 50)
y_pred4 = classifier.predict(X_test)
y_pred4 = (y_pred4 > 0.5)



y_Pred=[]
for pr in y_pred4:
    if pr==True:
        y_Pred.append(1)
    else:
        y_Pred.append(0)




raw_data = {'PassengerId':Y_id, 
        'Survived': y_Pred}
df = pd.DataFrame(raw_data, columns = ['PassengerId', 'Survived'])
df.to_csv('submission_RN-blaSate4.csv',index=False)
