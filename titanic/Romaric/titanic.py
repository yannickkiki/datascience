# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:23:07 2019

@author: roma
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 23:01:24 2019

@author: roma
"""


# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
titanic_train= pd.read_csv('train.csv')
titanic_test=pd.read_csv('test.csv')

#titanic train part

cols=["PassengerId","Survived","Ticket","Cabin","Name"]
y_titanic_train=titanic_train['Survived'].values
X_titanic_train=titanic_train.drop(cols,axis=1)


for dt in X_titanic_train :
    mean=X_titanic_train["Age"].mean()
    std=titanic_test["Age"].std()
    is_null=X_titanic_train["Age"].isnull().sum()
    rand_age=np.random.randint(mean-std,mean+std,size=is_null)
    age_slice=X_titanic_train["Age"].copy()
    age_slice[np.isnan(age_slice)]=rand_age
    X_titanic_train["Age"]=age_slice
    X_titanic_train["Age"]=X_titanic_train["Age"].astype(int)

for dt in X_titanic_train:
    X_titanic_train['re']=X_titanic_train['SibSp']+X_titanic_train['Parch']
    X_titanic_train.loc[X_titanic_train['re']>0,'voySeul']='No'
    X_titanic_train.loc[X_titanic_train['re']==0,'voySeul']='Yes'
    
dummies=[]
dum=['Pclass','Sex','Embarked','voySeul']
for col in dum:
    dummies.append(pd.get_dummies(X_titanic_train[col]))
titanic_dummies=pd.concat(dummies,axis=1)
X_titanic_train=pd.concat((X_titanic_train,titanic_dummies),axis=1)
X_titanic_train=X_titanic_train.drop(dum,axis=1)
    
X_titanic_train=X_titanic_train.iloc[:,:].values

#titanic test part

cols=["PassengerId","Name","Ticket","Cabin"]
X_titanic_test=titanic_test.drop(cols,axis=1)

for dt in X_titanic_test :
    mean=X_titanic_test["Age"].mean()
    std=titanic_test["Age"].std()
    is_null=X_titanic_test["Age"].isnull().sum()
    rand_age=np.random.randint(mean-std,mean+std,size=is_null)
    age_slice=X_titanic_test["Age"].copy()
    age_slice[np.isnan(age_slice)]=rand_age
    X_titanic_test["Age"]=age_slice
    X_titanic_test["Age"]=X_titanic_test["Age"].astype(int)
    
    means=X_titanic_test['Fare'].mean()
    stds=X_titanic_test['Fare'].std()
    is_nulls=X_titanic_test['Fare'].isnull().sum()
    rand_ages=np.random.randint(means-stds,means+stds,size=is_nulls)
    age_slices=X_titanic_test['Fare'].copy()
    age_slices[np.isnan(age_slices)]=rand_ages
    X_titanic_test['Fare']=age_slices
    X_titanic_test['Fare']=X_titanic_test['Fare'].astype(int)    
    
for dt in X_titanic_test:
    X_titanic_test['re']=X_titanic_test['SibSp']+X_titanic_test['Parch']
    X_titanic_test.loc[X_titanic_test['re']>0,'voySeul']='No'
    X_titanic_test.loc[X_titanic_test['re']==0,'voySeul']='Yes'
dummies=[]
dum=['Pclass','Sex','Embarked','voySeul']
for col in dum:
    dummies.append(pd.get_dummies(X_titanic_test[col]))
titanic_dummies=pd.concat(dummies,axis=1)
X_titanic_test=pd.concat((X_titanic_test,titanic_dummies),axis=1)
X_titanic_test=X_titanic_test.drop(dum,axis=1)

X_titanic_test=X_titanic_test.iloc[:,:]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_titanic_train = sc.fit_transform(X_titanic_train) 
X_titanic_test = sc.transform(X_titanic_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_titanic_train,y_titanic_train)

# Predicting the Test set results
y_pred = list(classifier.predict(X_titanic_test))
iddrop=["Pclass","Name","Sex","SibSp","Parch","Ticket","Cabin","Fare","Embarked","Age"]
idpassenger=titanic_test.drop(iddrop,axis=1)
df=pd.DataFrame(y_pred,columns=["Survived"])
idpassenger=pd.concat((idpassenger,df),axis=1)
idpassenger.to_csv("titan.csv",index=False)

