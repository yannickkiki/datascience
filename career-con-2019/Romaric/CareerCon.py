# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 07:23:58 2019

@author: roma
"""

# Importing the libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

# Importing the dataset
train_X1 = pd.read_csv('X_train.csv')
test_X1 = pd.read_csv('X_test.csv' )
cols=["row_id","series_id","measurement_number"]

train_X=train_X1.drop(cols,axis=1).iloc[:,3:].values.reshape(-1,128,10)

test_X=test_X1.drop(cols,axis=1).values.reshape(-1,128,10)

df_train_y =pd.read_csv('y_train.csv')

train_y=df_train_y['surface']

surface_names = df_train_y['surface'].unique()
num_surfaces = len(surface_names)
surface_to_numeric = dict(zip(surface_names, range(num_surfaces)))

#titanic train part
cols=["row_id","measurement_number"]
train_y = df_train_y['surface'].replace(surface_to_numeric).values
train_X=train_X1.drop(cols,axis=1)

test_X=test_X1.drop(cols,axis=1)

def FE(data):
    df = pd.DataFrame()
    #craetion of another variables(total angular velocity) using relationship
    data['total_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 +
                             data['angular_velocity_Z']**2)** 0.5
    data['total_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 +
                             data['linear_acceleration_Z']**2)**0.5
    data['total_xyzW'] = (abs(data['orientation_X']) + abs(data['orientation_Y']) +
                              abs(data['orientation_Z'])+abs(data['orientation_W']))/4
   
    data['acc_vs_vel'] = data['total_linr_acc'] / data['total_anglr_vel']
    data['acc_vs_vel'] = data['total_linr_acc'] * data['total_anglr_vel']
    data['quaternion1'] = 2*np.arccos(data['orientation_W'])
    
    #using python basics function and some gymnastics(obtained on internet)
    for col in data.columns:
        if col in ['row_id','series_id','measurement_number', 'orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W']:
            continue
        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()
        df[col + '_median'] = data.groupby(['series_id'])[col].median()
        df[col + '_max'] = data.groupby(['series_id'])[col].max()
        df[col + '_min'] = data.groupby(['series_id'])[col].min()
        df[col + '_std'] = data.groupby(['series_id'])[col].std()
        df[col + '_range'] = df[col + '_max'] - df[col + '_min']
        df[col + '_maxtoMin'] = df[col + '_max']/ df[col + '_min']
        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2
    return df
new_train = FE(train_X)

test = FE(test_X)
""" is an cross validation technique (randomly dividing datasets into k groups)"""

def k_folds(clf, X, t, X_test, k):
    folds = StratifiedKFold(n_splits = k, shuffle=True, random_state=13)
    y_test = np.zeros((X_test.shape[0], 9))
    y_oof = np.zeros((X.shape[0]))
    score = 0
    for i, (train_idx, val_idx) in  enumerate(folds.split(X, t)):
        clf.fit(X[train_idx], t[train_idx])
        y_oof[val_idx] = clf.predict(X[val_idx])
        y_test += clf.predict_proba(X_test) / folds.n_splits
        score += clf.score(X[val_idx], t[val_idx])
    return y_oof, y_test


label1 = pd.read_csv("y_train.csv")
surface_names = label1['surface'].unique()
num_surfaces = len(surface_names)
surface_to_numeric = dict(zip(surface_names, range(num_surfaces)))

label = label1['surface'].replace(surface_to_numeric).values
ntv=new_train.values
tst=test.values

#Feature Scaling
le = LabelEncoder()

label = le.fit_transform(label) 
label = le.transform(label)

rand = RandomForestClassifier(n_estimators=1000, random_state=13)
y_oof, y_test_rand = k_folds(rand,ntv, label, tst, k=10)

ext = ExtraTreesClassifier(n_estimators=1000, random_state=13)
y_oof, y_test_ext = k_folds(ext, ntv, label, tst, k=10)


y_test = (y_test_ext + y_test_rand)/2#mean of the test giving by the two algo
y_test = np.argmax(y_test, axis=1)
career = pd.read_csv(os.path.join("sample_submission.csv"))
career['surface'] = le.inverse_transform(y_test)
career['surface'].replace({0:"fine_concrete",1:"concrete",
                2:"soft_tiles",3:"tiled",4:"soft_pvc",
                5:"hard_tiles_large_space",6:"carpet",7:"hard_tiles",8:"wood"},inplace=True)
career.to_csv('careercon.csv', index=False)