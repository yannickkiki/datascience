"""
#just for test the code
cut_n = 40
data_X = data_X.iloc[0:128*cut_n]
data_y = data_y.iloc[0:cut_n]
validation_X = validation_X.iloc[0:128*cut_n]
"""



def feature_X(data_X):
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
    """
    data_X["num_tan_roll"] = data_X.apply(lambda row: 2*(row.oW+row.oX+row.oY+row.oZ),
          axis = 1)
    data_X["den_tan_roll"] = data_X.apply(lambda row: 1-2*(row.oX**2+row.oY**2),
          axis = 1)
    data_X["tan_roll"] = data_X.apply(lambda row: row.num_tan_roll/row.den_tan_roll,
          axis = 1)
    """
    data_X["roll"] = data_X.apply(lambda row: roll(row), axis = 1)
    ###pitch
    """
    data_X["sin_pitch"] = data_X.apply(lambda row: 2*(row.oW*row.oY-row.oZ*row.oX),
          axis = 1)
    """
    data_X["pitch"] = data_X.apply(lambda row: pitch(row), axis = 1)
    ###yaw
    """
    data_X["num_tan_yaw"] = data_X.apply(lambda row : 2*(row.oW*row.oZ+row.oX*row.oY),
          axis = 1)
    data_X["den_tan_yaw"] = data_X.apply(lambda row : 1-2*(row.oY**2+row.oZ**2),
          axis = 1)
    data_X["tan_yaw"] = data_X.apply(lambda row: row.num_tan_yaw/row.den_tan_yaw,
          axis = 1)
    """
    data_X["yaw"] = data_X.apply(lambda row: yaw(row), axis = 1)
    ##angular velocity
    """
    ###xy
    data_X["avXY"] = data_X.apply(lambda row: (row.avX**2+row.avY**2)**0.5, axis = 1)
    ###xz
    data_X["avXZ"] = data_X.apply(lambda row: (row.avX**2+row.avZ**2)**0.5, axis = 1)
    ###yz
    data_X["avYZ"] = data_X.apply(lambda row: (row.avY**2+row.avZ**2)**0.5, axis = 1)
    """
    ###xyz
    data_X["avXYZ"] = data_X.apply(lambda row: norm(row.avX, row.avY, row.avZ), axis = 1)
    ##linear acceleration
    """
    ###xy
    data_X["laXY"] = data_X.apply(lambda row: (row.laX**2+row.laY**2)**0.5, axis = 1)
    ###xz
    data_X["laXZ"] = data_X.apply(lambda row: (row.laX**2+row.laZ**2)**0.5, axis = 1)
    ###yz
    data_X["laYZ"] = data_X.apply(lambda row: (row.laY**2+row.laZ**2)**0.5, axis = 1)
    """
    ###xyz
    data_X["laXYZ"] = data_X.apply(lambda row: norm(row.laX, row.laY, row.laZ), axis = 1)
    
    features = ['roll', 'pitch', 'yaw', 'avX', 'avY',
                'avZ', 'avXYZ', 'laX', 'laY', 'laZ', 'laXYZ']
    return data_X[features]


data_X["avXYZ"] = data_X.apply(lambda row: norm(row.avX, row.avY, row.avZ), axis = 1)
    data_X["laXYZ"] = data_X.apply(lambda row: norm(row.laX, row.laY, row.laZ), axis = 1)
    
npmin, npmax = np.finfo(np.float32).min, np.finfo(np.float32).max
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (npmin, npmax))
sc.fit(X)
X = sc.transform(X)

numpy.finfo(numpy.float64).max

import copy
X_backup = copy.deepcopy(X)
X_backup.head()