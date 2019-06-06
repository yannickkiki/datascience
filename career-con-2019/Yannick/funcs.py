import numpy as np

def num_tan_roll(row):
    return 2*(row.oW+row.oX+row.oY+row.oZ)

def den_tan_roll(row):
    return 1-2*(row.oX**2+row.oY**2)

def tan_roll(row):
    return num_tan_roll(row)/den_tan_roll(row)

def roll(row):
    return np.arctan2(num_tan_roll(row),den_tan_roll(row))

def sin_pitch(row):
    return 2*(row.oW*row.oY-row.oZ*row.oX)

def pitch(row):
    return np.arcsin(sin_pitch(row))

def num_tan_yaw(row):
    return 2*(row.oW*row.oZ+row.oX*row.oY)

def den_tan_yaw(row):
    return 1-2*(row.oY**2+row.oZ**2)

def tan_yaw(row):
    return num_tan_yaw(row)/den_tan_yaw(row)

def yaw(row):
    return np.arctan2(num_tan_yaw(row),den_tan_yaw(row))

def norm(x,y,z):
    return (x**2+y**2+z**2)**0.5

def feature_X(data_X):
    data_X.rename(columns={'orientation_X':'oX','orientation_Y':'oY',
                         'orientation_Z':'oZ', 'orientation_W':'oW',
                         'angular_velocity_X':'avX', 'angular_velocity_Y':'avY',
                         'angular_velocity_Z':'avZ', 'linear_acceleration_X':'laX',
                         'linear_acceleration_Y':'laY',
                         'linear_acceleration_Z':'laZ'},inplace=True)
    data_X["roll"] = data_X.apply(lambda row: roll(row), axis = 1)
    data_X["pitch"] = data_X.apply(lambda row: pitch(row), axis = 1)
    data_X["yaw"] = data_X.apply(lambda row: yaw(row), axis = 1)
    
    
    features = ['oX','oY','oZ','oW', 'roll', 'pitch', 'yaw', 'avX', 'avY',
                'avZ', 'laX', 'laY', 'laZ']
    
    return data_X[features]