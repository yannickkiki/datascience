import numpy as np
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew

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

def norm(x,y,z,w = 0):
    return (x**2+y**2+z**2+w**2)**0.5

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
    
    data_X['norm_o'] = data_X.apply(lambda row: norm(row.oX,row.oY,row.oZ,row.oW), axis = 1)
    data_X['norm_X'] = data_X['oX']/data_X['norm_o']
    data_X['norm_Y'] = data_X['oY']/data_X['norm_o']
    data_X['norm_Z'] = data_X['oZ']/data_X['norm_o']
    data_X['norm_W'] = data_X['oW']/data_X['norm_o']
    
    data_X['norm_av'] = data_X.apply(lambda row: norm(row.avX,row.avY,row.avZ), axis = 1)
    
    data_X['norm_la'] = data_X.apply(lambda row: norm(row.laX,row.laY,row.laZ), axis = 1)
    
    data_X['av_vs_la'] = data_X['norm_av']/data_X['norm_la']
    
    features = data_X.columns.drop(["row_id", 'measurement_number'])
    
    return data_X[features]

def _kurtosis(x):
    return kurtosis(x)

def CPT5(x):
    den = len(x)*np.exp(np.std(x))
    return sum(np.exp(x))/den

def skewness(x):
    return skew(x)

def SSC(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x,x[1])
    xn = x[1:len(x)-1]
    xn_i2 = x[2:len(x)]    # xn+1 
    xn_i1 = x[0:len(x)-2]  # xn-1
    ans = np.heaviside((xn-xn_i1)*(xn-xn_i2),0)
    return sum(ans[1:]) 

def wave_length(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x,x[1])
    xn = x[1:len(x)-1]
    xn_i2 = x[2:len(x)]    # xn+1 
    return sum(abs(xn_i2-xn))
    
def norm_entropy(x):
    tresh = 3
    return sum(np.power(abs(x),tresh))

def SRAV(x):    
    SRA = sum(np.sqrt(abs(x)))
    return np.power(SRA/len(x),2)

def mean_abs(x):
    return sum(abs(x))/len(x)

def zero_crossing(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x,x[1])
    xn = x[1:len(x)-1]
    xn_i2 = x[2:len(x)]    # xn+1
    return sum(np.heaviside(-xn*xn_i2,0))

def flat_df(df):
    flat_df = DataFrame()
    for col in df.columns:
        if col == "series_id": continue
        flat_df[col + '_mean'] = df.groupby(['series_id'])[col].mean()
        flat_df[col + '_median'] = df.groupby(['series_id'])[col].median()
        flat_df[col + '_max'] = df.groupby(['series_id'])[col].max()
        flat_df[col + '_min'] = df.groupby(['series_id'])[col].min()
        flat_df[col + '_std'] = df.groupby(['series_id'])[col].std()
        flat_df[col + '_range'] = flat_df[col + '_max'] - flat_df[col + '_min']
        flat_df[col + '_maxtoMin'] = flat_df[col + '_max'] / flat_df[col + '_min']
        flat_df[col + '_mean_abs_chg'] = df.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        flat_df[col + '_mean_change_of_abs_change'] = df.groupby('series_id')[col].apply(lambda x: np.mean(np.diff(np.abs(np.diff(x)))))
        flat_df[col + '_abs_max'] = df.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        flat_df[col + '_abs_min'] = df.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        flat_df[col + '_abs_avg'] = (flat_df[col + '_abs_min'] + flat_df[col + '_abs_max'])/2
        flat_df[col + '_skew'] = df.groupby(['series_id'])[col].skew()
        flat_df[col + '_mad'] = df.groupby(['series_id'])[col].mad()
        flat_df[col + '_q25'] = df.groupby(['series_id'])[col].quantile(0.25)
        flat_df[col + '_q75'] = df.groupby(['series_id'])[col].quantile(0.75)
        flat_df[col + '_q95'] = df.groupby(['series_id'])[col].quantile(0.95)
        flat_df[col + '_iqr'] = flat_df[col + '_q75'] - flat_df[col + '_q25']
        flat_df[col + '_CPT5'] = df.groupby(['series_id'])[col].apply(CPT5) 
        flat_df[col + '_SSC'] = df.groupby(['series_id'])[col].apply(SSC) 
        flat_df[col + '_skewness'] = df.groupby(['series_id'])[col].apply(skewness)
        flat_df[col + '_wave_lenght'] = df.groupby(['series_id'])[col].apply(wave_length)
        flat_df[col + '_norm_entropy'] = df.groupby(['series_id'])[col].apply(norm_entropy)
        flat_df[col + '_SRAV'] = df.groupby(['series_id'])[col].apply(SRAV)
        flat_df[col + '_kurtosis'] = df.groupby(['series_id'])[col].apply(_kurtosis) 
        flat_df[col + '_zero_crossing'] = df.groupby(['series_id'])[col].apply(zero_crossing)
    return flat_df

def process_X(data_X):
    data_X = feature_X(data_X)
    print("Featured")
    flat_dX = flat_df(data_X)
    return flat_dX

def read_data():
    data_X = read_csv("datasets/X_train.csv")
    validation_X = read_csv("datasets/X_test.csv")
    data_y = read_csv("datasets/y_train.csv")
    return (data_X, validation_X, data_y)

def plot_feature_class_distribution(classes,tt, features,a=5,b=2):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(a,b,figsize=(16,24))

    for feature in features:
        i += 1
        plt.subplot(a,b,i)
        for clas in classes:
            ttc = tt[tt['surface']==clas]
            sns.kdeplot(ttc[feature], bw=0.5,label=clas)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show()
