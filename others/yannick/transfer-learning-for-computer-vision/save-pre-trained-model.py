from keras.applications.nasnet import NASNetMobile
from keras.preprocessing import image
from keras.applications.nasnet import preprocess_input, decode_predictions
import numpy as np
import pandas as pd

model = NASNetMobile(weights='imagenet')
pd.to_pickle(model, "nasnetmobile.model")
