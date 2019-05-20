import pandas as pd

#read data
df = pd.read_csv("train.csv")
X = df.drop("label", axis=1).values.astype(float)
X /= 255.0
X = X.reshape(X.shape[0],28,28,1)
y = df["label"].values
y = y.reshape(y.shape[0], 1)

#we dont need df anymore so free the memory
del df

#one hot encoding of y
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categories='auto')
y = onehotencoder.fit_transform(y).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
del X, y

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Initialising the CNN
model = Sequential()

# Step 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))

# Step 2 - Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the CNN
model.fit(X_train, y_train, batch_size = 256, epochs = 20)

#Evaluate the model
model.evaluate(X_train, y_train, verbose=False)
model.evaluate(X_test, y_test, verbose=False)

"""
#Save the model if good
pd.to_pickle(model, "model0.985.sv")

#Load a model
model = pd.read_pickle("model0.983.sv")
"""

#Make prediction on validation data
validation_X = pd.read_csv("test.csv")
validation_X = validation_X.values.astype(float)
validation_X /= 255
validation_X = validation_X.reshape(validation_X.shape[0],28,28,1)
validation_y = model.predict(validation_X)
validation_data = {"ImageId" : list(range(1,1+validation_X.shape[0])),
                   "Label" : [list(line).index(max(line)) for line in validation_y]}
validation_df = pd.DataFrame.from_dict(validation_data)
validation_df.set_index("ImageId",inplace=True)
validation_df.to_csv("submission0.983.csv")
