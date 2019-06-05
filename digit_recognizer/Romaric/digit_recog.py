
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import SGD


# Load the data

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

xtr = train.drop("label", axis=1).values.astype(float)
xtr/=255.0
ytr = train["label"].values

xte = test.values.astype(float) 
xte/=255.0

# Reshape data

x_training = xtr.reshape(xtr.shape[0],28,28,1)#reshape(-1, 64, 64, 1) #(-1, 28, 28, 1)
x_validation = xte.reshape(-1, 28, 28, 1)


# One-Hot encoding

y_training = to_categorical(ytr, num_classes=10)


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32,(5,5),input_shape=(28, 28, 1),activation='relu'))
classifier.add(Conv2D(32,(5,5),activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Dropout(0.5))

# Adding a second convolutional layer
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(Conv2D(32,(3,3),activation='relu'))

# Adding a second pooling 
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Dropout(0.5))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection

classifier.add(Dense(units=8192, activation='relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(units=2048, activation='relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(10, activation="softmax"))#fonction d'activation classes multiples(softmax)

# Compiling the CNN
sgd=SGD(lr=0.01, decay=1e-6,momentum=0.9,nesterov=True)
classifier.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=["accuracy"])
"""
classifier.compile(optimizer=RMSprop(lr=0.0001, # or adam like optimizer
                                rho=0.9,
                                epsilon=1e-08,
                                decay=0.00001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
"""
#fit_generator step
#classifier.fit(x_training, y_training, batch_size = 512, epochs = 100)
classifier.fit(x_training, y_training, batch_size = 256, epochs = 20)


# Testing
validation_y = classifier.predict(x_validation)
validation_data = {"ImageId" : list(range(1,1+x_validation.shape[0])),
                   "Label" : [list(line).index(max(line)) for line in validation_y]}
validation_df = pd.DataFrame.from_dict(validation_data)
validation_df.set_index("ImageId",inplace=True)
validation_df.to_csv("digit_recognizer.csv")
