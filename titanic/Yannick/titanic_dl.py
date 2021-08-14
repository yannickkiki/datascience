import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import utils

# read datasets
train_data, test_data = pd.read_csv('train.csv'), pd.read_csv('test.csv')

# select relevant features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked']

# X_train, X_test, y_train
X_train, X_test = train_data[features], test_data[features]
y_train = train_data.Survived

# fill not availables values if we have
X_train, X_test = X_train.fillna(X_train.mean()), X_test.fillna(X_test.mean())

# one hot encoding
X_train, X_test = pd.get_dummies(X_train), pd.get_dummies(X_test)
# align columns to fix eventual columns mix
X_train, X_test = X_train.align(X_test, join='inner', axis=1)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=5, kernel_initializer='uniform',
                     activation='relu', input_dim=10))

# Adding the second hidden layer
classifier.add(Dense(units=5, kernel_initializer='uniform',
                     activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform',
                     activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=32, epochs=100)

predicted_y = [int(round(value[0])) for value in classifier.predict(X_test)]
# predicted_y = (predicted_y > 0.5)

utils.create_submit_file("submission1.csv", 892, predicted_y)
