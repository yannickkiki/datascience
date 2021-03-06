import pandas as pd
from beans import extract_features

pt_model = pd.read_pickle("nasnetmobile.model")

batch_size = 32

import os
train_sample_count = len(os.listdir('dataset/train/sneakers'))+ len(os.listdir('dataset/train/tshirt'))
train_features, train_labels, ctr = extract_features('dataset/train', train_sample_count,
                                                pt_model, batch_size)  # Agree with our small dataset size
#test_sample_count = len(os.listdir('dataset/test/sneakers'))+ len(os.listdir('dataset/test/tshirt'))
test_sample_count =  len(os.listdir('dataset/val/tshirt'))
test_features, test_labels, cte = extract_features('dataset/val', test_sample_count,
                                              pt_model, batch_size)

# Define model
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=1000))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
              
# Train model
history = model.fit(train_features, train_labels, epochs=25,
                    batch_size=batch_size, validation_data = (test_features, test_labels))

model.evaluate(train_features, train_labels)
model.evaluate(test_features, test_labels)

test_labels_pred = model.predict(test_features)
test_labels_pred_rounded = list()
for label in test_labels_pred: test_labels_pred_rounded.append(round(label[0]))
    
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
metrics = dict()
metrics["cm"] = confusion_matrix(test_labels, test_labels_pred_rounded)
metrics["asc"] = accuracy_score(test_labels, test_labels_pred_rounded)
metrics["f1_score"] = f1_score(test_labels, test_labels_pred_rounded)
