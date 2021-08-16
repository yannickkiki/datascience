import pandas as pd
from beans import extract_features, plot_history, measure_metrics
import numpy as np

pt_model = pd.read_pickle("nasnetmobile.model")

batch_size = 32
    
train_features, train_labels, ctr = extract_features('dataset/train',
                                                pt_model, batch_size)  # Agree with our small dataset size
test_features, test_labels, cte = extract_features('dataset/test',
                                              pt_model, batch_size)

correspondances = dict([(value, key) for key, value in ctr.items()])

# Define model
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=1000))
model.add(Dropout(0.5))
model.add(Dense(len(ctr), activation='sigmoid'))
model.summary()

# Compile model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
              
# Train model
history = model.fit(train_features, train_labels, epochs=60,
                    batch_size=batch_size,
                    validation_data = (test_features, test_labels))

plot_history(history)

model.evaluate(train_features, train_labels)
model.evaluate(test_features, test_labels)

test_labels_pred = model.predict(test_features)

get_categories = lambda one_hot_encoded_values : [correspondances[list(line).index(max(line))] for line in one_hot_encoded_values]

test_categories = get_categories(test_labels)
predicted_categories = get_categories(test_labels_pred)

threshold = 0.5
row_idxs, col_idxs = np.where(test_labels_pred > threshold)
test_size = len(test_labels)
category_identified = [False]*test_size
for idx in row_idxs: category_identified[idx] = True
predicted_categories = [predicted_categories[idx] if category_identified[idx] else "Dont_know" for idx in range(test_size)]

filtered_test_categories, filtered_predicted_categories = list(), list()
for idx in range(test_size):
    if category_identified[idx]:
        filtered_test_categories.append(test_categories[idx])
        filtered_predicted_categories.append(predicted_categories[idx])

metrics = measure_metrics(test_categories,predicted_categories)
metrics_t = measure_metrics(filtered_test_categories,
                            filtered_predicted_categories)

vars_to_store = {"model": model, "correspondances": correspondances}
pd.to_pickle(vars_to_store,"vars.dict")

