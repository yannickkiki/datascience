import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from keras.preprocessing import image
from keras.applications.nasnet import preprocess_input

def one_hot_encode(y):
    y = y.reshape(y.shape[0], 1)
    onehotencoder = OneHotEncoder(categories='auto')
    y = onehotencoder.fit_transform(y).toarray()
    return y

def extract_features(directory = None, pt_model = None, batch_size = 32,
                     single_image = False, img_path = None):
    if single_image:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = pt_model.predict(x)
        return features
    else:
        datagen = ImageDataGenerator(rescale=1./255)
        
        generator = datagen.flow_from_directory(directory,
                                                target_size=(224, 224),
                                                batch_size = batch_size,
                                                class_mode='binary')
        
        sample_count = generator.__dict__["n"]
        
        features = np.zeros(shape=(sample_count, 1000))
        labels = np.zeros(shape=(sample_count))
    
        i = 0
        for inputs_batch, labels_batch in generator:
            features_batch = pt_model.predict(inputs_batch)
            features[i * batch_size: (i + 1) * batch_size] = features_batch
            labels[i * batch_size: (i + 1) * batch_size] = labels_batch
            i += 1
            if i * batch_size >= sample_count:
                break
        labels = one_hot_encode(labels)
        return features, labels, generator.__dict__["class_indices"]

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
def measure_metrics(test_categories, predicted_categories):   
    labels = list(set(test_categories+predicted_categories))   
    metrics = dict()
    metrics["asc"] = accuracy_score(test_categories, predicted_categories)
    metrics["cm"] = confusion_matrix(test_categories, predicted_categories,
           labels = labels)
    metrics["f1_score"] = f1_score(test_categories, predicted_categories,
           average = None, labels = labels)
    metrics["labels"] = labels
    return metrics