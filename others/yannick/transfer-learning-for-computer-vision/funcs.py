import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def extract_features(directory, sample_count, pt_model, batch_size):
    datagen = ImageDataGenerator(rescale=1./255)
    # Must be equal to the output of the convolutional base
    features = np.zeros(shape=(sample_count, 1000))
    labels = np.zeros(shape=(sample_count))
    
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
                                            target_size=(224, 224),
                                            batch_size = batch_size,
                                            class_mode='binary')
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = pt_model.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels, generator.__dict__["class_indices"]