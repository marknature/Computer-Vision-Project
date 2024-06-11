import os
import numpy as np
from TensorFlow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(data_dir, img_height=224, img_width=224, batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                 height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True, fill_mode='nearest')
    data = datagen.flow_from_directory(data_dir, target_size=(img_height, img_width),
                                       batch_size=batch_size, class_mode='binary')
    return data

if __name__ == "__main__":
    train_data = preprocess_data('data/train')
    val_data = preprocess_data('data/val')
    test_data = preprocess_data('data/test')
