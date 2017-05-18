import os
import argparse

import numpy as np
import pandas as pd
import cv2
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.backend import tf as ktf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from helpers.data_norman import read_all_csvs_in_folder, detect_input_shape, load_image
from helpers.augument_norman import augment

CROP_TOP, CROP_BOTTOM = 60, 26 #70, 24
random.seed()


# This function helps easily creating different models and contains the parts of the model
# that don't change, i.e. preprocessing and compiling
def create_model(input_shape, model_creator):
    model = Sequential()

    model.add(Cropping2D(((CROP_TOP, CROP_BOTTOM), (0, 0)), input_shape=input_shape))
    #model.add(Lambda(resize))
    model.add(Lambda(lambda x: (x / 255) - 0.5))

    model_creator(model)

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )
    return model

# My own model, which I designed for fast training and lower memory impact
def my_fast_model(model):
    model.add(Conv2D(12, (8, 8), strides=(2, 2), padding='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='elu'))
    model.add(Conv2D(32, (3, 3), padding='valid', activation='elu'))
    model.add(Conv2D(32, (3, 3), padding='valid', activation='elu'))
    model.add(Conv2D(32, (3, 3), padding='valid', activation='elu'))
    model.add(Flatten())
    model.add(Dropout(0.6))
    model.add(Dense(256, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(1))

# The NVidia model
def nvidia_model(model):
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='elu'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='elu'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='elu'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.2))

    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

# Function to resize images within the model
# as the NVidia model was designed with 66x200x3 images in mind
def resize(img):
    return ktf.image.resize_images(img, (66, 200))

# Train the model using a generator in order to save memory
def train(model, data, batch_size=32, epochs=5):
    data_train, data_test = train_test_split(data, test_size=0.2)
    steps_per_epoch = len(data_train) // batch_size
    print('Training set size:', len(data_train))
    print('Epochs:', epochs)
    print('Steps per epoch:', steps_per_epoch)
    model.fit_generator(
        generator(data_train, batch_size),
        steps_per_epoch,
        epochs=epochs,
        validation_data=generator(data_test, batch_size),
        validation_steps=(len(data_test) // batch_size)
    )

# Generator, which feeds training and validation data to the model
# the generator also augments the data
def generator(data, batch_size=32):
    size = len(data)
    while True:
        shuffle(data)
        for offset in range(0, size, batch_size):
            batch = data.iloc[offset:offset+batch_size]

            images = []
            steerings = []
            for i in range(len(batch)):
                row = batch.iloc[i]
                steering = row['steering']
                imgs, steers = augment(row, steering, threshold=0.05, droprate=0.8) # when including Udacity's data
                #imgs, steers = augment(row, steering, threshold=0.0, droprate=0.0) # when exclusively using my own data
                for img, steer in zip(imgs, steers):
                    images.append(img) #cv2.resize(img, (200, 66), interpolation=cv2.INTER_CUBIC))
                    steerings.append(steer)
            yield shuffle(np.array(images), np.array(steerings))

def parse_args():
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='recordings',
        help='Path to CSV and image folder.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    #args = parse_args()

    data = read_all_csvs_in_folder('NormanDataset')
    input_shape = detect_input_shape(data)
    print('{} center image frames in total'.format(len(data)))
    print('Input image shape: {}'.format(input_shape))

    print('Steering left: {}'.format(len(data[data['steering'] < 0])))
    print('Steering right: {}'.format(len(data[data['steering'] > 0])))
    print('Steering straight: {}'.format(len(data[data['steering'] == 0])))

    model = create_model(input_shape, my_fast_model)
    model.summary()

    train(model, data, batch_size=32, epochs=3)

    model.save('model.h5')
