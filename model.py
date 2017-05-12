import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
import sklearn
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import math
import random
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS 

#DEFINE FLAGS VARIABLES#
flags.DEFINE_integer('epo', 7, "The number of epochs.")
flags.DEFINE_integer('batch', 32, "batch size")

## Samples of the data.
samples = []
with open("Dataset/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	for sample in reader:
		samples.append(sample)

## Divided Train and validation data using sci-kit split with factor or 0.2
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

## Method to resize the image, but not used in the project as it didn't affect the accuracy and training time much.
def resize(img):
	return cv2.resize(img, (64,64))

'''
Added random brightness on HSV layer of the image so that it could 
learn any track with less brightness also.
'''
def random_brightness(img):
	hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	### Generate new random brightness
	rand = random.uniform(0.3,1.0)
	hsv_img[:,:,2] = rand * hsv_img[:,:,2]
	return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

'''
Generator that will generate the data in parallerl of defined batch size
'''
def generator(samples, batch_size=FLAGS.batch, is_training=True):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset: offset+batch_size]
			images = []
			measurments = []
			for line in batch_samples:
				steering_center = float(line[3])
				## Correction for left and right images so that car doesn't go off the track.
				correction = 0.15
				## center image 
				source_path = line[0]
				filename = source_path.split("/")[-1]
				center_img_filepath = "Dataset/IMG/" + filename
				images.append(cv2.imread(center_img_filepath))
				measurments.append(steering_center)
				## left image
				source_path = line[1]
				filename = source_path.split("/")[-1]
				left_img_filepath = "Dataset/IMG/" + filename
				left_correction = steering_center + correction
				images.append(cv2.imread(left_img_filepath))
				measurments.append(left_correction)
				## Right image
				source_path = line[2]
				filename = source_path.split("/")[-1]
				right_img_filepath = "Dataset/IMG/" + filename
				right_correction = steering_center - correction
				images.append(cv2.imread(right_img_filepath))
				measurments.append(right_correction)
			## Augument Data
			augumented_images, augumented_measurments = [], []

			for image, measurment in zip(images, measurments):
				#if (is_training):
				#image = random_brightness(image)

				#rand_shift_measurment = 1 + np.random.uniform(-0.10,0.10)
				#measurment = measurment * rand_shift_measurment
				augumented_images.append(image)
				augumented_measurments.append(measurment)
				#if (is_training):
				## Flipped the images so that model could learn left and right steering angles both of the image.
				augumented_images.append(cv2.flip(image, 1))
				augumented_measurments.append(measurment * -1.0)


			X_samples = np.array(augumented_images)
			y_samples = np.array(augumented_measurments)
			yield sklearn.utils.shuffle(X_samples, y_samples)

train_generator = generator(train_samples, is_training= True)
validation_generator = generator(validation_samples, is_training= False)

## Model is same as the model mentioned in the NVIDIA paper.
### Added drop out layers extra on FC to avoid overfitting.
model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x:x/255 - 0.5))
model.add(Convolution2D(24, 5,5, subsample=(2,2), activation='relu'))
#model.add(Dropout(0.25))
model.add(Convolution2D(36, 5,5, subsample=(2,2), activation='relu'))
#model.add(Dropout(0.5))
model.add(Convolution2D(48, 5,5, subsample=(2,2), activation='relu'))
#model.add(Dropout(0.5))
model.add(Convolution2D(64, 3,3, subsample=(1,1), activation='relu'))
model.add(Convolution2D(64, 3,3, subsample=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer = adam, metrics=['accuracy'])
model.summary()
model.fit_generator(train_generator, samples_per_epoch= \
	2 * math.ceil(len(train_samples)), validation_data= validation_generator, \
	nb_val_samples=len(validation_samples), nb_epoch=FLAGS.epo)
### Save the model 
model.save('model.h5')



