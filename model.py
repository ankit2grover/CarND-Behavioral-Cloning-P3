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
from helpers.data import read_all_csvs_folders, get_input_shape

flags = tf.app.flags
FLAGS = flags.FLAGS 

#DEFINE FLAGS VARIABLES#
flags.DEFINE_integer('epo', 7, "The number of epochs.")
flags.DEFINE_integer('batch', 32, "batch size")

## Samples of the data.
samples = read_all_csvs_folders('Dataset')
#image_input_shape = get_input_shape(samples)

print ("Total number of samples {}".format(len(samples)))
#print ("Image input shape {}".format(image_input_shape))

## Divided Train and validation data using sci-kit split with factor or 0.1
train_samples, validation_samples = train_test_split(samples, test_size=0.1)


# Function to resize images within the model
# as the NVidia model was designed with 66x200x3 images in mind
def resize(img):
    return ktf.image.resize_images(img, (66, 200))

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
def generator(samples, batch_size=FLAGS.batch):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples.iloc[offset: offset+batch_size]
			center_images = []
			center_measurments = []
			left_images = []
			left_measurments = []
			right_images = []
			right_measurments = []
			for i in range(len(batch_samples)):
				line = batch_samples.iloc[i]
				steering_center = float(line['steering'])
				## Correction for left and right images so that car doesn't go off the track.
				correction = 0.25

				## center image 
				source_path = line['center']
				img = cv2.imread(source_path)
				#if (abs(steering_center) >= 0.05):
				center_images.append(img)
				center_measurments.append(steering_center)

				#center_images.append(random_brightness(img))
				#center_measurments.append(steering_center)
				#center_images.append(random_brightness(img))
				#center_measurments.append(steering_center)
			
				## Flipped Center images help in genralizing the model well.
				center_images.append(cv2.flip(img, 1))
				center_measurments.append(steering_center * -1.0)
				#center_images.append(random_brightness(img))
				#center_measurments.append(steering_center)



				## left image
				source_path = line['left']
				left_correction = steering_center + 0.25
				img = cv2.imread(source_path)
				left_images.append(img)
				left_measurments.append(left_correction)
				
				#left_images.append(random_brightness(img))
				#left_measurments.append(left_correction)
				#left_images.append(random_brightness(img))
				#left_measurments.append(left_correction)

				## Right image
				source_path = line['right']
				right_correction = steering_center - 0.30
				img = cv2.imread(source_path)
				right_images.append(img)
				right_measurments.append(right_correction)
				
				#right_images.append(random_brightness(img))
				#right_measurments.append(right_correction)
				#right_images.append(random_brightness(img))
				#right_measurments.append(right_correction)


			concatenate_images = []
			concatenate_measurments = []
			concatenate_images.extend(center_images)
			concatenate_images.extend(left_images)
			concatenate_images.extend(right_images)
			concatenate_measurments.extend(center_measurments)
			concatenate_measurments.extend(left_measurments)
			concatenate_measurments.extend(right_measurments)

			X_samples = np.array(concatenate_images)
			y_samples = np.array(concatenate_measurments)
			yield sklearn.utils.shuffle(X_samples, y_samples)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

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
#model.add(Dropout(0.5))
model.add(Dense(50))
#model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer = adam, metrics=['accuracy'])
model.summary()
model.fit_generator(train_generator, samples_per_epoch= \
	(math.ceil(len(train_samples))/FLAGS.batch), validation_data= validation_generator, \
	nb_val_samples=(math.ceil(len(validation_samples))/FLAGS.batch), nb_epoch=FLAGS.epo)
### Save the model 
model.save('model.h5')





