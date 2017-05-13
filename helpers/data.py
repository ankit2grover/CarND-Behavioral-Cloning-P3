import os
import pandas as pd
import numpy as np
import cv2

'''
Read all the data from csv file
'''
def read_data_csv(filename):
	columns = ('center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed')
	df = pd.read_csv(filename, skipinitialspace = True)
	df.columns = columns
	return df

'''
Change the image path as path of images in csv file different from actual images path
'''
def change_image_path(df, folder):
	new_filepath = lambda s: os.path.join(folder, s.split('/')[-1])
	df['center'] = df['center'].map(new_filepath)
	df['left'] = df['left'].map(new_filepath)
	df['right'] = df['right'].map(new_filepath)
	return df

def get_input_shape(df):
	sample_input_image = df['center'].iloc[1]
	return load_image(sample_input_image).shape

# load image from file in HSV color space
def load_image(path):
    img = cv2.imread(path)
    print(path)
    return img

# Exponential Moving Average of steering angles
def smoothen_steering(df, count=5):
    return df.ewm(span=count).mean()['steering']

def read_all_csvs_folders(parent_folder):
	data = []
	folders = os.listdir(parent_folder)
	print(folders)
	for folder in folders:
		if ('.' in folder):
			print ("Found")
			continue
		folder = os.path.join(parent_folder, folder)
		img_folder_path = os.path.join(folder, "IMG")
		csv_filepath = os.path.join(folder , "driving_log.csv")
		print (csv_filepath)
		df = read_data_csv(csv_filepath)
		df = change_image_path(df, img_folder_path)
		data.append(df)

	concatenate_df = pd.DataFrame(np.concatenate(data, axis=0), columns = df.columns)
	#concatenate_df = [concatenate_df['speed'] >= 0.1]
	return concatenate_df