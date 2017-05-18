import random
import numpy as np
from data import load_image


def adjust_brightness(img_steering, amount):
	img, steering = img_steering
	img_copy = img.copy().astype(np.int16)
	img_copy = img_copy[:,:, 2] = img_copy[:,:, 2] + amount
	img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
	return img_copy, steering

def flip(img_steering):
	img, steering = img_steering
	return np.flipr(img.copy()), -steering

def side_image_steering_correction(img, steering, correction = 0.25, left:bool):
	if (left):
		img, steering + correction
	

