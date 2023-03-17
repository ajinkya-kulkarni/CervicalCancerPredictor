
import os
import cv2
import numpy as np
from tqdm.notebook import tqdm

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

##########################################################################

# Define a function to preprocess the images

def preprocess_images(folder_paths, img_size):
	'''
	Preprocesses a list of images.
	Parameters:
		- folder_paths (list of str): List of folder paths containing the images.
		- img_size (tuple of int): Tuple of target image size, e.g., (224, 224).
		- augment (bool): Flag indicating whether to perform data augmentation.
	Returns:
		- original_images (numpy array): Array of preprocessed images.
		- original_labels (numpy array): Array of labels for each image.
	'''
	# Initialize empty lists for images and labels
	original_images = []
	original_labels = []

	# Loop over each folder path and each image in the folder
	pbar = tqdm(total=len(folder_paths) * len(os.listdir(folder_paths[0])), desc='Loading images')

	for folder_path in folder_paths:
		for filename in os.listdir(folder_path):
			# Load image
			img = cv2.imread(os.path.join(folder_path, filename))

			# Resize image
			img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)

			# Convert to RGB
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			# Normalize pixel values
			img = img / 255.0

			# Convert image to numpy array
			img = np.array(img, dtype=np.float32)

			# Append preprocessed image and label to lists
			original_images.append(img)
			original_labels.append(folder_path.split('/')[-1])

			pbar.update(1)

	# Convert lists to numpy arrays
	original_images = np.array(original_images)
	original_labels = np.array(original_labels)

	pbar.close()

	return original_images, original_labels

##########################################################################

def new_model(X_train, regularization):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	# FC layer
	model.add(tf.keras.layers.Flatten())
	if regularization:
		model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2()))
	else:
		model.add(tf.keras.layers.Dense(32, activation='relu'))
	model.add(tf.keras.layers.Dropout(rate = 0.3))
	if regularization:
		model.add(tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2()))
	else:
		model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dropout(rate = 0.3))
	model.add(tf.keras.layers.Dense(3, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

	return model

##########################################################################