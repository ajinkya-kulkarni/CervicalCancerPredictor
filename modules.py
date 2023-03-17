
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
		- preprocessed_images (numpy array): Array of preprocessed images.
		- labels (numpy array): Array of labels for each image.
	'''
	# Initialize empty lists for images and labels
	images = []
	labels = []

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
			images.append(img)
			labels.append(folder_path.split('/')[-1])

			pbar.update(1)

	# Convert lists to numpy arrays
	preprocessed_images = np.array(images)
	labels = np.array(labels)

	pbar.close()

	return preprocessed_images, labels

##########################################################################

def augment_images(img_size, preprocessed_images, labels, n_augmentations=5):
	"""
	Perform data augmentation on a set of preprocessed images and their corresponding labels.

	Args:
		img_size: size of the images 
		preprocessed_images: numpy array of shape (num_images, height, width, channels)
			Array of preprocessed images to augment.
		labels: numpy array of shape (num_images,)
			Array of labels corresponding to the preprocessed images.
		n_augmentations: int, optional
			Number of augmented images to generate per original image.
			Default is 5.

	Returns:
		augmented_images: numpy array of shape (num_images * n_augmentations, height, width, channels)
			Array of augmented images.
		augmented_labels: numpy array of shape (num_images * n_augmentations,)
			Array of labels corresponding to the augmented images.
	"""
	# Create an ImageDataGenerator object with the desired augmentations
	datagen = ImageDataGenerator(
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')

	# Create an empty array to store the augmented images and labels
	augmented_images = np.zeros((len(preprocessed_images) * n_augmentations, img_size[0], img_size[1], 3))
	augmented_labels = np.zeros(len(preprocessed_images) * n_augmentations)

	# Perform data augmentation on each original image n times
	pbar = tqdm(total=len(preprocessed_images) * n_augmentations, desc="Augmenting images")
	for i in range(len(preprocessed_images)):
		for j in range(n_augmentations):
			# Generate a batch of one augmented image using the datagen
			img_batch = datagen.flow(preprocessed_images[i:i+1], batch_size=1)[0]
			# Add the augmented image and its label to the arrays
			idx = i * n_augmentations + j
			augmented_images[idx] = img_batch
			augmented_labels[idx] = labels[i]
			pbar.update(1)

	pbar.close()
	return augmented_images, augmented_labels

##########################################################################

def new_model(X_train, regularization):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu', 
									input_shape=X_train.shape[1:], 
									kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) if regularization else None))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', 
									kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) if regularization else None))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
									kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) if regularization else None))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	# FC layer
	model.add(tf.keras.layers.Flatten())
	if regularization:
		model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
	else:
		model.add(tf.keras.layers.Dense(32, activation='relu'))
	model.add(tf.keras.layers.Dropout(rate=0.3))
	if regularization:
		model.add(tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
	else:
		model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dropout(rate=0.3))
	model.add(tf.keras.layers.Dense(3, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

##########################################################################
