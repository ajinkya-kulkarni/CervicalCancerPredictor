
import os
import cv2
import numpy as np
from tqdm.notebook import tqdm

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

##########################################################################

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_augmented_images(image_paths, img_size, n_augmentations):
	'''
	Generates augmented images for a list of image paths.

	Parameters:
		- image_paths (list of str): List of paths to input images.
		- img_size (tuple of int): Tuple of target image size, e.g., (224, 224).
		- n_augmentations (int): Number of augmented images to generate for each input image.

	Returns:
		- augmented_images (numpy array): Array of augmented images.
		- labels (numpy array): Array of labels for each image.
	'''

	# Initialize ImageDataGenerator with desired augmentation parameters
	datagen = ImageDataGenerator(
		rotation_range=30,
		zoom_range=[0.9, 1.1],
		horizontal_flip=True,
		width_shift_range=0.1,
		height_shift_range=0.1,
		fill_mode='nearest'
	)

	# Load and preprocess each image, and generate augmented images
	images = []
	labels = []
	for image_path in image_paths:
		label = os.path.basename(os.path.dirname(image_path))
		img = cv2.imread(image_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
		img = np.array(img, dtype=np.float32)
		img /= 255.0
		images.append(img)
		labels.append(label)

		# Generate n_augmentations random augmented images
		augmented_images = datagen.flow(np.array([img]), batch_size=n_augmentations, shuffle=False)
		for i in range(n_augmentations):
			images.append(augmented_images.next()[0])
			labels.append(label)

	# Convert lists to numpy arrays
	augmented_images = np.array(images)
	labels = np.array(labels)

	return augmented_images, labels

##########################################################################

# Define a function to preprocess the images

def preprocess_images(folder_paths, img_size, n_augmentations = 5):
	'''
	Preprocesses a list of images with data augmentation.

	Parameters:
		- folder_paths (list of str): List of folder paths containing the images.
		- img_size (tuple of int): Tuple of target image size, e.g., (224, 224).
		- n_augmentations (int): Number of augmented images to generate for each input image.

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

			# Convert to RGB
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			# Resize image
			img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)

			# Normalize pixel values
			img = img / 255.0

			# Append preprocessed image and label to lists
			images.append(img)
			labels.append(folder_path.split('/')[-1])

			pbar.update(1)

	pbar.close()

	# Generate augmented images for each input image
	augmented_images, augmented_labels = generate_augmented_images(folder_paths, img_size, n_augmentations)

	# Append augmented images and labels to lists
	images.extend(augmented_images)
	labels.extend(augmented_labels)

	# Convert lists to numpy arrays
	preprocessed_images = np.array(images)
	labels = np.array(labels)

	return preprocessed_images, labels

##########################################################################

# Define the model architecture and compile it

# def create_model(input_shape, num_classes, learning_rate):
	
# 	# Initialize a sequential model
# 	model = Sequential()
	
# 	# Add a 2D convolutional layer with 16 filters, a 3x3 kernel size, and ReLU activation function
# 	model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
	
# 	# Add a max pooling layer with a 2x2 pool size
# 	model.add(MaxPooling2D((2, 2)))

# 	# Add batch normalization
# 	model.add(BatchNormalization())
	
# 	# Add another 2D convolutional layer with 32 filters, a 3x3 kernel size, and ReLU activation function
# 	model.add(Conv2D(32, (3, 3), activation='relu'))
	
# 	# Add another max pooling layer with a 2x2 pool size
# 	model.add(MaxPooling2D((2, 2)))

# 	# Add a third 2D convolutional layer with 64 filters, a 3x3 kernel size, and ReLU activation function
# 	model.add(Conv2D(64, (3, 3), activation='relu'))
	
# 	# Add another max pooling layer with a 2x2 pool size
# 	model.add(MaxPooling2D((2, 2)))
	
# 	# Flatten the output of the convolutional layers
# 	model.add(Flatten())
	
# 	# Add a dense layer with 64 units and ReLU activation function
# 	model.add(Dense(64, activation='relu'))

# 	model.add(Dropout(0.25))
	
# 	# Add a dense layer with num_classes units and softmax activation function
# 	model.add(Dense(num_classes, activation='softmax'))
	
# 	optimizer = Adam(learning_rate)
	
# 	# Compile the model with categorical cross-entropy loss and Adam optimizer
# 	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	
# 	return model

##########################################################################

import tensorflow as tf

def new_model(X_train):

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
	model.add(tf.keras.layers.Dense(32, activation='relu'))
	model.add(tf.keras.layers.Dropout(rate = 0.3))
	model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dropout(rate = 0.3))
	model.add(tf.keras.layers.Dense(3, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

	return model

##########################################################################