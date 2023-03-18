
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm

import tensorflow as tf

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

def augment_images(X_train, y_train, n_augmentations):
	# Define the number of augmentations per image
	n = n_augmentations

	# Create an ImageDataGenerator instance with the desired augmentations
	datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')

	# Initialize empty lists for the augmented images and labels
	X_train_augmented = []
	y_train_augmented = []

	# Iterate through the original images and labels to generate augmentations
	for i, (x, y) in tqdm(enumerate(zip(X_train, y_train)), total=len(X_train), desc='Augmenting images'):
		# Reshape the image to add a batch dimension
		x = x.reshape((1,) + x.shape)
		
		# Generate n augmentations for the image
		for _ in range(n):
			# Get the next augmented image
			augmented_image = datagen.flow(x, batch_size=1)[0]

			# Append the augmented image and label to the lists
			X_train_augmented.append(augmented_image[0])
			y_train_augmented.append(y)

	# Concatenate the original images and labels with the augmented ones
	X_train_final = np.concatenate((X_train, np.array(X_train_augmented)))
	y_train_final = np.concatenate((y_train, np.array(y_train_augmented)))

	return X_train_final, y_train_final

##########################################################################

def show_random_augmentation(X_train, X_train_final, n_augmentations):

	# Define the number of augmentations per image
	n = n_augmentations

	# Choose a random index from the original images in X_train_final
	random_index = random.randint(0, X_train.shape[0] - 1)

	# Calculate the index of the first augmentation for the selected image in X_train_final
	first_augmentation_index = X_train.shape[0] + random_index * n

	# Calculate the number of rows and columns for the subplots
	cols = int(np.ceil(np.sqrt(n + 1)))
	rows = int(np.ceil((n + 1) / cols))

	# Create a figure and axes for the subplots
	fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))

	# Flatten the axes array for easier indexing
	axes = axes.flatten()

	# Plot the original image in the first subplot
	original_image = X_train_final[random_index]
	axes[0].imshow(original_image)
	axes[0].set_title("Original")
	axes[0].axis('off')

	# Plot n augmentations of the original image
	for i in range(n):
		# Get the augmented image from X_train_final
		augmented_image = X_train_final[first_augmentation_index + i]

		# Plot the augmented image in the next subplot
		axes[i + 1].imshow(augmented_image)
		axes[i + 1].set_title(f"#{i + 1}")
		axes[i + 1].axis('off')

	# Remove any unused subplots
	for j in range(n + 1, rows * cols):
		fig.delaxes(axes[j])

	# Save and close the plot
	plt.savefig('RandomAugmentations.pdf', bbox_inches = 'tight')
	plt.close()

##########################################################################

def new_model(X_train, learning_rate, regularization):

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

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model

##########################################################################