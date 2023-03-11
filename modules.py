
import os
import cv2
import numpy as np
from tqdm.notebook import tqdm

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

##########################################################################

# Define a function to preprocess the images

def preprocess_images(folder_paths, img_size, batch_size, augment=False):
	'''
	Preprocesses a list of images.

	Parameters:
		- folder_paths (list of str): List of folder paths containing the images.
		- img_size (tuple of int): Tuple of target image size, e.g., (224, 224).
		- batch_size (int): Batch size for data augmentation.
		- augment (bool): Flag indicating whether to perform data augmentation.

	Returns:
		- preprocessed_images (numpy array): Array of preprocessed images.
		- labels (numpy array): Array of labels for each image.
	'''
	# Initialize empty lists for images and labels
	images = []
	labels = []

	# Loop over each folder path and each image in the folder
	pbar = tqdm(total=len(folder_paths) * len(os.listdir(folder_paths[0])), desc='Loading images:')

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

			if augment and len(images) % batch_size == 0:
				# Convert the list of images to a numpy array
				X = np.array(images)

				# Apply data augmentation to the batch of images
				datagen = ImageDataGenerator(
					rotation_range=20,  # randomly rotate the image by up to 20 degrees
					zoom_range=0.2,  # randomly zoom in or out on the image by up to 20%
					width_shift_range=0.1,  # randomly shift the image horizontally by up to 10% of the image width
					height_shift_range=0.1,  # randomly shift the image vertically by up to 10% of the image height
					shear_range=0.2,  # randomly apply shearing transformation to the image
					horizontal_flip=True,  # randomly flip the image horizontally
					fill_mode='reflect'  # fill any pixels that are outside the boundaries of the image with reflected values
				)

				datagen.fit(X)
				for batch in datagen.flow(X, batch_size=batch_size):
					for img in batch:
						images.append(img)
						labels.append(folder_path.split('/')[-1])

					if len(images) % batch_size == 0:
						break

					# Break the loop if we have generated enough augmented images
					if len(images) >= len(os.listdir(folder_path)):
						break

		if not augment:
			pbar.set_description(f'Folder: {folder_path.split("/")[-1]}')

	# Convert lists to numpy arrays
	preprocessed_images = np.array(images)
	labels = np.array(labels)

	pbar.close()

	return preprocessed_images, labels

##########################################################################

# Define the model architecture and compile it

def create_model(input_shape, num_classes, learning_rate, l1_lambda=0.01, l2_lambda=0.01):
	
	# Initialize a sequential model
	model = Sequential()
	
	# Add a 2D convolutional layer with 32 filters, a 3x3 kernel size, and ReLU activation function
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)))
	
	# Add a max pooling layer with a 2x2 pool size
	model.add(MaxPooling2D((2, 2)))
	
	# Add another 2D convolutional layer with 64 filters, a 3x3 kernel size, and ReLU activation function
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)))
	
	# Add another max pooling layer with a 2x2 pool size
	model.add(MaxPooling2D((2, 2)))
	
	# Add a third 2D convolutional layer with 128 filters, a 3x3 kernel size, and ReLU activation function
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)))
	
	# Add another max pooling layer with a 2x2 pool size
	model.add(MaxPooling2D((2, 2)))
	
	# Flatten the output of the convolutional layers
	model.add(Flatten())
	
	# Add a dense layer with 256 units and ReLU activation function
	model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)))
	
	# Add a dropout layer with probability
	model.add(Dropout(0.5))
	
	# Add a dense layer with 128 units and ReLU activation function
	model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)))
	
	# Add another dropout layer with probability
	model.add(Dropout(0.5))
	
	# Add a dense layer with num_classes units and softmax activation function
	model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)))
	
	optimizer = Adam(learning_rate)
	
	# Compile the model with categorical cross-entropy loss and Adam optimizer
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	
	return model

##########################################################################

