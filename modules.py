
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm

import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

######################################################################

# Define a function to preprocess the images
def preprocess_images(folder_paths, img_size):
	'''
	Preprocesses a list of images.

	Parameters:
		- folder_paths (list of str): List of folder paths containing the images.
		- img_size (tuple of int): Tuple of target image size, e.g., (224, 224).

	Returns:
		- original_images (numpy array): Array of preprocessed images.
		- original_labels (numpy array): Array of labels for each image.

	Raises:
		- FileNotFoundError: If a folder path in `folder_paths` does not exist.
		- cv2.error: If an image file in a folder cannot be read by OpenCV.

	Each image in `folder_paths` is loaded using OpenCV and preprocessed using the following steps:
	- Resized to `img_size`.
	- Converted to RGB format.
	- Normalized pixel values to [0, 1].
	- Converted to a NumPy array.

	The preprocessed images and their corresponding labels (extracted from the folder name) are stored in two separate lists,
	which are then converted to NumPy arrays and returned.
	'''
	# Initialize empty lists for images and labels
	original_images = []
	original_labels = []

	# Loop over each folder path and each image in the folder
	pbar = tqdm(total=sum(len([filename for filename in os.listdir(folder_path) if not filename.startswith('.')]) for folder_path in folder_paths), desc='Loading images')

	for folder_path in folder_paths:
		for filename in os.listdir(folder_path):
			if filename.startswith('.'):
				continue

			# Do not allow files other than certain extensions
			if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
				continue

			# Load image
			img = cv2.imread(os.path.join(folder_path, filename))

			# Check that the image is loaded correctly
			if img is None:
				raise Warning('Failed to load image ', filename)
				print()

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
			original_labels.append(os.path.basename(folder_path))

			pbar.update(1)

	# Convert lists to numpy arrays
	original_images = np.array(original_images)
	original_labels = np.array(original_labels)

	pbar.close()

	return original_images, original_labels

######################################################################

def augment_images(X_train, y_train, n_augmentations):
	'''
	Augments a dataset of images using data augmentation.
	
	Parameters:
		- X_train (numpy array): Array of original images.
		- y_train (numpy array): Array of labels for each image.
		- n_augmentations (int): Number of augmentations to generate per original image.
	
	Returns:
		- X_train_final (numpy array): Array of augmented images.
		- y_train_final (numpy array): Array of labels for each augmented image.
	
	This function uses the `ImageDataGenerator` class from TensorFlow's Keras module to generate `n_augmentations` variations of
	each image in `X_train`. The following augmentations are applied:
	- Random rotation up to 20 degrees.
	- Random horizontal and vertical shifts up to 10% of image dimensions.
	- Random zoom up to 10%.
	- Random horizontal flip.
	- Nearest-neighbor fill mode.
	
	The augmented images and their corresponding labels are stored in two separate lists, which are then concatenated with the
	original images and labels to create the final training set. The resulting augmented images and labels are returned as NumPy arrays.
	'''
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

######################################################################

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

######################################################################

def plot_confusion_matrix(num_classes, ConfusionMatrix):
	# Define class dictionary
	# Maps numeric labels to human-readable class names
	class_dict = {0: 'group_1', 1: 'group_2', 2: 'group_3'}

	# Plot confusion matrix
	# Creates a new figure and axis, and draws a heatmap of the confusion matrix
	fig, ax = plt.subplots()
	im = ax.imshow(ConfusionMatrix, cmap = 'Blues', interpolation = 'None')

	# Set axis tick labels and titles
	ax.set_xticks(np.arange(num_classes))
	ax.set_yticks(np.arange(num_classes))
	ax.set_xticklabels(class_dict.keys())
	ax.set_yticklabels(class_dict.keys())
	ax.set_xlabel('Predicted Class')
	ax.set_ylabel('True Class')
	ax.set_title('Confusion Matrix')

	# Add text annotations to each cell of the heatmap
	# The text shows the numerical value of the confusion matrix cell
	for i in range(num_classes):
		for j in range(num_classes):
			text = ax.text(j, i, format(ConfusionMatrix[i, j], '.2f'), ha="center", va="center", color="white" if ConfusionMatrix[i, j] > 0.5 else "black")

	# Add a colorbar to the plot
	plt.colorbar(im)

	# Save and close the plot
	# Saves the plot to a PDF file and closes the figure
	plt.savefig('ConfusionMatrix.pdf', bbox_inches = 'tight')
	plt.close()

######################################################################

def calculate_metrics(model, X_test, y_test, y_pred, y_pred_labels, y_true_labels):

	# Calculate performance metrics
	accuracy = accuracy_score(y_true_labels, y_pred_labels)
	precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
	recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
	f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
	cm = confusion_matrix(y_true_labels, y_pred_labels)

	# Print performance metrics
	print(f'Test accuracy: {accuracy:.3f}')
	print(f'Test precision: {precision:.3f}')
	print(f'Test recall: {recall:.3f}')
	print(f'Test F1-score: {f1:.3f}')
	print()
	print(f'Test confusion matrix:\n{cm}')
	print()

######################################################################

def CNN_model(X_train, learning_rate, regularization):
	'''
	Creates a CNN model with optional regularization.
	
	Parameters:
		- X_train (numpy array): Array of training images used to determine the input shape of the model.
		- learning_rate (float): Learning rate used by the Adam optimizer.
		- regularization (bool): Flag indicating whether to add L1/L2 regularization to the Dense layers.
	
	Returns:
		- model (tf.keras.Model): A new instance of the CNN model.
	
	This function creates a CNN model with the following layers:
	- Convolutional layer with 8 filters, 3x3 kernel, ReLU activation, and Batch Normalization.
	- Max Pooling layer with 2x2 pool size.
	- Convolutional layer with 16 filters, 3x3 kernel, ReLU activation, and Batch Normalization.
	- Max Pooling layer with 2x2 pool size.
	- Convolutional layer with 32 filters, 3x3 kernel, ReLU activation, and Batch Normalization.
	- Max Pooling layer with 2x2 pool size.
	- Flatten layer to convert 3D feature maps into a 1D feature vector.
	- Dense layer with 32 neurons and ReLU activation, with or without L1/L2 regularization.
	- Dropout layer with a rate of 0.3 to prevent overfitting.
	- Dense layer with 16 neurons and ReLU activation, with or without L1/L2 regularization.
	- Dropout layer with a rate of 0.3 to prevent overfitting.
	- Output layer with 3 neurons and a softmax activation for 3-class classification.
	
	If `regularization` is True, L1/L2 regularization is added to the Dense layers to prevent overfitting. The Adam optimizer with
	the specified `learning_rate` is used to optimize the model, with categorical crossentropy loss and accuracy metric.
	The model is returned as a TensorFlow Keras Model object.
	'''
	# Initialize a Sequential model
	model = tf.keras.models.Sequential()

	# Add a Convolutional layer with 8 filters, a 3x3 kernel, and ReLU activation function
	model.add(tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
	# Add Batch Normalization to normalize the output of the previous layer
	model.add(tf.keras.layers.BatchNormalization())
	# Add Max Pooling with a pool size of 2x2 to reduce spatial dimensions
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	# Add another Convolutional layer with 16 filters, a 3x3 kernel, and ReLU activation function
	model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
	# Add Batch Normalization to normalize the output of the previous layer
	model.add(tf.keras.layers.BatchNormalization())
	# Add Max Pooling with a pool size of 2x2 to reduce spatial dimensions
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	# Add another Convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation function
	model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
	# Add Batch Normalization to normalize the output of the previous layer
	model.add(tf.keras.layers.BatchNormalization())
	# Add Max Pooling with a pool size of 2x2 to reduce spatial dimensions
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	# Add a Flatten layer to convert the 3D feature maps into a 1D feature vector
	model.add(tf.keras.layers.Flatten())

	# Add a Fully Connected layer with 32 neurons and ReLU activation function, with or without regularization
	if regularization:
		model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2()))
	else:
		model.add(tf.keras.layers.Dense(32, activation='relu'))

	# Add Dropout with a rate of 0.3 to prevent overfitting
	model.add(tf.keras.layers.Dropout(rate=0.3))

	# Add another Fully Connected layer with 16 neurons and ReLU activation function, with or without regularization
	if regularization:
		model.add(tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2()))
	else:
		model.add(tf.keras.layers.Dense(16, activation='relu'))

	# Add Dropout with a rate of 0.3 to prevent overfitting
	model.add(tf.keras.layers.Dropout(rate=0.3))

	# Add the output layer with 3 neurons (for 3 classes) and a softmax activation function
	model.add(tf.keras.layers.Dense(3, activation='softmax'))

	# Create the Adam optimizer with the specified learning rate
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	# Compile the model with the optimizer, categorical crossentropy loss, and accuracy metric
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model

######################################################################