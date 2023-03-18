# Example usage of the predict_image function

import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

######################################################################

def predict_image(model_name: str, image_path: str):
	"""
	Predicts the class probabilities for an image using a trained Keras model.

	Args:
		model_name (str): The filename of the trained Keras model.
		image_path (str): The filename of the image to predict.

	Returns:
		A 1D numpy array containing the predicted class probabilities.
	"""
	# Load the model
	model = tf.keras.models.load_model(model_name)

	# Get the input shape of the model
	input_shape = model.layers[0].input_shape[1:3]

	# Load and preprocess the image using cv2
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, input_shape)
	img = np.expand_dims(img, axis=0) / 255.0

	# Make predictions using the model
	prediction = model.predict(img)

	# Return the predicted class probabilities
	return prediction[0]

######################################################################

# Define a dictionary that maps the class indices to their names
class_names = {0: 'group_1', 1: 'group_2', 2: 'group_3'}

######################################################################

# Load and preprocess the image
image_probs = predict_image('Trained_CNN_Model.h5', 'prediction_image.jpg')

######################################################################

# Get the predicted class label
class_label = np.argmax(image_probs)

######################################################################

# Get the corresponding class name from the dictionary
class_name = class_names[class_label]

######################################################################

# Print the predicted class label
print(f"Uploaded image belongs to the {class_name} class.")
print()

######################################################################
