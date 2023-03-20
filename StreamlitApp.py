import cv2
import numpy as np
import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

######################################################################

# Define a dictionary that maps the class indices to their names

class_names = {0: 'group_1', 1: 'group_2', 2: 'group_3'}

######################################################################

# Define a function to predict the image class probabilities
def predict_image(model_name, image):

	# Load the model

	model = tf.keras.models.load_model(model_name)

	# Get the input shape of the model

	input_shape = model.layers[0].input_shape[1:3]

	# Preprocess the image using cv2

	img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, input_shape)

	img = np.expand_dims(img, axis=0) / 255.0

	# Make predictions using the model

	prediction = model.predict(img)

	# Return the predicted class probabilities

	return prediction[0]

######################################################################

# Create the Streamlit app

st.set_page_config(page_title="Image Classifier", layout="centered", page_icon = ":microscope:")

st.title("Image Classifier")

st.write("This app predicts the class of an uploaded image.")

st.markdown("")

######################################################################

# Create a file uploader and display the uploaded image

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif", "tiff"])

st.markdown("")

if uploaded_file is not None:

	# Read the uploaded image using OpenCV

	image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

	###################################################################

	# Load and preprocess the image using the model

	image_probs = predict_image('Trained_CNN_Model.h5', image)

	###################################################################

	# Get the predicted class label and name

	class_label = np.argmax(image_probs)
	class_name = class_names[class_label]

	###################################################################
	
	st.markdown("")

	# Display the uploaded image and predicted class label

	st.image(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB), caption="Uploaded image", use_column_width=True)

	st.write(f"Result: Uploaded image belongs to the {class_name} class.")

	st.stop()

else:
	
	st.warning("Please upload an image.")

	st.stop()

######################################################################