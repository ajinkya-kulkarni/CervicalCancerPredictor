[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cervicalcancerpredictor.streamlit.app/)

## Image Classification using CNNs

This project demonstrates the use of Convolutional Neural Networks (CNNs) to classify images into different categories.

### App Overview

A web application developed using Streamlit is available at https://cervicalcancerpredictor.streamlit.app/

### Requirements

- Python 3.x
- TensorFlow
- NumPy
- scikit-learn
- tqdm
- Matplotlib

### Installation

1. Clone the repository to your local machine.
2. Install the required packages using the following command:

	```
	pip install -r requirements.txt
	```

### Usage

1. Place the images to be classified into different folders based on their category.
2. Open `parameters.py` file and update the parameters as per your requirement.
3. Run `workflow.py` file to train the model and generate predictions for the test data.
4. Check the performance of the model by referring to the `ModelPerformance.pdf` file.

### Files

- `workflow.py` - Main script to train the CNN model and generate predictions.
- `parameters.py` - Script to define the parameters for the model.
- `modules.py` - Script containing helper functions for preprocessing, augmenting and plotting images and contains the CNN model.
- `ModelPerformance.pdf` - PDF file containing the plots of model accuracy and loss over epochs.
- `confusion_matrix.pdf` - PDF file containing the confusion matrix for the model.
- `Trained_CNN_Model.h5` - Trained CNN model saved in H5 format.

#### References

- [Convolutional Neural Networks (CNNs) for Image Classification](https://towardsdatascience.com/convolutional-neural-networks-cnns-for-image-classification-3d7123b9f8ff)
- [Convolutional Neural Networks (CNNs) for CIFAR-10 Dataset Classification](https://towardsdatascience.com/convolutional-neural-networks-cnns-for-cifar-10-image-classification-a5e5a5b3ce08)