import numpy as np

# from tqdm.notebook import tqdm
from tqdm import tqdm

import os
os.system('rm *.pdf')
os.system('clear')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
# %config InlineBackend.figure_format='retina'

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

######################################################################

from modules import *

from parameters import *

######################################################################

# Preprocess images and convert labels to integers

folder_paths = ['./group_1', './group_2', './group_3']

original_images, original_labels = preprocess_images(folder_paths, img_size)

print()

######################################################################

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(original_images, original_labels, test_size=split_percentage, random_state=42)

######################################################################

# Convert string labels to numerical labels

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)

######################################################################

# Convert labels to one-hot encoding

num_classes = len(np.unique(original_labels))

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(label_encoder.transform(y_test), num_classes)

######################################################################

# Perform augmentations on X_train data

X_train_final, y_train_final = augment_images(X_train, y_train, n_augmentations)

print()

######################################################################

# Plot and verify/show that augmentations are indeed correct. Plot some random augmentations. 

show_random_augmentation(X_train, X_train_final, n_augmentations)

######################################################################

# Define the model

model = CNN_model(X_train_final, learning_rate, regularization = RegularizationKey)

######################################################################

early_stop = EarlyStopping(monitor='val_accuracy', patience=10, min_delta=0.1, verbose=1, mode='auto')

with tqdm(total=epochs) as pbar:
	for epoch in range(epochs):
		
		# Train the model for one epoch and track the history of training
		history = model.fit(X_train_final, y_train_final, batch_size=batch_size, epochs=1, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stop])

		# Retrieve the training and validation accuracy and loss for the current epoch
		train_acc = history.history['accuracy'][0]
		val_acc = history.history['val_accuracy'][0]
		train_loss = history.history['loss'][0]
		val_loss = history.history['val_loss'][0]

		# Append the accuracy and loss values to their respective lists
		train_acc_list.append(train_acc)
		val_acc_list.append(val_acc)
		train_loss_list.append(train_loss)
		val_loss_list.append(val_loss)

		# Check if early stopping criteria have been met
		if early_stop.stopped_epoch > 0:
			print(f'Stopping training at epoch {epoch+1} as validation accuracy has not improved by at least 0.1 for 10 consecutive epochs.')
			break

		# Update the progress bar with the current training and validation accuracy and epoch number
		pbar.set_description(f'Epoch: {epoch+1}/{epochs}, Train Acc: {train_acc * 100:.0f}%, Val Acc: {val_acc * 100:.0f}%')

		# Increment the progress bar
		pbar.update(1)

# Print a blank line after the progress bar is complete
print()

######################################################################

# Generate predictions for the test data

y_pred = model.predict(X_test)

# Convert predictions to labels

y_pred_labels = np.argmax(y_pred, axis=1)

y_true_labels = np.argmax(y_test, axis=1)

######################################################################

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Plot training and validation accuracy over epochs
axs[0].plot(train_acc_list, linewidth=2)
axs[0].plot(val_acc_list, linewidth=2)
axs[0].set_title('Model Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].legend(['train', 'validation'], loc='best')

# Plot training and validation loss over epochs
axs[1].plot(train_loss_list, linewidth=2)
axs[1].plot(val_loss_list, linewidth=2)
axs[1].set_title('Model Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend(['train', 'validation'], loc='best')

# Save and close the plot
plt.savefig('ModelPerformance.pdf', bbox_inches = 'tight')
plt.close()

######################################################################

# Generate confusion matrix
ConfusionMatrix = confusion_matrix(y_true_labels, y_pred_labels, normalize='true')

# Plot confusion matrix
plot_confusion_matrix(num_classes, ConfusionMatrix)

######################################################################

# Evaluate the model on the test data
score = model.evaluate(X_test, y_test, verbose=0)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Convert predictions from probabilities to labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Calculate performance metrics
accuracy = accuracy_score(y_true_labels, y_pred_labels)
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

# Print performance metrics
print()
print(f'Test accuracy: {accuracy:.3f}')
print(f'Test precision: {precision:.3f}')
print(f'Test recall: {recall:.3f}')
print(f'Test F1-score: {f1:.3f}')
print()
print(f'Test confusion matrix:\n{ConfusionMatrix}')
print()

######################################################################

# Save the model

model.save('Trained_CNN_Model.h5')

######################################################################