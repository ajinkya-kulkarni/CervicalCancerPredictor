import numpy as np

# from tqdm.notebook import tqdm
from tqdm import tqdm

import os
os.system('clear')

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
# %config InlineBackend.figure_format='retina'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

######################################################################

from modules import *

######################################################################

# Parameters

batch_size = 8

epochs = 5

learning_rate = 1e-4

img_size = (64, 64)

n_augmentations = 24

split_percentage = 0.1

######################################################################

# Preprocess images and convert labels to integers

folder_paths = ['./group_1', './group_2', './group_3']

original_images, original_labels = preprocess_images(folder_paths, img_size)

print()

######################################################################

# Split data into training and testing sets, and convert labels to one-hot encoding

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

model = new_model(X_train_final, learning_rate, regularization = True)

######################################################################

train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

with tqdm(total=epochs) as pbar:
	for epoch in range(epochs):
		
		history = model.fit(X_train_final, y_train_final, batch_size=batch_size, epochs=1, verbose=0, validation_data=(X_test, y_test))

		train_acc = history.history['accuracy'][0]
		val_acc = history.history['val_accuracy'][0]
		train_loss = history.history['loss'][0]
		val_loss = history.history['val_loss'][0]

		train_acc_list.append(train_acc)
		val_acc_list.append(val_acc)
		train_loss_list.append(train_loss)
		val_loss_list.append(val_loss)

		pbar.set_description(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

		pbar.update(1)

print()

######################################################################

# Generate predictions for the test data

y_pred = model.predict(X_test)

# Convert predictions to labels

y_pred_labels = np.argmax(y_pred, axis=1)

y_true_labels = np.argmax(y_test, axis=1)

print()

######################################################################

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi = 300)

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

# Define class dictionary
class_dict = {0: 'group_1', 1: 'group_2', 2: 'group_3'}

######################################################################

# Generate confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels, normalize='true')

# Plot confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap = 'Blues', interpolation = 'None')

ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))

ax.set_xticklabels(class_dict.keys())
ax.set_yticklabels(class_dict.keys())

ax.set_xlabel('Predicted Class')
ax.set_ylabel('True Class')

ax.set_title('Confusion Matrix')

for i in range(num_classes):
	for j in range(num_classes):
		text = ax.text(j, i, format(cm[i, j], '.2f'), ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")

plt.colorbar(im)

# Save and close the plot
plt.savefig('ConfusionMatrix.pdf', bbox_inches = 'tight')
plt.close()

######################################################################

print()

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