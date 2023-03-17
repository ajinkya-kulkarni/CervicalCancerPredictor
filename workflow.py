
import os

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

#####################################################################################

from modules import *

from parameters import *

#####################################################################################

# os.system('cls || clear')

#####################################################################################

# Preprocess images and convert labels to integers

preprocessed_images, labels = preprocess_images(folder_paths, img_size)

print()

#####################################################################################

# Convert string labels to numerical labels

le = LabelEncoder()

labels = le.fit_transform(labels)

#####################################################################################

# Generate augmented images for all input images

augmented_images, augmented_labels = augment_images(img_size, preprocessed_images, labels, n_augmentations=n_augmentations)

print()

#####################################################################################

# Select a random image to plot

idx = np.random.randint(preprocessed_images.shape[0])

# Create a list of images to plot (original + augmented)

images = [preprocessed_images[idx]]
images.extend(augmented_images[i] for i in range(idx*n_augmentations, (idx+1)*n_augmentations))

# Create a list of titles for the subplots

titles = ['Original']
titles.extend([f"Augmentation #{i+1}" for i in range(n_augmentations)])

# Plot the images as subplots in a grid

num_images = len(images)
num_rows = int(np.sqrt(num_images))
num_cols = int(np.ceil(num_images / num_rows))
fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
	if i < num_images:
		ax.imshow(images[i])
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title(titles[i])
	else:
		ax.axis('off')
		
# Delete last subplot if empty
if num_images < num_rows * num_cols:
	fig.delaxes(axes.flat[-1])

plt.tight_layout()
plt.savefig('Augmentations.png', dpi = 200)
plt.close()

#####################################################################################

# Check if each label in the original dataset is equal to the label of its corresponding augmented images

for i in range(labels.shape[0]):
	label = labels[i]
	assert all(augmented_labels[i*n_augmentations:(i+1)*n_augmentations] == label)

# Combine the original and augmented images

all_images = np.concatenate((preprocessed_images, augmented_images), axis=0)

# Combine the original and augmented labels

all_labels = np.concatenate((labels, np.repeat(labels, n_augmentations)), axis=0)

# Check if the number of images and labels in the combined dataset is equal

assert all_images.shape[0] == all_labels.shape[0]

#####################################################################################

# Split data into training and testing sets, and convert labels to one-hot encoding

X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=split_percentage, random_state=42)

num_classes = len(np.unique(all_labels))

y_train = to_categorical(y_train, num_classes)

y_test = to_categorical(y_test, num_classes)

#####################################################################################

# input_shape = X_train.shape[1:]
# model = create_model(input_shape, num_classes, learning_rate)

model = new_model(X_train, regularization = regularizationkey)

#####################################################################################


# Print the model summary

model.summary()

print()

#####################################################################################

train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

with tqdm(total=epochs) as pbar:
	for epoch in range(epochs):
		
		history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0, validation_data=(X_test, y_test))

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

#####################################################################################

# Generate predictions for the test data

y_pred = model.predict(X_test)

# Convert predictions to labels

y_pred_labels = np.argmax(y_pred, axis=1)

y_true_labels = np.argmax(y_test, axis=1)

#####################################################################################

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

plt.savefig('Result.png', dpi = 200)
plt.close()

#####################################################################################

# Define class dictionary
class_dict = {0: 'group_1', 1: 'group_2', 2: 'group_3'}

#####################################################################################

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
		text = ax.text(j, i, format(cm[i, j], '.2f'),
					ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")

plt.colorbar(im)
plt.savefig('confusion_matrix.png', dpi = 200)
plt.close()

#####################################################################################

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
print()
print(f'Test accuracy: {accuracy:.3f}')
print(f'Test precision: {precision:.3f}')
print(f'Test recall: {recall:.3f}')
print(f'Test F1-score: {f1:.3f}')
print()
print(f'Test confusion matrix:\n{cm}')
print()

#####################################################################################





