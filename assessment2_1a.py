# -*- coding: utf-8 -*-
"""Assessment2.1a

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-U5IyKNu9FL54OEOYqeRDgripTpxUB3R
"""

# Import libraries
import tensorflow as tf
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from itertools import product
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras import layers
from keras import models
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout

# Load fashion mnist dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# summarize loaded dataset
print('Train: x=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: x=%s, y=%s' % (x_test.shape, y_test.shape))

# Data Exploration
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])

# Reshaping the images
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

# Normalizing the images
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# One-hot encoding to convert integer data into categorical data
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('Labels: y_train=%s, y_test=%s' % (y_train.shape, y_test.shape))

# Load fashion mnist dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshaping the images
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

# Normalizing the images
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Defining the CNN models
net = models.Sequential()
net.add(Conv2D(32,kernel_size=2, activation='relu', input_shape=(28,28,1))) # Input Layer
net.add(MaxPooling2D((2, 2)))
net.add(Dropout(0.3))
net.add(Conv2D(64, kernel_size=2, activation='relu'))
net.add(MaxPooling2D((2, 2)))
net.add(Dropout(0.3))
net.add(Flatten())
net.add(Dense(128, activation='relu'))
net.add(Dense(10, activation='softmax')) # Output Layer
# Model Summary
net.summary()

# Compiling model
net.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

history = net.fit(x_train, y_train, epochs = 20, batch_size = 150, validation_split=0.2)

testing_loss, testing_accuracy = net.evaluate(x_test, y_test)

print("Model accuracy: %.2f"% (testing_accuracy*100))

plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val_Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss evolution')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val_accuracy')
plt.title(" Accuracy Evolution")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

classes = ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

#Create Multiclass Confusion Matrix
preds = net.predict(x_test)

print(classification_report(np.argmax(y_test,axis=1), np.argmax(preds,axis=1), target_names= classes))

cm = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(preds,axis=1))

plt.figure(figsize=(8,8))
plt.imshow(cm,cmap=plt.cm.Blues)
plt.title('Fashion MNIST Confusion Matrix')
plt.xticks(np.arange(10), classes, rotation=90)
plt.yticks(np.arange(10), classes)

for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
    horizontalalignment="center",
    color="white" if cm[i, j] > 500 else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label');





