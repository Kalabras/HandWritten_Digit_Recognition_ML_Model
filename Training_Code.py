# importing all the relevant libraries:
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# loading the dataset:
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalizing the data so that it is between 0 and 1:
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# defining the model we're gonna use:
model = tf.keras.models.Sequential()

# defining the layers:
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  # Flat input layer
model.add(tf.keras.layers.Dense(128, activation='relu')) # Hidden Layer
model.add(tf.keras.layers.Dense(128, activation='relu')) # Hidden Layer
model.add(tf.keras.layers.Dense(10, activation='softmax')) # Output layer with 10 outputs

# Compiling the Model:
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"] )

# Training the model!!
model.fit(x_train, y_train, epochs=15)
model.save('Digits_Model.model')   #you can replace 'Digits_Model with the path where you want to save your model
