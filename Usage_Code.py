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

# Loading the model
model = tf.keras.models.load_model('Path/Digits_Model.model')  #use the path where you saved your model

#Evaluating the loss and accuracy
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

# Getting our own images:
image_number = 1
while os.path.isfile(f"Path image{image_number}.png"):    #input the path where the images are, the model takes 28x28 pixels images
  img = cv2.imread(f"Path image{image_number}.png")[:,:,0]    #input the path where the images are, the model takes 28x28 pixels images
  img = np.invert(np.array([img]))
#Predicting their value
  prediction = model.predict(img)
  print(f"This digit is propably a {np.argmax(prediction)}")
  plt.imshow(img[0], cmap=plt.cm.binary)
  plt.show()
  image_number += 1
