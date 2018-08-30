#Test tensorflow and numpy

import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import save_model, load_model
import cv2

# Load Data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

for idx in range(0,10):
    image = np.asarray(x_test[idx], dtype="uint8")
    cv2.imwrite('image_gray'+str(idx)+'.png',image)

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

#Save:
save_model(model, 'digit_model.h5')
del model # delete

# Notes:
# 1. Installed packages : virtualenv, tensorflow, h5py (for saving models)

