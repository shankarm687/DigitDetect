#Test tensorflow and numpy

import tensorflow as tf
import pickle

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
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

#TODO : Make this 5
model.fit(x_train, y_train, epochs=1)

#Save:
tf.keras.models.save_model(model, 'digit_model.h5')
del model # delete

rebuilt_model = tf.keras.models.load_model('digit_model.h5')
rebuilt_model.evaluate(x_test, y_test)

#model.predict()

# Notes:
# 1. Installed packages : virtualenv, tensorflow, h5py (for saving models)

