#Test tensorflow and numpy

import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import save_model
import os
import errno

def create_model():
    MODEL_FILENAME = "build/model/digit_model.h5"

    # Load Data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    # Normalize pixel grayscale to be between 0 and 1:
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

    #Create Path
    if not os.path.exists(os.path.dirname(MODEL_FILENAME)):
        try:
            os.makedirs(os.path.dirname(MODEL_FILENAME))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    save_model(model, MODEL_FILENAME)
    del model # delete

if __name__ == "__main__":
    import sys
    create_model()
# Notes:
# 1. Installed packages : virtualenv, tensorflow, h5py (for saving models)
# 2. Add pylint; Google python style guide to enforce patterns.
# 3. Add References : http://yann.lecun.com/exdb/mnist/