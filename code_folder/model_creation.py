#Test tensorflow and numpy

import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import save_model, load_model
#from matplotlib import pyplot as plt
import cv2

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
image_0 = x_test[0]
print image_0
image = np.asarray(image_0, dtype="uint8")
cv2.imwrite('image_gray.png',image)

x_train, x_test = x_train / 255.0, x_test / 255.0




#plt.imshow(images[0].reshape(28, 28), cmap=plt.cm.binary)
#plt.show()

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# #TODO : Make this 5
# model.fit(x_train, y_train, epochs=5)

# #Save:
# save_model(model, 'digit_model.h5')
# del model # delete

# digit_model = load_model('digit_model.h5')
# #digit_model.evaluate(x_test, y_test)



# output_digits = digit_model.predict(images)
# print output_digits
# #model.predict()

# # Notes:
# # 1. Installed packages : virtualenv, tensorflow, h5py (for saving models)

