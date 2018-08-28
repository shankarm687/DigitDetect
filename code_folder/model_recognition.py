#Test tensorflow and numpy

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

import numpy as np
out = np.array([1, 2, 3])
print out

