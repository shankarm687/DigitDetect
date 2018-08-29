from tensorflow.keras.models import load_model
import tensorflow as tf

mnist = tf.keras.datasets.mnist

digit_model = load_model('digit_model.h5')
images = mnist.test.images[0:10]


output_digits = digit_model.predict(images)
print output_digits