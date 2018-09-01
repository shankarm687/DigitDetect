# Test the model with first 10 test images from Mnist data set:

from tensorflow.keras.models import load_model
import tensorflow as tf
from utils.numpy_helpers import interpret_output_array
from utils.constants import MODEL_FILENAME


mnist = tf.keras.datasets.mnist
(_, _),(x_test, y_test) = mnist.load_data()
NUM_IMAGES = 10
images = x_test[0:NUM_IMAGES]

digit_model = load_model(MODEL_FILENAME)

output_array = digit_model.predict(images, NUM_IMAGES)
print interpret_output_array(output_array, NUM_IMAGES)

print 'expected_list' + str (y_test[0:NUM_IMAGES])