from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(_, _),(x_test, y_test) = mnist.load_data()

TEST_DIM = 10
images = x_test[0:TEST_DIM]

def interpret_output_array(output_array, array_size):
    predicted_list = []
    THRESHOLD_VALUE = 0.8 # Sample threshold to prevent non-digits from being recognized as digits
    for idx in range (0,array_size):
        if( np.max(output_array[idx]) > THRESHOLD_VALUE):
            predicted_list.append(np.argmax(output_array[idx]))
        else:
            predicted_list += -1 #'UNRECOGNIZED_SYMBOL' 
    return predicted_list

digit_model = load_model('digit_model.h5')
output_array = digit_model.predict(images, TEST_DIM)
print interpret_output_array(output_array, TEST_DIM)

print 'expected_list' + str (y_test[0:TEST_DIM])