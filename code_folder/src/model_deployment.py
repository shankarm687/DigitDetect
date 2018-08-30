from tensorflow.keras.models import load_model
import tensorflow as tf
from utils.numpy_helpers import interpret_output_array
import cv2

# TODO: Put in a constants file
MODEL_FILENAME = "build/model/digit_model.h5"

mnist = tf.keras.datasets.mnist

(_, _),(x_test, y_test) = mnist.load_data()

NUM_IMAGES = 10
images = x_test[0:NUM_IMAGES]

#TODO Add Validation when loading model
digit_model = load_model(MODEL_FILENAME)

# TODO :Confirm with Pronoy if Images can be assumed to be white on black background. Else use OpenCV operations:
# Usual operations are convert to Grayscale; Increase constrast; Contouring to thicken the digit pixels. 

# TODO Create images, NUM_IMAGES from digit_input file. Move above images to test.
#STeps : 1. Read digit_input; 2. Figure out size in pixels of image in pixels 3. Cut image into digit blocks
# 4. Resize number blocks to 28x28 5. Use the model.predict funtion below (NUM_IMAGES = Number of digit-blocks)

INPUT_IMAGE_FILENAME = "digits_input.png"
input_image = cv2.imread(INPUT_IMAGE_FILENAME, cv2.IMREAD_GRAYSCALE)
print  input_image.shape
#Outputs were 66 x 480  (Height pixels vs Width) 

#TODO Steps 3, 4 ,5 
# Before predicting, normalize pixels between 0 and 1

output_array = digit_model.predict(images, NUM_IMAGES)
print interpret_output_array(output_array, NUM_IMAGES)

print 'expected_list' + str (y_test[0:NUM_IMAGES])
