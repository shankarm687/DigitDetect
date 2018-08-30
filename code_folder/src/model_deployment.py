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
# Also assumed is that the input image has digits on a single line otherwise, the algorithm will change.

# TODO Create images, NUM_IMAGES from digit_input file. Move above images to test.
#STeps : 1. Read digit_input; 2. Figure out size in pixels of image in pixels 3. Cut image into digit blocks
# 4. Resize number blocks to 28x28 5. Use the model.predict funtion below (NUM_IMAGES = Number of digit-blocks)

INPUT_IMAGE_FILENAME = "digits_input.png"
input_image = cv2.imread(INPUT_IMAGE_FILENAME, cv2.IMREAD_GRAYSCALE)
print  input_image.shape
#Outputs were 66 x 480  (Height pixels vs Width) 

#Algorithm to partition numbers (Step 3):
# Initialize a list of start and end cols (s, e)
# Loop across width:
#     While not end:
#         When we encounter white pixel, initialize s, continue till e 
#         save (s,e) pair to list; and continue to the right
# Finally size of list gives the number, the s->e gives the width. Then cut out all the black rows top and bottom


#TODO Step 4: Resize image to 28x28 (But checkout what size of the image is actually used.)  normalized to fit in a 20x20 pixel box 
# Before predicting, normalize pixels between 0 and 1 . References : http://yann.lecun.com/exdb/mnist/
#TODO Step 5 

output_array = digit_model.predict(images, NUM_IMAGES)
print interpret_output_array(output_array, NUM_IMAGES)

print 'expected_list' + str (y_test[0:NUM_IMAGES])
