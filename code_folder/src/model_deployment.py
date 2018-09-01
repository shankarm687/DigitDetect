from __future__ import print_function

from tensorflow.keras.models import load_model
import tensorflow as tf
from utils.numpy_helpers import interpret_output_array
from utils.constants import MODEL_FILENAME, THRESHOLD_GRAYSCALE, INPUT_IMAGE_FILENAME
import numpy as np
import cv2

# Detect start(left) and end(right) indices for each digit in multi-digit image :
def find_list_of_start_end_column_indices(multiple_digits_image):
    _, COL_SIZE = multiple_digits_image.shape
    list_start_end_tuples = []
    # If any of the pixels in a column is white (feature/digit), then mark the column as white
    white_pixel_cols = np.amax(multiple_digits_image, axis=0)

    #Loop across columns and keep detecting start and end indices for white pixels (represting digits):
    col_idx = 0
    while (col_idx < COL_SIZE):
        if(white_pixel_cols[col_idx] < THRESHOLD_GRAYSCALE):
            col_idx+=1
        else:
            start_idx = col_idx
            end_idx = col_idx
            col_idx+=1
            while (end_idx<COL_SIZE and white_pixel_cols[end_idx]>THRESHOLD_GRAYSCALE):
                col_idx+=1  
                end_idx+=1
            list_start_end_tuples.append((start_idx, end_idx))
    return list_start_end_tuples

# Find top and bottom indices for each digit image:
def find_top_and_bottom(digit_image):
    ROW_SIZE, _ = digit_image.shape
    #Remove Top and bottom:
    white_pixel_rows = np.amax(digit_image, axis=1)
    top = np.argmax(white_pixel_rows>THRESHOLD_GRAYSCALE)
    row_idx = 0
    #Assumption black pixel padding on the top AND bottom
    while (row_idx < ROW_SIZE and white_pixel_rows[row_idx] < THRESHOLD_GRAYSCALE):
        row_idx+=1
    start_idx = row_idx
    end_idx = row_idx
    while (end_idx<ROW_SIZE and white_pixel_rows[end_idx]>THRESHOLD_GRAYSCALE):
        row_idx+=1  
        end_idx+=1
    return start_idx,end_idx

def crop_top_and_bottom(digit_image):
    start_row,end_row = find_top_and_bottom(digit_image)
    return digit_image[start_row:end_row, :]

def resize_to_mnist_image(cropped_digit_img):
    resized_digit_image = cv2.resize(cropped_digit_img,(20,20))
    return cv2.copyMakeBorder(resized_digit_image, 4,4,4,4, borderType=cv2.BORDER_CONSTANT)

# Assumptions for the algorithm to work:
# 1. Images can be assumed to be white on black background. Else use OpenCV operations:
# Usual operations are convert to Grayscale; Increase constrast; Contouring to thicken the digit pixels. 
# 2. The input image has digits on a single line otherwise, the algorithm will change.
# 3. There is a border of black pixels both on all 4 sides of the image (This is a stronger stament of 1)
# References : http://yann.lecun.com/exdb/mnist/

# Algorithm to partition image into digit images:
# Step 1. Find a list of start and end column indices for each digit detected
# Step 2. Size of list represents the number of digits detected
# Step 3. Use the tuple indices to extract detected digits into their own images.
# Step 4. Crop the top and bottom, using grayscale thresholds.
# Step 5. Resize cropped digit into 20x20 grid, and add border to create 28x28 image
def segment_input_image_into_digit_images(multiple_digits_image):
    list_start_end_tuples = find_list_of_start_end_column_indices(multiple_digits_image)
    digit_images=[]
    num_digit_images=0
    for each_tuple in list_start_end_tuples:
        each_digit_img = multiple_digits_image[:, each_tuple[0]:each_tuple[1]] 
        cropped_digit_img = crop_top_and_bottom(each_digit_img)
        resized_digit_image = resize_to_mnist_image(cropped_digit_img)
        digit_images.append(resized_digit_image)
        #Test
        #cv2.imwrite('image2_'+str(num_digit_images)+'+.png',digit_images[num_digit_images])
        num_digit_images+=1
    return num_digit_images, digit_images

def stack_2d_arrays(list_of_images):
    y=np.dstack(list_of_images)
    y=np.rollaxis(y,-1) # To get the shape to be Nx28x28 instead of 28x28xN
    return y

def run_digit_recognition():
    input_image = cv2.imread(INPUT_IMAGE_FILENAME, cv2.IMREAD_GRAYSCALE)
    num_digit_images, digit_images = segment_input_image_into_digit_images(input_image)
    #Convert list of images into 3D np array (Nx28*28)
    np_digit_images = stack_2d_arrays(digit_images)
    # Before predicting, normalize pixels between 0 and 1 . 
    np_digit_images = np_digit_images / 255.0

    digit_model = load_model(MODEL_FILENAME) #TODO Add Validation when loading model
    output_array = digit_model.predict(np_digit_images, num_digit_images)
    return interpret_output_array(output_array, num_digit_images)

def lambda_handler(event, context):
    out_array = run_digit_recognition()
    print(out_array)

if __name__ == "__main__":
    out_array = run_digit_recognition()
    print(out_array)