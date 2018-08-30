from tensorflow.keras.models import load_model
import tensorflow as tf
from utils.numpy_helpers import interpret_output_array
import numpy as np
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

INPUT_IMAGE_FILENAME = "final_digits_input.png"
input_image = cv2.imread(INPUT_IMAGE_FILENAME, cv2.IMREAD_GRAYSCALE)
ROW_SIZE,COL_SIZE = input_image.shape
#Outputs were 66 x 480  (Height pixels vs Width) 

#Algorithm to partition numbers (Step 3):
# Initialize a list of start and end cols (s, e)
# Loop across width:
#     While not end:
#         When we encounter white pixel, initialize s, continue till e 
#         save (s,e) pair to list; and continue to the right
# Finally size of list gives the number, the s->e gives the width. Then cut out all the black rows top and bottom

list_start_end_tuples =[]

white_cols = np.amax(input_image, axis=0)
print white_cols
THRESHOLD_GRAYSCALE = 40
col_idx = 0
#Assumption black pixel padding on the left AND RIGHT
while (col_idx < COL_SIZE):
    if(white_cols[col_idx] < THRESHOLD_GRAYSCALE):
        col_idx+=1
    else:
        start_idx = col_idx
        end_idx = col_idx
        col_idx+=1
        while (end_idx<COL_SIZE and white_cols[end_idx]>THRESHOLD_GRAYSCALE):
            col_idx+=1  
            end_idx+=1
        list_start_end_tuples.append((start_idx, end_idx))
        
print list_start_end_tuples
#NUM_IMAGES = list_start_end_tuples.length

start = list_start_end_tuples[0][0]
end = list_start_end_tuples[0][1]
print start, end

# Cut off top and bottom for each digit:
def cut_and_resize(image):
    #Remove Top and bottom:
    white_rows = np.amax(image, axis=1)
    top = np.argmax(white_rows>THRESHOLD_GRAYSCALE)
    row_idx = 0
    #Assumption black pixel padding on the top AND bottom
    while (row_idx < ROW_SIZE and white_rows[row_idx] < THRESHOLD_GRAYSCALE):
        row_idx+=1
    start_idx = row_idx
    end_idx = row_idx
    while (end_idx<ROW_SIZE and white_rows[end_idx]>THRESHOLD_GRAYSCALE):
        row_idx+=1  
        end_idx+=1
    return start_idx,end_idx


crop_imgs=[]
idx=0
for each_tuple in list_start_end_tuples:
    crop_img= input_image[:, each_tuple[0]:each_tuple[1]]
    start_row,end_row = cut_and_resize(crop_img)
    final_crop = crop_img[start_row:end_row, :]
    crop_imgs.append(final_crop)
    #Test
    cv2.imwrite('image_'+str(idx)+'+.png',crop_imgs[idx])
    print crop_imgs[idx].shape # Max is around 34
    idx+=1

    #Test
    #cv2.imwrite('image_'+str(idx)+'+.png',out_img)

#TODO Step 4: Resize image to 28x28 (But checkout what size of the image is actually used.)  normalized to fit in a 20x20 pixel box 
# Before predicting, normalize pixels between 0 and 1 . References : http://yann.lecun.com/exdb/mnist/

# def Scale - From image size to 20x20
#for image in crop_imgs:
#    out_img= scale(image)

#TODO Step 5 

#output_array = digit_model.predict(images, NUM_IMAGES)
#print interpret_output_array(output_array, NUM_IMAGES)

#print 'expected_list' + str (y_test[0:NUM_IMAGES])
