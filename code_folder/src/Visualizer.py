import numpy as np
import cv2

def visualize_images(array_images, num_images):
    for idx in range(0,num_images):
        image = np.asarray(x_test[idx], dtype="uint8")
        cv2.imwrite('image_'+str(idx)+'.png',image)
