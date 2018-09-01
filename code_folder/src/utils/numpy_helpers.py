import numpy as np

def interpret_output_array(output_array, array_size):
    predicted_list = []
    THRESHOLD_VALUE = 0.5 # Sample threshold to prevent non-digits from being recognized as digits
    for idx in range (0,array_size):
        if( np.max(output_array[idx]) > THRESHOLD_VALUE):
            predicted_list.append(np.argmax(output_array[idx]))
        else:
            predicted_list.append(-1) #'UNRECOGNIZED_SYMBOL' 
    return predicted_list
