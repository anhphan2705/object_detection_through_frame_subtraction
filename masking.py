import numpy as np
import cv2

def get_mask(directory):
    mask = cv2.imread(directory)
    return cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]

def apply_mask(image, mask):
    bit_mask = mask/255
    return np.array(np.multiply(bit_mask, image), np.uint8)
    