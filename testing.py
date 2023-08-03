import cv2
import numpy as np
import os

#Load the selected image
img = cv2.imread('./data/frame.jpg')
mask = cv2.imread('./data/blackframe.jpg')
mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]

mask = mask/255
# Apply the mask to image
# result = cv2.bitwise_and(mask,img)
result = np.array(np.multiply(mask, img), np.uint8)
print(result)

# save the processed image and mask
cv2.imshow("hfj", result)
cv2.imwrite('banana.jpg',result)
cv2.waitKey()