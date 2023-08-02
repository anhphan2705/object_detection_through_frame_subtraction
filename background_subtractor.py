# importing libraries
import numpy as np
import cv2

import argparse
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
 OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='./data/vid_1.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (MOG2, GMG, KNN).', default='MOG2')
args = parser.parse_args()

if args.algo == 'KNN':
    backSub = cv2.createBackgroundSubtractorKNN()
elif args.algo == "GMG":
    backSub = cv2.bgsegm.createBackgroundSubtractorGMG()
elif args.algo == "MOG":
    backSub = cv2.bgsegm.createBackgroundSubtractorMOG()
else:
    backSub = cv2.createBackgroundSubtractorMOG2()
  
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  
# capture frames from a camera 
cap = cv2.VideoCapture("./data/vid_1.mp4")
while(1):
    # read frames
    ret, img = cap.read()
    if img is None:
        break

    # # apply mask for background subtraction
    fgmask = backSub.apply(img)
            
    # apply transformation to remove noise
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
      
    # after removing noise
    cv2.imshow('MOG2', fgmask)
      
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()