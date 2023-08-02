# importing libraries
import numpy as np
import cv2
  
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  
fgbg = cv2.createBackgroundSubtractorMOG2()
  
# capture frames from a camera 
cap = cv2.VideoCapture("./data/vid_1.mp4")
while(1):
    # read frames
    ret, img = cap.read()

    # # apply mask for background subtraction
    fgmask = fgbg.apply(img)
            
    # apply transformation to remove noise
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
      
    # after removing noise
    cv2.imshow('MOG2', fgmask)
      
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()