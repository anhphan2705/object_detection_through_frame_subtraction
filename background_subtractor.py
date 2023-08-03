# importing libraries
import cv2
import argparse
from masking import get_mask, apply_mask

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
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
  
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

mask = get_mask('./data/mask/blackframe2.jpg')
  
# capture frames from a camera 
video = cv2.VideoCapture("./data/videos/vnpt1.mp4")
if (video.isOpened() == False): 
    print("Error reading video file")
    
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'avc1')
result = cv2.VideoWriter('./output/output.mp4', fourcc, 30.0, size)

while(1):
    # read frames
    ret, frame = video.read()
    if ret:
        frame = apply_mask(frame, mask)
        
        # apply mask for background subtraction
        mog_frame = backSub.apply(frame)
                
        # apply transformation to remove noise
        mog_frame = cv2.morphologyEx(mog_frame, cv2.MORPH_OPEN, kernel)
        
        result.write(frame)
        # cv2.imshow('Frame', frame)
        cv2.imshow('MOG', mog_frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
    else:
        break     

video.release()
result.release()
cv2.destroyAllWindows()