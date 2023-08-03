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
    backSub = cv2.createBackgroundSubtractorMOG2()
  
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

mask = get_mask('./data/blackframe1.jpg')
  
# capture frames from a camera 
video = cv2.VideoCapture("./data/vnpt1.mp4")
if (video.isOpened() == False): 
    print("Error reading video file")
    
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter('./output/output.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         30, size)
while(1):
    # read frames
    ret, frame = video.read()
    if ret:
        frame = apply_mask(frame, mask)
        result.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    else:
        break

    # # # apply mask for background subtraction
    # fgmask = backSub.apply(img)
            
    # # apply transformation to remove noise
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # # after removing noise
    # cv2.imshow(f'{args.algo}', fgmask)
    # cv2.imshow(f'{args.algo}', img)

    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
       


video.release()
result.release()
cv2.destroyAllWindows()