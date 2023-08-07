# importing libraries
import cv2
import argparse
from masking import get_mask, apply_mask
import time


def get_contours(image):
    '''
    Finds contours in a binary image.

    Parameters:
        image (numpy.ndarray): The input binary image.

    Returns:
        list: A list of contours found in the image.
    '''
    # print("[Console] Finding contours")
    contours = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


def get_diff_rect(image, diff_image, minDiffArea):
    '''
    Draws rectangles around the differences in an image.

    Parameters:
        image (numpy.ndarray): The input image.
        diff_image (numpy.ndarray): The difference image between two images.
        minDiffArea (int): The minimum contour area threshold for considering a difference.

    Returns:
        numpy.ndarray: The image with rectangles drawn around the differences.
    '''
    # print("[Console] Drawing rectangle around the differences")
    img = image.copy()
    contours = get_contours(diff_image)
    for c in contours:
        area = cv2.contourArea(c)
        if area > minDiffArea:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
    return img

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
if video.isOpened() == False:
    raise Exception("Error reading video")
else:
    TOTAL_FRAME = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = int(video.get(cv2.CAP_PROP_FPS))
    
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'avc1')
# result = cv2.VideoWriter('./output/output.mp4', fourcc, 30.0, size)

since = time.time()
frame_count = 0
while(1):
    # read frames
    ret, frame = video.read()
    elapsed = time.time() - since
    process_fps = round((frame_count + 1) / elapsed, 1)
    expect_time = (TOTAL_FRAME+1 - frame_count) // process_fps
    print(
        f"\rProcessing frame {frame_count}/{TOTAL_FRAME+1} in {(elapsed // 60):.0f}m{(elapsed % 60):.0f}s at speed {process_fps} FPS. Expect done in {(expect_time // 60):.0f}m {(expect_time % 60):.0f}s",
        end=" ",
        flush=True,
    )
    if ret:
        masked = apply_mask(frame, mask)
        
        # apply mask for background subtraction
        mog_frame = backSub.apply(masked)
                
        # apply transformation to remove noise
        mog_frame = cv2.morphologyEx(mog_frame, cv2.MORPH_OPEN, kernel)
        
        rect = get_diff_rect(frame, mog_frame, minDiffArea=750)
        # result.write(frame)
        # cv2.imshow('Frame', rect)
        cv2.imshow('MOG', mog_frame)
        frame_count +=1
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
    else:
        break     

video.release()
# result.release()
cv2.destroyAllWindows()