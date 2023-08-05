from skimage.exposure import equalize_adapthist
import cv2
import numpy as np
from masking import get_mask, apply_mask
import time


def convert_to_gray(image):
    """
    Converts an image from BGR color space to grayscale.

    Parameters:
        image (numpy.ndarray): The input image in BGR color space.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_to_cv2_format(image):
    """
    Converts an image with any data type to a format readable by OpenCV (BGR, uint8).

    Parameters:
        image (SSIM image): The input image with any data type

    Returns:
        numpy.ndarray: The image converted to OpenCV readable format (BGR, uint8).
    """
    image = (image * 255).astype("uint8")
    return image


def get_blur(image, d=30, sigColor=80, sigSpace=80):
    """
    Applies a bilateral filter blur to an image.

    Parameters:
        image (numpy.ndarray): The input image.
        d (int): Diameter of each pixel neighborhood.
        sigColor (float): Value of sigma in the color space.
        sigSpace (float): Value of sigma in the coordinate space.

    Returns:
        numpy.ndarray: The blurred image.
    """
    return cv2.bilateralFilter(image, d, sigColor, sigSpace)


def get_equalize_adapt(image, c_limit=0.1):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the contrast of an image.

    Parameters:
        image (numpy.ndarray): The input image.
        c_limit (float): Clipping limit, normalized between 0 and 1.

    Returns:
        numpy.ndarray: The image with adjusted contrast.
    """
    equalized = equalize_adapthist(
        image, kernel_size=None, clip_limit=c_limit, nbins=256
    )
    return convert_to_cv2_format(equalized)


def get_threshold(image):
    """
    Applies a binary thresholding operation to convert an image to a binary form.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The binary thresholded image.
    """
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def get_edge(gray_img):
    """
    Detects edges in a grayscale image using the Sobel edge detection algorithm.

    Parameters:
        gray_img (numpy.ndarray): The input grayscale image.

    Returns:
        numpy.ndarray: The edge-detected image.
    """
    img_sobelx = cv2.Sobel(gray_img, -1, 1, 0, ksize=1)
    img_sobely = cv2.Sobel(gray_img, -1, 0, 1, ksize=1)
    img_sobel = cv2.addWeighted(img_sobelx, 0.5, img_sobely, 0.5, 0)
    return img_sobel


def get_contours(image):
    """
    Finds contours in a binary image.

    Parameters:
        image (numpy.ndarray): The input binary image.

    Returns:
        list: A list of contours found in the image.
    """
    # print("[Console] Finding contours")
    # threshold_img = get_threshold(image)
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


def get_dif_box(image, diff_image, minDiffArea):
    """
    Draws rectangles around the differences in an image.

    Parameters:
        image (numpy.ndarray): The input image.
        diff_image (numpy.ndarray): The difference image between two images.
        minDiffArea (int): The minimum contour area threshold for considering a difference.

    Returns:
        numpy.ndarray: The image with rectangles drawn around the differences.
    """
    # print("[Console] Drawing rectangle around the differences")
    img = image.copy()
    box_pos = []
    contours = get_contours(diff_image)
    for c in contours:
        area = cv2.contourArea(c)
        if area > minDiffArea:
            x, y, w, h = cv2.boundingRect(c)
            new_box = [x, y, x + w, y + h]
            if len(box_pos) > 0:
                valid = True
                for box in box_pos:
                    iou = get_iou(new_box, box)
                    if iou > 0.02:
                        x1, y1, x2, y2 = new_box
                        area_new = (x2 - x1) * (y2 - y1)
                        x1, y1, x2, y2 = box
                        area_box = (x2 - x1) * (y2 - y1)
                        if area_new > area_box:
                            box_pos.append(new_box)
                            if box in box_pos:
                                box_pos.remove(box)
                        valid = False
                        break
                if valid:
                    box_pos.append(new_box)
            else:
                box_pos.append(new_box)
    for box in box_pos:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)

    return img, box_pos


def get_diff_filled(image, diff_image, minDiffArea=750):
    """
    Fills the differences in an image with a specific color.

    Parameters:
        image (numpy.ndarray): The input image.
        diff_image (numpy.ndarray): The difference image between two images.
        minDiffArea (int): The minimum contour area threshold for considering a difference.

    Returns:
        numpy.ndarray: The image with differences filled.
    """
    contours = get_contours(diff_image)
    for c in contours:
        area = cv2.contourArea(c)
        if area > minDiffArea:
            cv2.drawContours(image, [c], 0, (0, 255, 0), -1)
    return image


def preprocess_image(image, gray=True, contrast=False, blur=False, edge=False):
    """
    Preprocesses an image by applying various image processing techniques.

    Parameters:
        image (numpy.ndarray): The input image to be preprocessed.
        gray (bool): Flag indicating whether to convert the image to grayscale (default: True).
        contrast (bool): Flag indicating whether to adjust the contrast of the image (default: False).
        blur (bool): Flag indicating whether to apply a blur to the image (default: False).
        edge (bool): Flag indicating whether to detect edges in the image (default: False).

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    if gray:
        image = convert_to_gray(image)
    # show_image("Gray", image)
    # write_image("./gray.jpg", image)
    if contrast:
        image = get_equalize_adapt(
            image
        )  # Optional. Adjust contrast level through skimage.exposure.equalize_adapthist
    # show_image("Adjust Contrast", image)
    # write_image("./equal.jpg", image)
    if blur:
        image = get_blur(
            image
        )  # Optional. Bilateral Filter Blur for edge detect purpose
    # show_image("Blur", image)
    # write_image("./blur.jpg", image)
    if edge:
        image = get_edge(
            image
        )  # Optional. Detect different object through shape mainly, less dependent on color and noise
    # show_image("Edge", image)
    # write_image("./edge.jpg", image)
    return image


def get_iou(a, b, epsilon=1e-5):
    """Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = x2 - x1
    height = y2 - y1
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def still_object_detection(cur_obj, new_obj, wiggle_percent=0.85):
    still_obj = []
    if len(cur_obj) == 0:
        return new_obj
    else:
        for obj in new_obj:
            for still in cur_obj:
                iou = get_iou(still, obj)
                if iou > wiggle_percent:
                    still_obj.append(obj)
    if len(still_obj) == 0:
        return new_obj
    else:
        return still_obj
    
def label_time_obj(frame, key, time_still, font_size=0.8):
    x1, y1, x2, y2 = box_pos
    frame = cv2.putText(frame,
                        f'ID={key}',
                        org = (x1, y2),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=font_size, 
                        color=(0, 255, 0), 
                        thickness=1, 
                        lineType=cv2.LINE_AA)
    frame = cv2.putText(frame,
                        f'Time:{(time_still // 60):.0f}m {(time_still % 60):.0f}',
                        org = (x1, y1),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=font_size, 
                        color=(0, 255, 0), 
                        thickness=1, 
                        lineType=cv2.LINE_AA)
    return frame
        
        
if __name__ == "__main__":
    # mask = get_mask("./data/mask/blackframe2.jpg")
    # video = cv2.VideoCapture("./data/videos/vnpt1.mp4")
    mask = get_mask(
        "C:\\Users\\black\\Documents\\VNPT\\Object-Differences-Video\\data\\mask\\blackframe2.jpg"
    )
    video = cv2.VideoCapture(
        "C:\\Users\\black\\Documents\\VNPT\\Object-Differences-Video\\data\\videos\\vnpt1.mp4"
    )
    if video.isOpened() == False:
        raise Exception("Error reading video")
    else:
        TOTAL_FRAME = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        FPS = int(video.get(cv2.CAP_PROP_FPS))
        FRAME_WIDTH = int(video.get(3))
        FRAME_HEIGHT = int(video.get(4))
        FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    # result = cv2.VideoWriter('./output/out.mp4', fourcc, 30.0, FRAME_SIZE)
    # result = cv2.VideoWriter(
    #     "C:\\Users\\black\\Documents\\VNPT\\Object-Differences-Video/output/out.mp4",
    #     fourcc,
    #     30.0,
    #     FRAME_SIZE,
    # )
    # dif = cv2.VideoWriter('./output/dif.mp4', fourcc, 30.0, FRAME_SIZE)

    frame_count = 0
    static_frame = None
    temp_obj = []
    still_obj = {}

    since = time.time()
    while True:
        # read frames
        ret, frame = video.read()
        frame_count += 1
        elapsed = time.time() - since
        process_fps = round((frame_count + 1) / elapsed, 1)
        expect_time = (TOTAL_FRAME+1 - frame_count) // process_fps
        print(
            f"\rProcessing frame {frame_count}/{TOTAL_FRAME+1} in {(elapsed // 60):.0f}m {(elapsed % 60):.0f}s at speed {process_fps} FPS. Expect done in {(expect_time // 60):.0f}m {(expect_time % 60):.0f}s",
            end=" ",
            flush=True,
        )
        if ret:
            masked = apply_mask(frame, mask)
            if frame_count == 1:
                static_frame = masked
                static_frame = preprocess_image(
                    static_frame, gray=True, contrast=False, blur=False, edge=False
                )
            if frame_count % 1 == 0:
                cur_frame = preprocess_image(
                    masked, gray=True, contrast=False, blur=False, edge=False
                )
            dif_img = np.subtract(cur_frame, static_frame)

            # Absolute white to black
            white_loc = np.where(dif_img > 225)
            dif_img[white_loc] = 0
            # Black-ish to black
            black_loc = np.where(dif_img < 50)
            dif_img[black_loc] = 0
            # Object to white
            obj_loc = np.where((dif_img > 50) & (dif_img < 225))
            dif_img[obj_loc] = 255
            dif_img = cv2.dilate(dif_img, (7, 7))

            rect, box_pos = get_dif_box(frame, dif_img, minDiffArea=750)
            
            # Calculating still object
            if frame_count % FPS == 0:
                still_pos = []
                last_obj = temp_obj
                temp_obj = still_object_detection(temp_obj, box_pos, wiggle_percent=0.87)
                
                # Reseting active object status
                if still_obj is not None:
                    for key, value in still_obj.items():
                        value[0] = False
                        still_obj.update({key:value})
                for old in last_obj:
                    for new in temp_obj:
                        iou = get_iou(new, old)
                        if iou > 0.87:
                            still_pos.append(new)
                if still_obj is None:
                    for i, pos in enumerate(still_pos):
                        still_obj.update({i:[True, frame_count-3*FPS, frame_count, pos]})
                        # cv2.imwrite(f"./output/still_obj/still_{frame_count}_{i}.jpg", frame[pos[1]: pos[3], pos[0]: pos[2]])
                        cv2.imwrite(f"C:\\Users\\black\\Documents\\VNPT\\Object-Differences-Video\\output\\still_obj\\still_{frame_count}_{i}.jpg", frame[pos[1]: pos[3], pos[0]: pos[2]])
                else:
                    for pos in still_pos.copy():
                        for key, value in still_obj.items():
                            past_still = value[3]
                            iou = get_iou(past_still, pos)
                            if iou > 0.87:
                                value = [True, value[1], frame_count, pos]
                                still_obj.update({key:value})
                                if len(still_pos) > 0:
                                    still_pos.remove(pos)
                                break
                    if len(still_pos) > 0:
                        for pos in still_pos:
                            still_obj.update({len(still_obj):[True, frame_count-3*FPS, frame_count, pos]})  
                            # cv2.imwrite(f"./output/still_obj/still_{frame_count}_{i}.jpg", frame[pos[1]: pos[3], pos[0]: pos[2]])
                            cv2.imwrite(f"C:\\Users\\black\\Documents\\VNPT\\Object-Differences-Video\\output\\still_obj\\still_{frame_count}_{len(still_obj)-1}.jpg", frame[pos[1]: pos[3], pos[0]: pos[2]])          
                print(temp_obj)
                print(still_obj)
            # dif.write(dif_img)
            # result.write(rect)
            # cv2.imshow("Frame", dif_img)
            
            # Write time
            for key, value in still_obj.items():
                [active, start_frame, end_frame, box_pos] = value
                if active:
                    x1, y1, x2, y2 = box_pos
                    time_still = int(round((end_frame - start_frame) / FPS, 0))
                    rect = label_time_obj(rect, key, time_still, font_size=0.8)
            # Show frame
            cv2.imshow("Frame", rect)
            if cv2.waitKey(1) & 0xFF == ord("c"):
                break
        else:
            break

    video.release()
    # result.release()
    # dif.release()
    cv2.destroyAllWindows()
