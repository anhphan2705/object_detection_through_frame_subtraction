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


def get_diff_rect(image, diff_image, minDiffArea):
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
    contours = get_contours(diff_image)
    for c in contours:
        area = cv2.contourArea(c)
        if area > minDiffArea:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
    return img


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


if __name__ == "__main__":
    # # Marking the differences
    # first_rect = get_diff_rect(first_img, diff_img, 750)
    # second_rect = get_diff_rect(second_img, diff_img, 750)
    # mask = get_diff_mask(first_img, diff_img, 750)
    # filled_img = get_diff_filled(second_img, diff_img, 750)

    mask = get_mask("./data/mask/blackframe2.jpg")
    video = cv2.VideoCapture("./data/videos/vnpt1.mp4")
    if video.isOpened() == False:
        raise Exception("Error reading video")
    else:
        total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # result = cv2.VideoWriter('./output/out.mp4', fourcc, 30.0, size)

    frame_count = 0
    static_frame = None
    
    since = time.time()
    while True:
        # read frames
        ret, frame = video.read()
        elapsed = time.time() - since
        process_fps = (frame_count+1)//elapsed
        expect_time = (total_frame-frame_count+1) // process_fps
        print(f"\rProcessing frame {frame_count+1}/{total_frame} in {(elapsed // 60):.0f}m {(elapsed % 60):.0f}s at speed {process_fps} FPS. Expect done in {(expect_time // 60):.0f}m {(expect_time % 60):.0f}s", end=' ', flush=True)
        if ret:
            masked = apply_mask(frame, mask)
            if frame_count == 0:
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

            # filled = get_diff_filled(frame, dif_img, minDiffArea=750)
            rect = get_diff_rect(frame, dif_img, minDiffArea=750)

            # result.write(dif_img)
            # cv2.imshow("Frame", dif_img)
            cv2.imshow("Frame", rect)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("c"):
                break
        else:
            break

    video.release()
    # result.release()
    cv2.destroyAllWindows()
