from skimage.metrics import structural_similarity
from skimage.exposure import equalize_adapthist
import cv2
import numpy as np


def get_image(directory):
    '''
    Loads and returns an image from the specified directory.

    Parameters:
        directory (str): The directory path of the image file.

    Returns:
        numpy.ndarray: The loaded image.
    '''
    print("[Console] Getting image")
    return cv2.imread(directory)


def show_image(header, image):
    '''
    Displays an image in a new window.

    Parameters:
        header (str): The window title/header.
        image (numpy.ndarray): The image to be displayed.
    '''
    print("[Console] Showing image")
    cv2.imshow(header, image)
    cv2.waitKey()


def write_image(directory, image):
    '''
    Saves an image to the specified directory.

    Parameters:
        directory (str): The directory path to save the image.
        image (numpy.ndarray): The image to be saved.
    '''
    print("[Console] Saving image")
    cv2.imwrite(directory, image)


def resize_image(image, height, width):
    '''
    Resizes an image to the specified dimensions.

    Parameters:
        image (numpy.ndarray): The image to be resized.
        height (int): The desired height of the resized image.
        width (int): The desired width of the resized image.

    Returns:
        tuple: A tuple containing the new dimensions (height, width) and the resized image.
    '''
    print("[Console] Resizing image to 720p")
    dim = (height, width)
    return dim, cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def convert_to_gray(image):
    '''
    Converts an image from BGR color space to grayscale.

    Parameters:
        image (numpy.ndarray): The input image in BGR color space.

    Returns:
        numpy.ndarray: The grayscale image.
    '''
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_to_cv2_format(image):
    '''
    Converts an image with any data type to a format readable by OpenCV (BGR, uint8).

    Parameters:
        image (SSIM image): The input image with any data type

    Returns:
        numpy.ndarray: The image converted to OpenCV readable format (BGR, uint8).
    '''
    image = (image * 255).astype("uint8")
    return image


def get_blur(image, d=30, sigColor=80, sigSpace=80):
    '''
    Applies a bilateral filter blur to an image.

    Parameters:
        image (numpy.ndarray): The input image.
        d (int): Diameter of each pixel neighborhood.
        sigColor (float): Value of sigma in the color space.
        sigSpace (float): Value of sigma in the coordinate space.

    Returns:
        numpy.ndarray: The blurred image.
    '''
    return cv2.bilateralFilter(image, d, sigColor, sigSpace)


def get_equalize_adapt(image, c_limit=0.1):
    '''
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the contrast of an image.

    Parameters:
        image (numpy.ndarray): The input image.
        c_limit (float): Clipping limit, normalized between 0 and 1.

    Returns:
        numpy.ndarray: The image with adjusted contrast.
    '''
    equalized = equalize_adapthist(image, kernel_size=None, clip_limit=c_limit, nbins=256)
    return convert_to_cv2_format(equalized)


def get_threshold(image):
    '''
    Applies a binary thresholding operation to convert an image to a binary form.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The binary thresholded image.
    '''
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def get_edge(gray_img):
    '''
    Detects edges in a grayscale image using the Sobel edge detection algorithm.

    Parameters:
        gray_img (numpy.ndarray): The input grayscale image.

    Returns:
        numpy.ndarray: The edge-detected image.
    '''
    img_sobelx = cv2.Sobel(gray_img, -1, 1, 0, ksize=1)
    img_sobely = cv2.Sobel(gray_img, -1, 0, 1, ksize=1)
    img_sobel = cv2.addWeighted(img_sobelx, 0.5, img_sobely, 0.5, 0)
    return img_sobel


def get_contours(image):
    '''
    Finds contours in a binary image.

    Parameters:
        image (numpy.ndarray): The input binary image.

    Returns:
        list: A list of contours found in the image.
    '''
    print("[Console] Finding contours")
    threshold_img = get_threshold(image)
    contours = cv2.findContours(
        threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


def get_diff_mask(image, diff_image, minDiffArea):
    '''
    Generates a mask image highlighting the differences between two images.

    Parameters:
        image (numpy.ndarray): The input image.
        diff_image (numpy.ndarray): The difference image between two images.
        minDiffArea (int): The minimum contour area threshold for considering a difference.

    Returns:
        numpy.ndarray: The mask image with differences marked.
    '''
    mask = np.zeros(image.shape, dtype="uint8")
    contours = get_contours(diff_image)
    for c in contours:
        area = cv2.contourArea(c)
        if area > minDiffArea:
            cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
    return mask


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
    print("[Console] Drawing rectangle around the differences")
    img = image.copy()
    contours = get_contours(diff_image)
    for c in contours:
        area = cv2.contourArea(c)
        if area > minDiffArea:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
    return img


def get_diff_filled(image, diff_image, minDiffArea):
    '''
    Fills the differences in an image with a specific color.

    Parameters:
        image (numpy.ndarray): The input image.
        diff_image (numpy.ndarray): The difference image between two images.
        minDiffArea (int): The minimum contour area threshold for considering a difference.

    Returns:
        numpy.ndarray: The image with differences filled.
    '''
    contours = get_contours(diff_image)
    for c in contours:
        area = cv2.contourArea(c)
        if area > minDiffArea:
            cv2.drawContours(image, [c], 0, (0, 255, 0), -1)
    return image


def get_structural_similarity(first_image, second_image):
    '''
    Calculates the structural similarity between two images.

    Parameters:
        first_image (numpy.ndarray): The first input image.
        second_image (numpy.ndarray): The second input image.

    Returns:
        float: The structural similarity index between the two images.
        numpy.ndarray: The difference image highlighting the dissimilarities.
    '''
    print("[Console] Calculating differences")
    (score, diff_img) = structural_similarity(first_image, second_image, full=True)
    diff_img = convert_to_cv2_format(diff_img)
    print("[Console] Similarity score of {:.4f}%".format(score * 100))
    return score, diff_img


def preprocess_image(image, gray=True, contrast=False, blur=False, edge=False):
    '''
    Preprocesses an image by applying various image processing techniques.

    Parameters:
        image (numpy.ndarray): The input image to be preprocessed.
        gray (bool): Flag indicating whether to convert the image to grayscale (default: True).
        contrast (bool): Flag indicating whether to adjust the contrast of the image (default: False).
        blur (bool): Flag indicating whether to apply a blur to the image (default: False).
        edge (bool): Flag indicating whether to detect edges in the image (default: False).

    Returns:
        numpy.ndarray: The preprocessed image.
    '''
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
    cap = cv2.VideoCapture('./data/vid_1.mp4')
    # Get Image
    first_img = get_image("./images/real/1.jpg")
    second_img = get_image("./images/real/2.jpg")

    # Resize image if there is a difference in size
    # Modify this if needed
    if first_img.shape != second_img.shape:
        first_img = resize_image(first_img, 1280, 720)[1]
        second_img = resize_image(second_img, 1280, 720)[1]

    # Preprocess the image before comparing
    # Main step for the accuracy of the program
    # Only set True for the methods that are needed for the processing images, otherwise False
    # Remember for process both image the same
    first_pre = preprocess_image(
        first_img, 
        gray=True, 
        contrast=True, 
        blur=True, 
        edge=True
    )
    second_pre = preprocess_image(
        second_img, 
        gray=True, 
        contrast=True, 
        blur=True, 
        edge=True
    )

    # Compare and get the result
    score, diff_img = get_structural_similarity(first_pre, second_pre)

    # Marking the differences
    first_rect = get_diff_rect(first_img, diff_img, 750)
    second_rect = get_diff_rect(second_img, diff_img, 750)
    mask = get_diff_mask(first_img, diff_img, 750)
    filled_img = get_diff_filled(second_img, diff_img, 750)
