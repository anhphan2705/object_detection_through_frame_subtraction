import cv2
import numpy as np
from skimage.exposure import equalize_adapthist
from utilities import set_ndarray

class PreProcessImage:
    
    def __init__(self, gray=True, contrast=False, blur=False, edge=False, mask_path=None):
        """
        This class is built to pre-process images with different tools such as
            - Convert to gray
            - Blur with bilateralFilter
            - Adjust contrast with equalize_adapthist from skimage
            - Get edge with Sobel
            - Apply any mask needed

        Parameters:
            gray (bool): Flag indicating whether to convert the image to grayscale. Default to True.
            contrast (bool): Flag indicating whether to adjust the contrast of the image. Default to False.
            blur (bool): Flag indicating whether to apply a blur to the image. Default to False.
            edge (bool): Flag indicating whether to detect edges in the image. Default to False.
            mask_path (str): Path to the mask image file.
        """
        self.gray = gray
        self.contrast = contrast
        self.blur = blur
        self.edge = edge
        self.mask = self.get_mask(mask_path) if mask_path else None
        
        
    def get_gray(self, image):
        """
        Convert an image from BGR color space to grayscale.

        Parameters:
            image (numpy.ndarray): The input image in BGR color space.

        Returns:
            numpy.ndarray: The grayscale image.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    def get_blur(self, image, d=30, sigColor=80, sigSpace=80):
        """
        Apply a bilateral filter blur to an image.

        Parameters:
            image (numpy.ndarray): The input image.
            d (int): Diameter of each pixel neighborhood.
            sigColor (float): Value of sigma in the color space.
            sigSpace (float): Value of sigma in the coordinate space.

        Returns:
            numpy.ndarray: The blurred image.
        """
        return cv2.bilateralFilter(image, d, sigColor, sigSpace)


    def get_equalize_adapt(self, image, c_limit=0.1):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the contrast of an image.

        Parameters:
            image (numpy.ndarray): The input image.
            c_limit (float): Clipping limit, normalized between 0 and 1.

        Returns:
            numpy.ndarray: The image with adjusted contrast.
        """
        equalized = equalize_adapthist(
            image, kernel_size=None, clip_limit=c_limit, nbins=256
        )
        return set_ndarray(equalized)


    def get_threshold(self, image):
        """
        Apply a binary thresholding operation to convert an image to a binary form.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The binary thresholded image.
        """
        return cv2.threshold(self, image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


    def get_edge(self, gray_img):
        """
        Detect edges in a grayscale image using the Sobel edge detection algorithm.

        Parameters:
            gray_img (numpy.ndarray): The input grayscale image.

        Returns:
            numpy.ndarray: The edge-detected image.
        """
        img_sobelx = cv2.Sobel(gray_img, -1, 1, 0, ksize=1)
        img_sobely = cv2.Sobel(gray_img, -1, 0, 1, ksize=1)
        img_sobel = cv2.addWeighted(img_sobelx, 0.5, img_sobely, 0.5, 0)
        return img_sobel 
    
    
    def get_mask(self, directory):
        """
        Load a mask image from a file.

        Parameters:
            directory (str): Path to the mask image file.

        Returns:
            numpy.ndarray: The mask image.
        """
        mask = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
        return cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]
    
    
    def set_mask(self, image, mask):
        """
        Apply a mask to an image.

        Parameters:
            image (numpy.ndarray): The input image.
            mask (numpy.ndarray): The mask to be applied.

        Returns:
            numpy.ndarray: The masked image.
        """
        bit_mask = mask/255
        return np.array(np.multiply(bit_mask, image), np.uint8)

    
    def get_preprocess(self, image):
        """
        Preprocess an image by applying various image processing techniques.

        Parameters:
            image (numpy.ndarray): The input image to be preprocessed.

        Returns:
            numpy.ndarray: The preprocessed image.
        """
        if self.gray:
            image = self.get_gray(image)
        if self.contrast:
            image = self.get_equalize_adapt(image)
        if self.blur:
            image = self.get_blur(image)
        if self.gray and self.edge:
            image = self.get_edge(image)
        if self.mask is not None:
            image = self.set_mask(image, self.mask)
        return image