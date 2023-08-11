import cv2
import ast

class Tracking:
    
    DEFAULT_TRACK_RATE = 30
    DEFAULT_MIN_SIZE = 750
    DEFAULT_IOU_THRESHOLD = 0.87
    
    
    def __init__(self, track_rate=DEFAULT_TRACK_RATE, ignore_path=None, min_size=DEFAULT_MIN_SIZE, iou_threshold=DEFAULT_IOU_THRESHOLD):
        self.TRACKRATE = track_rate
        self.ignores = self.get_ignore_list(ignore_path) if ignore_path else None
        self.MIN_SIZE_THRESHOLD = min_size
        self.IOU_THRESHOLD = iou_threshold
        

    def get_contours(self, image):
        """
        Finds contours in a binary image and arrange it from largest area to smallest.

        Parameters:
            image (numpy.ndarray): The input binary image.

        Returns:
            list: A list of contours found in the image sorted from largest to smallest area.
        """
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        return contours_sorted
    
    
    def get_ignore_list(self, directory):
        """
        Read and parse a file containing locations and return a list of parsed locations.

        Parameters:
            directory (str): Path to the file containing locations.

        Returns:
            list: A list of parsed locations.
        """       
        locs = []
        with open(directory, "r") as file:
            lines = file.readlines()
            for line in lines:
                locs.append(ast.literal_eval(line.replace("\n", "")))
        return locs
    

    def is_overlapping(self, new_box, existing_boxes, iou_threshold=None, epsilon=1e-5):
        """
        Checks if a new bounding box overlaps with any of the existing bounding boxes.

        Parameters:
            new_box (tuple): Bounding box coordinates (x1, y1, x2, y2) of the new object.
            existing_boxes (list): List of existing bounding boxes.
            iou_threshold (float): IoU threshold for considering overlap.
            epsilon:    (float) Small value to prevent division by zero. Default=1e-5

        Returns:
            bool: True if the new box overlaps with any existing box, False otherwise.
        """
        #Determine IoU threshold
        iou_thres = self.IOU_THRESHOLD if iou_threshold is None else iou_threshold

        for existing_box in existing_boxes:
            # COORDINATES OF THE INTERSECTION BOX
            x1 = max(new_box[0], existing_box[0])
            y1 = max(new_box[1], existing_box[1])
            x2 = min(new_box[2], existing_box[2])
            y2 = min(new_box[3], existing_box[3])

            # AREA OF OVERLAP - Area where the boxes intersect
            width = x2 - x1
            height = y2 - y1
            
            # handle case where there is NO overlap
            if (width < 0) or (height < 0):
                return False
            area_overlap = width * height

            # COMBINED AREA
            area_a = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
            area_b = (existing_box[2] - existing_box[0]) * (existing_box[3] - existing_box[1])
            area_combined = area_a + area_b - area_overlap

            # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
            iou = area_overlap / (area_combined + epsilon)

            # Determine if it is overlapping
            if iou > iou_thres:
                if (area_a / area_b) > 1.3 :
                    return False
                return True
        return False
    
    
    def filter_ignored_objects(self, detected_objects):
        """
        Filters out detected objects that overlap with ignored regions.

        Parameters:
            detected_objects (list): List of detected object bounding boxes.

        Returns:
            list: List of filtered detected object bounding boxes.
        """
        filtered_objects = []
        for obj_box in detected_objects:
            is_ignored = False
            if self.is_overlapping(obj_box, self.ignores, iou_threshold=0.2):
                is_ignored = True
            if not is_ignored:
                filtered_objects.append(obj_box)

        return filtered_objects
        
    
    def find_objects(self, frame, difference_mask):
        """
        Detects and highlights new objects in an image.

        Parameters:
            frame (numpy.ndarray): The input image.
            difference_mask (numpy.ndarray): The difference image between two images.

        Returns:
            numpy.ndarray: The image with rectangles drawn around the detected objects.
        """
        detected_objects = []
        contours = self.get_contours(difference_mask)
        
        # Filter overlapping contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.MIN_SIZE_THRESHOLD:
                x, y, w, h = cv2.boundingRect(contour)
                object_box = [x, y, x + w, y + h]
                if not self.is_overlapping(object_box, detected_objects, iou_threshold=0.015):
                    detected_objects.append(object_box)
    
        filtered_objects = self.filter_ignored_objects(detected_objects) 

        for box in filtered_objects:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame, filtered_objects