import cv2
from utilities import get_iou, get_ignore_list, get_contours

class Tracking:
    
    DEFAULT_TRACK_RATE = 30
    DEFAULT_MIN_SIZE = 750
    DEFAULT_IOU_THRESHOLD = 0.87
    
    
    def __init__(self, track_rate=DEFAULT_TRACK_RATE, ignore_path=None, min_size=DEFAULT_MIN_SIZE, iou_threshold=DEFAULT_IOU_THRESHOLD):
        self.TRACKRATE = track_rate
        self.ignores = get_ignore_list(ignore_path) if ignore_path else None
        self.MIN_SIZE_THRESHOLD = min_size
        self.IOU_THRESHOLD = iou_threshold
        
        
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
            if self.ignores is not None:
                for ignore_box in self.ignores:
                    if self.is_overlapping(obj_box, ignore_box):
                        is_ignored = True
                        break
            if not is_ignored:
                filtered_objects.append(obj_box)

        return filtered_objects
        
        
        
    # def find_objects(self, frame, difference_mask):
    #     """
    #     Draws rectangles around the new objects in an image.

    #     Parameters:
    #         frame (numpy.ndarray): The input image.
    #         difference_mask (numpy.ndarray): The difference image between two images.

    #     Returns:
    #         numpy.ndarray: The image with rectangles drawn around the differences.
    #     """
    #     detected_objects = []
    #     contours = get_contours(difference_mask)
        
    #     for c in contours:
    #         area = cv2.contourArea(c)
    #         if area > self.MIN_SIZE_THRESHOLD:
    #             x, y, w, h = cv2.boundingRect(c)
    #             new_box = [x, y, x + w, y + h]
    #             if len(detected_objects) > 0:
    #                 valid = True
    #                 for box in detected_objects:
    #                     iou = get_iou(new_box, box)
    #                     if iou > 0.02:
    #                         x1, y1, x2, y2 = new_box
    #                         area_new = (x2 - x1) * (y2 - y1)
    #                         x1, y1, x2, y2 = box
    #                         area_box = (x2 - x1) * (y2 - y1)
    #                         if area_new > area_box:
    #                             detected_objects.append(new_box)
    #                             if box in detected_objects:
    #                                 detected_objects.remove(box)
    #                         valid = False
    #                         break
    #                 if valid:
    #                     detected_objects.append(new_box)
    #             else:
    #                 detected_objects.append(new_box)
                    
    #     # Remove ignored object from new detected list
    #     for pos in detected_objects.copy():
    #         if self.ignores:
    #             for ignore in self.ignores:
    #                 iou = get_iou(ignore, pos)
    #                 if iou > self.IOU_THRESHOLD:
    #                     if pos in detected_objects:
    #                         detected_objects.remove(pos)
                            
    #     # Highlight the detected objects
    #     for box in detected_objects:
    #         x1, y1, x2, y2 = box
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #     return frame, detected_objects
    
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
        contours = get_contours(difference_mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.MIN_SIZE_THRESHOLD:
                x, y, w, h = cv2.boundingRect(contour)
                object_box = [x, y, x + w, y + h]
                if not self.is_overlapping(object_box, detected_objects, iou_threshold=0.2):
                    detected_objects.append(object_box)

        filtered_objects = self.filter_ignored_objects(detected_objects)

        for box in filtered_objects:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame, filtered_objects