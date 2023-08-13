import cv2
import ast

class Tracking:
    
    # Constant
    DEFAULT_TRACK_RATE = 30
    DEFAULT_MIN_SIZE = 750
    DEFAULT_IOU_THRESHOLD = 0.87
    
    
    def __init__(self, track_rate=DEFAULT_TRACK_RATE, ignore_path=None, min_size=DEFAULT_MIN_SIZE, iou_threshold=DEFAULT_IOU_THRESHOLD):
        """
        Initializes a Tracking object for tracking objects and detect stationary objects.

        Parameters:
            track_rate (int, optional): The rate at which tracking is performed (in frames). Default is 30.
            ignore_path (str, optional): Path to a file containing ignored regions for tracking.
            min_size (int, optional): Minimum size threshold for considering detected objects. Default is 750.
            iou_threshold (float, optional): Intersection over Union threshold for object overlap. Default is 0.87.
        """
        self.TRACKRATE = track_rate
        self.ignores = self.get_ignore_list(ignore_path) if ignore_path else None
        self.MIN_SIZE_THRESHOLD = min_size
        self.IOU_THRESHOLD = iou_threshold
        self.stationary_objects = {}
        

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
    
    
    def get_stationary_objects(self):
        """
        Get the dictionary of stationary objects.

        Returns:
            dict: The dictionary containing stationary objects.
        """
        return self.stationary_objects
    

    def set_label(self, frame, thickness=2, color=(51, 153, 255), font_size=0.7):
        """ 
        Annotates an image on a bounding box with relevant information such as ID and stationary time.
        
        Args:
            frame (numpy.ndarray): The input image or frame to annotate.
            thickness (int, optional): The thickness of the text. Default is 2.
            color (tuple, optional): The color of the text in BGR format. Default is orange (51, 153, 255)
            font_size (float, optional): The font size of the text. Default is 0.7.

        Returns:
            numpy.ndarray: An annotated image with bounding box and text information.

        Note:
            This function uses the OpenCV library to draw text on the input frame.
        """
        for key, value in self.stationary_objects.items():
            [active, start_frame, end_frame, position] = value
            if active:
                duration = (end_frame - start_frame) // self.DEFAULT_TRACK_RATE
                x1, y1, x2, y2 = position
                
                # Draw the object identifier (key)
                frame = cv2.putText(
                    frame,
                    f'ID={key}',
                    org=(x1 + 2, y2 - 4),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_size,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA
                )
                
                # Draw the time duration in minutes and seconds
                frame = cv2.putText(
                    frame,
                    f'{int(duration // 60):d}m {int(duration % 60):d}s',
                    org=(x1 + 2, y1 - 4),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_size,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA
                )
                
        return frame
    

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
        
        # Case where nothing to compare to
        if not existing_boxes:
            return False
        
        # Comparing
        for existing_box in existing_boxes:
            # Coordinates of the intersection box
            x1 = max(new_box[0], existing_box[0])
            y1 = max(new_box[1], existing_box[1])
            x2 = min(new_box[2], existing_box[2])
            y2 = min(new_box[3], existing_box[3])

            # AREA OF OVERLAP - Area where the boxes intersect
            width = x2 - x1
            height = y2 - y1
            
            # Handle case where there is NO overlap
            if (width < 0) or (height < 0):
                continue

            # COMBINED AREA
            area_overlap = width * height
            area_a = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
            area_b = (existing_box[2] - existing_box[0]) * (existing_box[3] - existing_box[1])
            area_combined = area_a + area_b - area_overlap

            # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
            iou = area_overlap / (area_combined + epsilon)

            # Determine if it is overlapping
            if iou > iou_thres:
                # if (area_a / area_b) > 1.3 :
                #     continue
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
    
    
    def reset_stationary_status(self):
        for key, value in self.stationary_objects.items():
            value[0] = False
        
    
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
        tracked_frame = frame.copy()
        contours = self.get_contours(difference_mask)
        
        # Filter overlapping contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.MIN_SIZE_THRESHOLD:
                x, y, w, h = cv2.boundingRect(contour)
                object_box = [x, y, x + w, y + h]
                if not self.is_overlapping(object_box, detected_objects, iou_threshold=0.02):
                    detected_objects.append(object_box)
    
        filtered_objects = self.filter_ignored_objects(detected_objects) 

        for box in filtered_objects:
            x1, y1, x2, y2 = box
            cv2.rectangle(tracked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return tracked_frame, filtered_objects
    
    
    def find_potential_stationary(self, prev_objects, new_objects):
        """
        Find temporary stationary objects by comparing new objects with previous objects.

        Parameters:
            prev_objects (list): List of previous objects.
            new_objects (list): List of new objects.

        Returns:
            list: List of temporary stationary objects.
        """
        stationary_objects = []
        if len(prev_objects) == 0:
            return new_objects
        else:
            for obj in new_objects:
                if self.is_overlapping(obj, prev_objects):
                    stationary_objects.append(obj)
        if len(stationary_objects) == 0:
            return new_objects
        else:
            return stationary_objects
        
        
    def update_existing_stationary_objects(self, stationary_potentials, frame_count):
        """
        Updates the status of existing stationary objects.

        Parameters:
            stationary_potentials (list): List of positions of potentially stationary objects.
        """
        
        for stationary_potential in stationary_potentials.copy():
            for key, value in self.stationary_objects.items():
                prev_stationary = value[3]
                if self.is_overlapping(prev_stationary, [stationary_potential]):
                    value[0] = True
                    value[2] = frame_count
                    value[3] = stationary_potential
                    if stationary_potential in stationary_potentials:
                        stationary_potentials.remove(stationary_potential)
                    break


    def add_new_stationary_object(self, stationary_potentials, frame_count, frame):
        """
        Adds new stationary objects to the dictionary.

        Parameters:
            stationary_potentials (list): List of positions of potentially stationary objects.
            frame_count (int): The current frame count.
            frame (numpy.ndarray): The current frame.
        """
        for position in stationary_potentials:
            index = len(self.stationary_objects)
            self.stationary_objects[index] = [True, frame_count - 3 * self.DEFAULT_TRACK_RATE, frame_count, position]
            self.save_still_image(frame, position, index)


    def save_still_image(self, frame, position, index):
        """
        Saves a cropped image of a stationary object.

        Parameters:
            frame (numpy.ndarray): The current frame.
            position (tuple): Bounding box coordinates (x1, y1, x2, y2) of the stationary object.
            index (int): Index of the stationary object.
        """
        x1, y1, x2, y2 = position
        cropped_image = frame[y1:y2, x1:x2]
        image_path = f"./output/id_{index}.jpg"
        cv2.imwrite(image_path, cropped_image)
        
        
    def update_stationary_objects(self, frame_count, prev_temp_stationary, temp_stationary, frame):
        """
        Updates the list of stationary objects based on frame differences.

        Parameters:
            frame_count (int): The current frame count.
            prev_temp_stationary (list): List of positions of stationary objects in the previous frame.
            temp_stationary (list): List of positions of objects in the current frame.
            frame (numpy.ndarray): The current frame.

        Returns:
            dict: Updated dictionary of stationary object information.
        """
        stationary_potentials = []
        self.reset_stationary_status()
        
        if prev_temp_stationary or temp_stationary:
            for new_object in temp_stationary:
                if self.is_overlapping(new_object, prev_temp_stationary):
                    stationary_potentials.append(new_object)
                    
        if self.stationary_objects is None:
            for i, position in enumerate(stationary_potentials):
                self.stationary_objects[i] = [True, frame_count - 3 * self.DEFAULT_TRACK_RATE, frame_count, position]
                self.save_still_image(frame, position, i)
        else:
            self.update_existing_stationary_objects(stationary_potentials, frame_count)
            self.add_new_stationary_object(stationary_potentials, frame_count, frame)

        return self.stationary_objects