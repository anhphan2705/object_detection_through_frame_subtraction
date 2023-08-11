import cv2
import time
import numpy as np
from utilities import get_progress

class VideoProcessor:
    
    # Constant
    DEFAULT_WHITE_THRESHOLD = 225
    DEFAULT_BLACK_THRESHOLD = 50
    
    
    def __init__(self, source_path, preprocess, tracking, white_threshold=DEFAULT_WHITE_THRESHOLD, black_threshold=DEFAULT_BLACK_THRESHOLD):
        self.video = cv2.VideoCapture(source_path)
        self.FPS = self.video.get(cv2.CAP_PROP_FPS)
        self.TOTAL_FRAME = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        # self.FRAME_WIDTH = int(self.video.get(3))
        # self.FRAME_HEIGHT = int(self.video.get(4))
        # self.FRAME_SIZE = (self.FRAME_WIDTH, self.FRAME_HEIGHT)
        self.preprocess = preprocess
        self.tracking = tracking
        self.black_threshold = black_threshold
        self.white_threshold = white_threshold
        self.frame_count = 0
        self.still_objects = {}
        self.background_frame = np.array([])


    def preprocess_frame(self, frame):
        """
        Preprocess a single frame using the specified preprocessing techniques.

        Parameters:
            frame (numpy.ndarray): The input frame to be preprocessed.

        Returns:
            numpy.ndarray: The preprocessed frame.
        """
        return self.preprocess.get_preprocess(frame)


    def detect_differences(self, background_frame, curr_frame):
        """
        Detect differences between two frames and create a difference mask.

        Parameters:
            background_frame (numpy.ndarray): The background frame.
            curr_frame (numpy.ndarray): The current frame to compare.

        Returns:
            numpy.ndarray: The difference mask highlighting changes between the frames.
        """
        difference_mask = np.subtract(curr_frame, background_frame)
        
        # Convert white differences to black
        white_loc = np.where(difference_mask > self.white_threshold)
        difference_mask[white_loc] = 0
        
        # Convert black-ish differences to black
        black_loc = np.where(difference_mask < self.black_threshold)
        difference_mask[black_loc] = 0
        
        # Convert object differences to white
        obj_loc = np.where((difference_mask > self.black_threshold) & (difference_mask < self.white_threshold))
        difference_mask[obj_loc] = 255
        
        # Dilate the result to enhance differences
        difference_mask = cv2.dilate(difference_mask, (7, 7))
        
        return difference_mask


    def track_objects(self, frame, difference_mask):
        detected_frame, detected_objects = self.tracking.find_objects(frame, difference_mask)
        return detected_frame, detected_objects


    def find_stationary_objects(self, new_objects):
        # Implement logic to find stationary objects
        stationary_objects = []
        return stationary_objects


    def write_log(self):
        """
        Write information about detected stationary objects to a log file.

        Parameters:
            log_path (str, optional): The path to the log file. Default is './output/log.txt'.
        """
        log = open('./output/log.txt', 'w')
        textLines = []
        textLines.append('############################## Detected New Stationary Objects ##############################\n')
        for key, value in self.still_objects.items():
            [status, start_frame, end_frame, pos] = value
            time_still = time_still = (end_frame - start_frame) // self.FPS
            text = f'[ID: {key}]    Existing in frame: {status} | Time existed: {(time_still // 60):.0f}m{(time_still % 60):.0f}s | Position: ({pos[0]}, {pos[1]}) ({pos[2]}, {pos[3]})\n'
            textLines.append(text)
        log.writelines(textLines) 
        log.close()

    # def process_video(self):
    #     while True:
    #         ret, frame = self.video.read()
    #         if not ret:
    #             break

    #         preprocessed_frame = self.preprocess_frame(frame)
    #         if self.frame_count == 0:
    #             prev_frame = preprocessed_frame
    #         difference_mask = self.detect_differences(prev_frame, preprocessed_frame)
    #         tracked_frame = self.track_objects(preprocessed_frame, difference_mask)

    #         stationary_objects = self.find_stationary_objects(new_objects)
    #         self.still_objects = stationary_objects

    #         self.write_log()

    #         self.frame_count += 1
    #         prev_frame = preprocessed_frame

    #     self.video.release()
    
    
    def process_video(self):
        
        since = time.time()
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            
            self.frame_count += 1
            get_progress(time_start=since, frame_processed=self.frame_count, total_frame=self.TOTAL_FRAME)

            # Preprocess Frame
            processed_frame = self.preprocess_frame(frame)
            if self.frame_count == 1:
                self.background_frame = processed_frame
            
            # Doing background subtraction
            difference_mask = self.detect_differences(self.background_frame, processed_frame)
            
            # Tracking objects
            tracked_frame, object_detected = self.track_objects(frame, difference_mask)
                
                
            cv2.imshow("Video", tracked_frame)
            if cv2.waitKey(1) & 0xFF == ord("c"):
                break
            
        self.video.release()
        cv2.destroyAllWindows()
        