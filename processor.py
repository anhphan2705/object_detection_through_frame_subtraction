import cv2
import time
import numpy as np

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
        """
        Track objects in a frame based on a difference mask.

        Parameters:
            frame (numpy.ndarray): The input frame.
            difference_mask (numpy.ndarray): The difference mask.

        Returns:
            Tuple[numpy.ndarray, list]: A tuple containing the frame with objects tracked and a list of detected objects.
        """
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
        
        
    def get_progress(self, time_start, frame_processed, total_frame):
        """
        Display progress information during frame processing.

        Parameters:
            time_start (float): The start time of frame processing.
            frame_processed (int): The number of frames processed.
            total_frame (int): The total number of frames to be processed.
        """
        elapsed = time.time() - time_start
        process_fps = round((frame_processed + 1) / elapsed, 1)
        expect_time = (total_frame + 1 - frame_processed) // process_fps
        print(
            f"\rProcessing frame {frame_processed}/{int(total_frame+1)} in {(elapsed // 60):.0f}m{(elapsed % 60):.0f}s at speed {process_fps} FPS. Expect done in {(expect_time // 60):.0f}m {(expect_time % 60):.0f}s",
            end=" ",
            flush=True,
        )
    
    
    def set_label(self, frame, box_pos, key, time_still, thickness=2, color=(51, 153, 255), font_size=0.7):
        """
        Annotates an image on a bounding box with relevant information such as ID and stationary time.
        
        Args:
            frame (numpy.ndarray): The input image or frame to annotate.
            box_pos (tuple): A tuple representing the position of the bounding box
                in the format (x1, y1, x2, y2), where (x1, y1) is the top-left corner
                and (x2, y2) is the bottom-right corner of the bounding box.
            key (str): An identifier or label associated with the bounding box.
            time_still (float): The time duration (in seconds) indicating how long
                the object in the bounding box has been still.
            thickness (int, optional): The thickness of the text. Default is 2.
            color (tuple, optional): The color of the text in BGR format. Default is (51, 153, 255),
                which corresponds to a shade of orange.
            font_size (float, optional): The font size of the text. Default is 0.7.

        Returns:
            numpy.ndarray: An annotated image with bounding box and text information.

        Note:
            This function uses the OpenCV library to draw text on the input frame.
        """
        x1, y1, x2, y2 = box_pos
        # Draw the object identifier (key)
        frame = cv2.putText(frame,
                            f'ID={key}',
                            org=(x1 + 2, y2 - 4),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_size,
                            color=color,
                            thickness=thickness,
                            lineType=cv2.LINE_AA)
        
        # Draw the time duration in minutes and seconds
        frame = cv2.putText(frame,
                            f'{int(time_still // 60):d}m {int(time_still % 60):d}s',
                            org=(x1 + 2, y1 - 4),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_size,
                            color=color,
                            thickness=thickness,
                            lineType=cv2.LINE_AA)
        return frame

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
            self.get_progress(time_start=since, frame_processed=self.frame_count, total_frame=self.TOTAL_FRAME)

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
        