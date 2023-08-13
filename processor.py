import cv2
import time
import numpy as np

class VideoProcessor:
    
    # Constant
    DEFAULT_WHITE_THRESHOLD = 225
    DEFAULT_BLACK_THRESHOLD = 50
    DEFAULT_OUT_PATH = './output'
    
    
    def __init__(self, source_path: str, out_path: str, preprocess: object, tracking: object, white_threshold=DEFAULT_WHITE_THRESHOLD, black_threshold=DEFAULT_BLACK_THRESHOLD):
        """
        Initializes a VideoProcessor object for processing videos apply stationary object detection.

        Parameters:
            source_path (str): The path to the input video file.
            out_path (str): The directory where output files will be saved.
            preprocess (object): The preprocessing techniques to be applied on each frame.
            tracking (object): The tracking methods to be used for object detection.
            white_threshold (int): Threshold for considering differences as white.
            black_threshold (int): Threshold for considering differences as black.
        """
        self.video = cv2.VideoCapture(source_path)
        self.OUT_PATH = out_path
        self.FPS = self.video.get(cv2.CAP_PROP_FPS)
        self.TOTAL_FRAME = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.FRAME_WIDTH = int(self.video.get(3))
        self.FRAME_HEIGHT = int(self.video.get(4))
        self.preprocess = preprocess
        self.tracking = tracking
        self.black_threshold = black_threshold
        self.white_threshold = white_threshold
        self.frame_count = 0
        self.temp_stationary = []
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
        difference_mask = cv2.dilate(difference_mask, (5, 5))
        
        return difference_mask


    def write_log(self, stationary_objects):
        """
        Write information about detected stationary objects to a log file.

        Parameters:
            log_path (str, optional): The path to the log file. Default is './output/log.txt'.
        """
        log = open(self.OUT_PATH + '/log.txt', 'w')
        textLines = []
        textLines.append('############################## Detected New Stationary Objects ##############################\n')
        for key, value in stationary_objects.items():
            [status, start_frame, end_frame, position] = value
            duration = (end_frame - start_frame) // self.FPS
            text = f'[ID: {key}]    Existing in frame: {status} | Time existed: {(duration // 60):.0f}m{(duration % 60):.0f}s | Position: ({position[0]}, {position[1]}) ({position[2]}, {position[3]})\n'
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
        
        
    def set_background_frame(self, frame):
        """
        Set the input frame as the background frame.

        Parameters:
            frame (numpy.ndarray): The frame to be set as the background frame.
        """
        self.background_frame = frame
    
    
    def process_video(self):
        """
        Process a video by iterating over frames, applying preprocessing, detecting objects,
        tracking stationary objects, updating logs, and displaying frames.

        The loop continues until all frames are processed or the user stops it.

        Note: You should have the appropriate values assigned to self.video, self.frame_count,
        self.TOTAL_FRAME, self.preprocess_frame, self.detect_differences, self.tracking,
        self.temp_stationary, self.write_log, and other necessary attributes.
        """
        # Start recording
        # fourcc = cv2.VideoWriter_fourcc(*"avc1")
        # result_video = cv2.VideoWriter(self.DEFAULT_OUT_PATH + '/result_video.mp4', fourcc, 30.0, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        
        since = time.time()
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            
            self.frame_count += 1
            self.get_progress(time_start=since, frame_processed=self.frame_count, total_frame=self.TOTAL_FRAME)

            # Preprocess Frame
            processed_frame = self.preprocess_frame(frame)
            
            # Select the first frame to compare to
            # IF YOU WANT TO SELECT MORE FRAME TO COMPARE THROUGH OUT THE VIDEO DO IT HERE
            if self.frame_count == 1:
                self.set_background_frame(processed_frame)
            
            # Doing background subtraction
            difference_mask = self.detect_differences(self.background_frame, processed_frame)
            
            # Tracking objects
            tracked_frame, object_detected = self.tracking.find_objects(frame, difference_mask)

            # Track stationary objects
            if self.frame_count % self.tracking.DEFAULT_TRACK_RATE == 0:
                prev_temp_stationary = self.temp_stationary
                
                self.temp_stationary = self.tracking.find_potential_stationary(
                    prev_objects=self.temp_stationary, 
                    new_objects=object_detected
                )
                
                self.tracking.update_stationary_objects(
                    frame_count=self.frame_count, 
                    prev_temp_stationary=prev_temp_stationary, 
                    temp_stationary=self.temp_stationary, 
                    frame=frame
                )
                
                # Update log
                self.write_log(self.tracking.get_stationary_objects())
            
            # Set label to frame
            result_frame = self.tracking.set_label(tracked_frame)
            
            # Write frame
            # result_video.write(result_frame)
            
            # Show frame
            cv2.imshow("Video", tracked_frame)
            if cv2.waitKey(1) & 0xFF == ord("c"):
                break
            
        self.video.release()
        # result_video.release()
        cv2.destroyAllWindows()
        