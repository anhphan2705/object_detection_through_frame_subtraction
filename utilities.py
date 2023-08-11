import cv2
import ast
import os
import time

def get_contours(image):
    """
    Finds contours in a binary image.

    Parameters:
        image (numpy.ndarray): The input binary image.

    Returns:
        list: A list of contours found in the image.
    """
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


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


def get_ignore_list(directory):
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


def get_progress(time_start, frame_processed, total_frame):
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
    
    
def get_area(x1, y1, x2, y2):
    """
    Calculate the area of a rectangle given its coordinates.

    Parameters:
        x1 (int): The x-coordinate of the top-left corner.
        y1 (int): The y-coordinate of the top-left corner.
        x2 (int): The x-coordinate of the bottom-right corner.
        y2 (int): The y-coordinate of the bottom-right corner.

    Returns:
        int: The area of the rectangle.
    """
    return (x2 - x1) * (y2 - y1)

    
def set_ndarray(image):
    """
    Converts an image with any data type to a format readable by OpenCV (BGR, uint8).

    Parameters:
        image (SSIM image): The input image with any data type

    Returns:
        numpy.ndarray: The image converted to OpenCV readable format (BGR, uint8).
    """
    image = (image * 255).astype("uint8")
    return image


def set_label(frame, box_pos, key, time_still, thickness=2, color=(51, 153, 255), font_size=0.7):
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

def create_dir():
    os.makedirs("./output/saved_result")
    os.makedirs