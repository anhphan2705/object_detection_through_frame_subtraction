import argparse
from processor import VideoProcessor
from preprocess import PreProcessImage
from track import Tracking


def main():
    
    parser = argparse.ArgumentParser(description='This program tracks differences and detects stationary objects in a video.')
    parser.add_argument('-i', '--input', type=str, help='Path to the input video.', required=True)
    parser.add_argument('-m', '--mask', type=str, help='Path to a mask image.', default=None)
    parser.add_argument('--ignore', type=str, help='Path to a list of positions of boxes in the frame that you want the program to ignore. Each line contains 1 box position. Example format: [x1, y1, x2, y2]', default=None)
    parser.add_argument('--iou', type=float, help='IOU threshold for object matching.', default=Tracking.DEFAULT_IOU_THRESHOLD)
    parser.add_argument('--min-size', type=int, help='Minimun area of the contour box to be recorded as an object.', default=Tracking.DEFAULT_MIN_SIZE)
    parser.add_argument('--track-rate', type=int, help='Tracking rate for stationary object detection.', default=Tracking.DEFAULT_TRACK_RATE)
    parser.add_argument('--white', type=int, help='Determine the minimum value to be white pixel otherwise will be turned black.', default=VideoProcessor.DEFAULT_WHITE_THRESHOLD)
    parser.add_argument('--black', type=int, help='Determine the minimum value to be black pixel otherwise will be turned white.', default=VideoProcessor.DEFAULT_BLACK_THRESHOLD)
    parser.add_argument("--gray", type=bool, help="(bool) False to turn off grayscale in preprocessing, True otherwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contrast", type=bool, help="(bool) True to turn on auto contrast in preprocessing, False otherwise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blur", type=bool, help="(bool) True to turn on blurring in preprocessing, False otherwise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--edge", type=bool, help="(bool) True to turn on finding edge in preprocessing, False otherwise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save", type=bool, help="(bool) True to turn on saving result video in preprocessing, False otherwise", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    
    
    pre_process = PreProcessImage(
        gray=args.gray, 
        contrast=args.contrast, 
        blur=args.blur, 
        edge=args.edge, 
        mask_path=args.mask
        )
    
    tracking = Tracking(
        track_rate=args.track_rate,
        ignore_path=args.ignore,
        min_size=args.min_size,
        iou_threshold=args.iou
    )
    
    video_processor = VideoProcessor(
        source_path=args.input, 
        preprocess=pre_process, 
        tracking=tracking,
        white_threshold=args.white,
        black_threshold=args.black
        )
    
    video_processor.process_video()


if __name__ == "__main__":
    main()