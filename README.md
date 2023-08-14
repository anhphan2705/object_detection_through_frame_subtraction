# Object Detection Camera Feed 

![License](https://img.shields.io/badge/license-Apache%202.0-blue)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Command-line Options](#command-line-options)
- [Run Examples](#run-examples)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

The  is a Python tool designed for processing videos or sequences of images to detect and track changes between frames. It employs various image processing techniques and algorithms to identify differences, track objects, and manage stationary objects. This script is useful in scenarios where you need to analyze and visualize motion patterns or stationary periods within a video stream.

## Features

- Detects and tracks differences between frames in a video or sequence of images.
- Marks detected objects with rectangles and tracks their movements.
- Detects and logs stationary objects based on user-defined thresholds.
- Provides command-line interface for customization and control.
- Generates output videos and log files to visualize and analyze results.

## Prerequisites

- Python 3.6 or higher
- OpenCV (cv2)
- NumPy (np)
- scikit-image (skimage)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/anhphan2705/Object-Detection-Camera-Feed.git
   ```

2. Install the required dependencies:

   ```bash
   pip install opencv-python numpy scikit-image
   ```

## Usage

To use the Video Object Detection and Tracking Script, follow these steps:

1. Ensure that you have Python and the required libraries installed (see Prerequisites).

2. Arrange your input folder and output folder accordingly. Recommended folder arrangement:

```
Object-Detection-Camera-Feed/
   main.py
   preprocess.py
   processor.py
   tracker.py
   ignore.txt
   data/
      videos/
         vid.mp4
         ...
      masks/
         mask.jpg
         ...
   output/
      ...
```

3. Modify the `ingore.txt` file as you wish. 

- This text file will ignore the box location that you want. 
- Keep in mind that it will only ignore objects that have matching or bigger `iou` with the boxes inside `ingore.txt`.
- Format is one `[x1, y1, x2, y2]` per line.

4. Change directory to the project folder then run the program using the following command:

   ```bash
   python main.py -i your_video.mp4
   ```

5. Customize the program's behavior using the available command-line arguments (see [Command-line Arguments](#command-line-arguments)).

6. Customize preprocess variables accordingly to your input video by setting `--white` and `--black` threshold to optimize the result.

7. The processed video and log files will be saved in the `./output/` directory or the path you provided

## Command-line Options

- `-i`, `--input`: Path to the input video or image sequence. (required)
- `-o`, `--output`: Path to the directory where outputs will be saved.
- `-m`, `--mask`: Path to a mask image to exclude specific areas from detection.
- `-g`, `--ignore`: Path to a list of positions of boxes that will be ignored.
- `--iou`: Intersection over Union (IoU) threshold for matching object positions.
- `--min-size`: Minimun area of the contour box to be recorded as an object.
- `--track-rate`: Number of frames between stationary object checks.
- `--white`: Set minimum value (from 0 to 255) to be white pixel otherwise will be turned black.
- `--black`: Set the minimum value (from 0 to 255) to be black pixel otherwise will be turned white.
- `--gray`, `--no-gray`: Turn on or off grayscale convertion in preprocessing.
- `--contrast`, `--no-contrast`: Turn on or off auto contrast in preprocessing.
- `--blur`, `--no-blur`: Turn on or off blurring in preprocessing.
- `--edge`, `--no-edge`: Turn on or off edge detection in preprocessing.
- `--save`, `--no-save`: Turn on or off saving result video.

## Run Examples

1. Basic:
  
```bash
python main.py -i './input_images' -o './output_images'
```

2. Run with modified stationary object tracking rate and higher iou:

```bash
python main.py -i './vid.mp4' -o './output' --track-rate 15 --iou 0.9
```

## Output

The script generates the following output to the path of your choice:

- `./output/result_video.mp4`: Processed video with detected and tracked objects.
- `./output/log.txt`: Log file containing information about detected stationary objects.
- `./output/id_[id_number].jpg`: Snapshot of all the objects detected as new stationary object

## Contributing

Contributions are welcome! If you have any suggestions or improvements for this code, please feel free to submit a pull request.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments

- OpenCV: https://opencv.org/
- The scikit-image library: https://scikit-image.org/