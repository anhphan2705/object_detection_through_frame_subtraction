# Video Object Detection and Tracking

![License](https://img.shields.io/badge/license-Apache%202.0-blue)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Command-line Arguments](#command-line-arguments)
- [Output](#output)
- [License](#license)

## Introduction

The Video Object Detection and Tracking Script is a Python tool designed for processing videos or sequences of images to detect and track changes between frames. It employs various image processing techniques and algorithms to identify differences, track objects, and manage stationary objects. This script is useful in scenarios where you need to analyze and visualize motion patterns or stationary periods within a video stream.

## Features

- Detects and tracks differences between frames in a video or sequence of images.
- Marks detected objects with rectangles and tracks their movements.
- Detects and logs stationary objects based on user-defined thresholds.
- Provides command-line interface for customization and control.
- Generates output videos and log files to visualize and analyze results.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- OpenCV (cv2)
- NumPy (np)
- scikit-image (skimage)

### Installation

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

2. Place your input video or image sequence in the `./data/videos/` directory.

3. Run the script using the following command:

   ```bash
   python video_object_detection_tracking.py -i ./data/videos/your_video.mp4
   ```

4. Customize the behavior using the available command-line arguments (see [Command-line Arguments](#command-line-arguments)).

5. The processed video and log files will be saved in the `./output/` directory.

## Command-line Arguments

- `-i`, `--input`: Path to the input video or image sequence. (required)
- `-o`, `--output`: Path to the directory where outputs will be saved.
- `-m`, `--mask`: Path to a mask image to exclude specific areas from detection.
- `-u`, `--iou`: Intersection over Union (IoU) threshold for matching object positions.
- `-r`, `--refresh`: Number of frames between stationary object checks.
- `-g`, `--ignore`: Path to the file containing ignored box locations (one per line).
- `-t`, `--track`: Enable tracking of detected objects (True or False).
- `-b`, `--bsub`: Enable background subtraction (True or False).
- `-k`, `--outmask`: Save masked video (True or False).

## Output

The script generates the following output:

- `./output/output.mp4`: Processed video with detected and tracked objects.
- `./output/log.txt`: Log file containing information about detected stationary objects.
- `./output/snapshot.jpg`: Snapshot of all the objects detected as new stationary object

## License

This project is licensed under the [Apache License 2.0](LICENSE).