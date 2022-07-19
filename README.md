# yolo3-to-csv
## Description
- Takes in videos/images from `/data` directory and extracts 5 frames per second for each video.
- Runs frames through local yolo model.
- Returns true (yes) if object is detected in **any one frame**, otherwise returns false (no).
- Generates csv file (`results.csv`) in the following format:

    | Filename   | Is object detected? |
    | ---------- | ----------------- |
    | sample.mp4 | yes |


## Quick Start
To detect with custom weights:
1. Make sure to have your model (.h5) file ready.
2. `python main.py --model path/to/model.h5`

_Note: `--model` must be specified._

## Additional Info

To change classes: `--classes path/to/classes.txt`

To change anchors: `--anchors path/to/anchors.txt`

To add more data to be detected: add them into `data/` directory.
Input directory can be changed with `--input path/to/data/directory`

To change threshold for detection:  `--threshold 0.9`

## Results
Results are saved into `results.csv`

## Future Steps
1. To remove local model file requirement, replaced with api calls to models
2. To generalise to models other than yolov3 (e.g. FasterRCNN)
