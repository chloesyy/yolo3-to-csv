# yolo3-to-csv
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
