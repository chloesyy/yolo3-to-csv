# yolo3-to-csv
## Quick Start
To detect using default yolov3 model:
`python main.py`

To detect with custom weights and classes:
`python main.py --model path/to/model.h5 --classes path/to/classes.txt`

To change anchors:
`--anchors path/to/anchors.txt`

To add more data to be detected: add them into `data/` directory.
Input directory can be changed with `--input path/to/data/directory`

To change threshold for detection: 
`--threshold 0.9`

## Results
Results are saved into `results.csv`
