# ImageAPI-to-csv
## Description
- Takes in videos/images from `/inputs` directory and extracts 5 frames per second for each video.
- Runs frames through API calls.
- Process and saves responses into `.csv` format in `/outputs/results.csv`

## Quick Start
To detect with custom weights:
1. Make sure to have your API url and key ready.
2. Execute the following command: `cp example.config.json config.json`
3. Fill in respective details into `config.json`
4. `python main.py --api_capability` 

_Note: Either `--lisa` or `--mobius` must be specified, but not both._

## Available API Capabilities
- LISA
- Mobius
- TorusVision (Not yet implemented)

Capabilities are appended at the end of the execution command. (e.g. `python main.py --lisa`) Only one capability must be chosen.

### LISA
`results.csv` is in the following format:
| Filename | Check1 | Check2 | ... | Width | Height | Passed |
| -------- | ------ | ------ | --- | ----- | ------ | ------ |
| sample.csv | 1 | 0 | ... | 1024 | 1024 | No |
- Checks can be omitted and included by commenting out the respective check in `main.py` at `LISA_CHECK_TYPES`

Photos which pass the checks are copied into `/outputs/success`.
Photos which fail the checks are copied into `/outputs/fail`.

_Note: LISA assumes all files in `/inputs` directory are photos._

### Mobius
`results.csv` is in the following format:
| Filename | Label1 | Label2 | ... | LabelN |
| -------- | ------ | ------ | --- | ------ |
| sample.csv | True |        | ... | True |
- If LabelX was not detected in a file, LabelX will be left blank.
- If LabelX was detected in a file, LabelX will be `True`

Any frames which successfully detect objects are copied into `/outputs/filename` with their detections. 

In cases where no objects are detected in all inputs, only the filename will show up in `results.csv`.

#### Optional Arguments
`--threshold your_threshold`: Set threshold for confidence. (Not yet implemented)
`--fps your_fps`: Set number of frames extracted per second (Not yet implemented)

### TorusVision
Not yet implemented.
