import os
import cv2
import csv
import shutil
import argparse
from yolo3.yolo import YOLO
from PIL import Image

frame_directory = 'frames'
data_directory = 'data/'
threshold = 0.8

def extract_frame(filename):
    print("extracting frames from", filename, "...")
    
    # get video
    vidcap = cv2.VideoCapture(filename)
    if os.path.exists(frame_directory):
        shutil.rmtree(frame_directory)
    os.mkdir(frame_directory)
    
    # extract frames from video
    success, image = vidcap.read()
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count*200))                              # extract 5 frames per second
        cv2.imwrite(os.path.join(frame_directory, "frame%d.jpg" % count), image)
        success, image = vidcap.read()
        count += 1
    print(count, "frames extracted from", filename)

# change inputs to api call (maybe dont need inputs anymore?)
def main(yolo):
    rows = []
    # loop through all files in data
    for filename in sorted(os.listdir(data_directory)):
        result = False
        extract_frame(os.path.join(data_directory, filename))
        
        # loop through all frames extracted 
        for frame_filename in sorted(os.listdir(frame_directory)):
            try:
                image = Image.open(os.path.join(frame_directory, frame_filename))
            except:
                print('Open error! Please try again.')
                continue
            else:
                """
                TODO: make api call to retrieve result
                """
                
                detection, scores, classes = yolo.detect_image(image)
                count = 0
                
                # count number of detections above specified threshold
                for i, c in reversed(list(enumerate(classes))):
                    score = scores[i]
                    if score > threshold:
                        count += 1
                    
                if count > 0:           # object above threshold is detected
                    detection.show()
                    result = True 
                    break               
        if os.path.exists(frame_directory):
            shutil.rmtree(frame_directory)
            
        # append to csv rows
        row = [filename]
        row.append("yes") if result else row.append("no")
        rows.append(row)
    
    yolo.close_session()

    # write to csv
    f = open('results.csv', 'w')
    writer = csv.writer(f)
    writer.writerows(rows)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    
    parser.add_argument(
        '--model', type=str, required=True,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )
    
    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )
    
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./data',
        help = "Video input directory"
    )
    
    parser.add_argument(
        "--threshold", nargs='?', type=str,required=False,default=threshold,
        help = "Threshold to count detected objects"
    )
    
    FLAGS = parser.parse_args()
    if "input" in FLAGS:
        data_directory = FLAGS.input
    if "threshold" in FLAGS:
        threshold = FLAGS.threshold
    main(YOLO(**vars(FLAGS)))
    
    
    
    