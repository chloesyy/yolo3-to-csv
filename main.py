import os
import cv2
import csv
import json
import base64
import shutil
import requests
import datetime
import argparse
import numpy as np
# from yolo3.yolo import YOLO
from PIL import Image, ImageDraw, ImageFont

FRAME_DIR = 'frames'
OUTPUT_DIR = 'outputs'
INPUT_DIR = 'inputs'
THRESHOLD = 0.8
API_CAP = None

# API urls
LISA_API = None
MOBIUS_API = None

# LISA check constants
LISA_CHECK_TYPES = [
        'background_check',
        'blur_check',
        'eye_close_check',
        'face_front_check',
        'face_presence_check',
        'file_type_check',
#        'frame_cover_eye_check',
        'gaze_check',
#        'hair_cover_eye_check',
#        'headcover_check',
#        'image_size_check',
#        'lighting_check',
#        'mouth_open_check',
#        'shoulder_alignment_check',
        'skin_specular_reflection_check',
#        'watermark_check'
    ]
CHECK_IMG_MINSIZE = False
MIN_WIDTH = 100
MIN_HEIGHT = 120

def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def extract_frame(filename):
    print("Extracting frames from", filename, "...")
    
    # get video
    vidcap = cv2.VideoCapture(filename)
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.mkdir(FRAME_DIR)
    
    # extract frames from video
    success, image = vidcap.read()
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count*200))                              # extract 5 frames per second
        cv2.imwrite(os.path.join(FRAME_DIR, "frame%d.jpg" % count), image)
        success, image = vidcap.read()
        count += 1
    print(count, "frame(s) extracted from", filename)

def draw_boxes(results, image):
    """
    Draw box based on api results; follwing mobius results
    """
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 500
    for result in results:
        label = result["Label"]
        for instance in result["Instances"]:
            box = instance["BoundingBox"]
            top = box["Top"] * image.size[1]
            bottom = top + box["Height"] * image.size[1]
            left = box["Left"] * image.size[0]
            right = left + box["Width"] * image.size[0]
            
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            
            # place label on top of box
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            
            # draw box
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i]
                )
                
            # draw label
            draw.text(tuple(text_origin), label, fill=(255,255,255), font=font)
            del draw
    
    return image

def process_lisa_results(response, image_path):
    row_result = []
    results = response.json()['results']
    
    for check in LISA_CHECK_TYPES:
        if check in results:
            row_result.append(results[check]["status"])
        else:
            row_result.append(-3)       # check not implemented
    
    image = Image.open(image_path)
    row_result.append(image.width)
    row_result.append(image.height)

    fail = any(v == 0 for v in row_result[1:])

    # Ensure photo is portrait
    if image.width > image.height:
        fail = True
    
    # TODO: Check min image size
    
    if fail:
        row_result.append('0')
    else:
        row_result.append('1')
    
    return row_result
    

def process_mobius_results(response):
    print(response.json())
    results = response.json()['Labels']
    detections = {}
    for result in results:
        detections[result['Label']] = False
        instances = result['Instances']
        for instance in instances:
            if instance['Confidence'] >= THRESHOLD:
                detections[result['Label']] = True
    return detections

# change inputs to api call (maybe dont need inputs anymore?)
def main(config):
    rows = []
    
    # ensure there are inputs
    if not os.path.isdir(INPUT_DIR):
        print("Input directory not found. Please place files in /inputs")
    
    # create output directory
    # if os.path.exists(output_directory):
    #     print("Output folder already exists. Deleting output folder...")
    #     shutil.rmtree(output_directory)
    # os.mkdir(output_directory)
    
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        print("Output directory did not exist and was created")
        
    csv_header = {"Filename"} 
    
    # loop through all files in data
    for filename in sorted(os.listdir(INPUT_DIR)):
        # TODO: check file type for images if it's LISA
        
        
        # create directory in output folder to store images with detections
        print(f"Generating outputs for {filename}...")
        # os.mkdir(os.path.join(OUTPUT_DIR, filename))
        
        # is_detected = False
        extract_frame(os.path.join(INPUT_DIR, filename))
        row_result = {}
        
        if API_CAP == 'mobius':
            row_result['Filename'] = filename
        
        # loop through all frames extracted 
        for frame_filename in sorted(os.listdir(FRAME_DIR)):
            image_path = os.path.join(FRAME_DIR, frame_filename)
            # call api per frame
            with open(image_path, 'rb') as image_file:
                b64str = base64.b64encode(image_file.read()).decode('utf-8')
                PARAMS = {
                    'image_base64': b64str,
                    'datetime': datetime.datetime.now().isoformat(),
                }

                HEADERS = {
                    'vas-api-key': config['api_key']
                }
                
                try:
                    if API_CAP == "lisa":
                        print("Querying LISA API...")
                        PARAMS['debug_image'] = False
                        response = requests.post(url=LISA_API, json=PARAMS, headers=HEADERS)
                        row_result = [filename] + process_lisa_results(response, image_path)
                        print("Query completed.")
                    elif API_CAP == "mobius":
                        print("Querying Mobius API...")
                        response = requests.post(url=MOBIUS_API, json=PARAMS, headers=HEADERS)
                        frame_result = process_mobius_results(response)
                        # update row_result - as long as detected in one frame, considered object detected
                        for label in frame_result:
                            if frame_result[label]:
                                row_result[label] = True
                            elif frame_result[label] == False and not label in row_result:
                                row_result[label] = False
                        print("Query completed.")
                        
                except Exception as e:
                    print(e)
                    print(f"Error calling API for image {filename}")
                    if API_CAP == 'lisa':
                        row_result.append(['Err-']*len(LISA_CHECK_TYPES))
            # try:
            #     image = Image.open(os.path.join(frame_directory, frame_filename))
            # except:
            #     print('{frame_filename} failed to open.')
            #     continue
            # else:
            #     """
            #     TODO: make api call to retrieve result
            #     """
                
            #     # detection, scores, classes = yolo.detect_image(image)
            #     count = 0
                
            #     # count number of detections above specified threshold
            #     for i, c in reversed(list(enumerate(classes))):
            #         score = scores[i]
            #         if score > threshold:
            #             count += 1
            #             """
            #             TODO: store output images with detections in output folder; not applicable for LISA
            #             """
            #             if API_CAP == "mobius":
            #                 image.save(os.path.join(output_directory, filename, frame_filename))
                    
            #     if count > 0:           # object above threshold is detected
            #         # detection.show()
            #         is_detected = True 
            #         # break               
        if os.path.exists(FRAME_DIR):
            shutil.rmtree(FRAME_DIR)

        # append to csv rows
        if row_result:
            csv_header.update(set(row_result.keys()))
        rows.append(row_result)
    

    # write to csv
    if API_CAP == 'lisa':
        csv_header = [check for check in LISA_CHECK_TYPES]
        csv_header = ["Image Name"] + csv_header + ["width","height","PASSED"]
    elif API_CAP == 'mobius':
        csv_header = list(csv_header)
    f = open(os.path.join(OUTPUT_DIR, 'results.csv'), 'w')
    
    if API_CAP == 'lisa':
        writer = csv.writer(f)
        writer.writerow(csv_header)
    elif API_CAP == 'mobius':
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()
        
    writer.writerows(rows) 
    f.close()

if __name__ == '__main__':
    """
    TODO: take in api call as argument??
    """
    # # TEST DRAWING BOXES AND SAVING IMAGES
    # # create output directory
    # if not os.path.exists(output_directory):
    #     os.mkdir(output_directory)
    # if not os.path.exists(os.path.join(output_directory, "mobius.png")):
    #     os.mkdir(os.path.join(output_directory, "mobius.png"))
    
    # # draw boxes
    # image = Image.open(os.path.join("data", "mobius.png"))
    # mobius = read_json(os.path.join("example_results", "mobius.json"))
    # image = draw_boxes(mobius["Labels"], image)
    
    # # save image
    # image.save(os.path.join(output_directory, "mobius.png", "frame1.png"))
    # image.show()

    
    
    
    # NEW VERSION OF PARSER
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lisa', action='store_true')
    group.add_argument('--mobius', action='store_true')
    
    parser.set_defaults(lisa=False, mobius=False)
    args = parser.parse_args()
    
    if args.lisa:
        API_CAP = "lisa"
    elif args.mobius:
        API_CAP = "mobius"
    
    config = read_json('config.json')
    LISA_API = config["lisa_api"] 
    MOBIUS_API = config["mobius_api"]
    main(config)
    
    
    # parser.add_argument(
    #     '--model', type=str, required=True,
    #     help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    # )
    
    # parser.add_argument(
    #     '--anchors', type=str,
    #     help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    # )

    # parser.add_argument(
    #     '--classes', type=str,
    #     help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    # )
    
    # parser.add_argument(
    #     "--input", nargs='?', type=str,required=False,default='./data',
    #     help = "Video input directory"
    # )
    
    # parser.add_argument(
    #     "--threshold", nargs='?', type=str,required=False,default=threshold,
    #     help = "Threshold to count detected objects"
    # )
    
    # FLAGS = parser.parse_args()
    # if "input" in FLAGS:
    #     data_directory = FLAGS.input
    # if "threshold" in FLAGS:
    #     threshold = FLAGS.threshold
    
    # config = read_json('config.json')
    # main(YOLO(**vars(FLAGS)), config)
    
    
    
    