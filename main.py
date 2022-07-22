import os
from pickle import FRAME
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

API_URL = None
API_KEY = None

# LISA check constants
# LISA_CHECK_TYPES = [
#         'background_check',
#         'blur_check',
#         'eye_close_check',
#         'face_front_check',
#         'face_presence_check',
#         'file_type_check',
# #        'frame_cover_eye_check',
#         'gaze_check',
# #        'hair_cover_eye_check',
# #        'headcover_check',
# #        'image_size_check',
# #        'lighting_check',
# #        'mouth_open_check',
# #        'shoulder_alignment_check',
#         'skin_specular_reflection_check',
# #        'watermark_check'
#     ]
# CHECK_IMG_MINSIZE = False
# MIN_WIDTH = 100
# MIN_HEIGHT = 120

def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def extract_frame(filename):
    print(f"Extracting frames from {filename}...")
    
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
    print(f"{count} frame(s) extracted from {filename}.")

def draw_box(label, instance, image):
    """
    Draw box based on api results; follwing mobius results
    """
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 500
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

# def process_lisa_results(response, image_path):
#     row_result = []
#     results = response.json()['results']
    
#     for check in LISA_CHECK_TYPES:
#         if check in results:
#             row_result.append(results[check]["status"])
#         else:
#             row_result.append(-3)       # check not implemented
    
#     image = Image.open(image_path)
#     row_result.append(image.width)
#     row_result.append(image.height)

#     fail = any(v == 0 for v in row_result[1:])

#     # Ensure photo is portrait
#     if image.width > image.height:
#         fail = True
    
#     # TODO: Check min image size
    
#     if fail:
#         row_result.append('0')
#     else:
#         row_result.append('1')
    
#     return row_result
    

def process_mobius_results(response, image_path):
    # print(response.json())        # for debugging
    image = Image.open(image_path)
    try: 
        results = response.json()['Labels']
        detections = {}
        for result in results:
            label = result['Label']
            instances = result['Instances']
            
            for instance in instances:
                if instance['Confidence'] >= THRESHOLD:
                    image = draw_box(label, instance, image)
                    detections[label] = True
        
        return detections, image
    except Exception as e:
        print(response.json())
        return {}, image

def lisa():
    pass

def mobius():
    csv_header = set()
    csv_rows = []
    for filename in sorted(os.listdir(INPUT_DIR)):

        # create directory in output folder to store images with detections
        print(f"\nGenerating outputs for {filename}...")
        file_output_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(file_output_path):
            os.mkdir(file_output_path)
            print(f"Output directory {file_output_path} did not exist and was created.")
        
        # extract 5 frames per second
        extract_frame(os.path.join(INPUT_DIR, filename))

        csv_row = {}
        csv_row['Filename'] = filename
        
        for frame_filename in sorted(os.listdir(FRAME_DIR)):
            print(f"Processing {frame_filename}...")
            image_path = os.path.join(FRAME_DIR, frame_filename)
            
            with open(image_path, 'rb') as image_file:
                b64str = base64.b64encode(image_file.read()).decode()

                params = {
                    'image_base64': b64str,
                    'datetime': datetime.datetime.now().isoformat()
                }
                
                headers = {
                    'vas-api-key': API_KEY
                }
                
                response = requests.post(url=API_URL, json=params, headers=headers)
                detections, image = process_mobius_results(response, image_path)
                if len(detections.keys()) > 0:
                    image.save(os.path.join(file_output_path, frame_filename))
                csv_row.update(detections)
                csv_header.update(set(detections.keys()))
        
        csv_rows.append(csv_row)
    
    # Write to csv file
    csv_header = list(csv_header)
    csv_header.insert(0, "Filename")        # Puts filename at the first column
    
    csv_file = open(os.path.join(OUTPUT_DIR, 'results.csv'), 'w')
    writer = csv.DictWriter(csv_file, csv_header)
    writer.writeheader()
    writer.writerows(csv_rows)
    csv_file.close()   
        

def main(args):
    global API_URL, API_KEY
    config = read_json('config.json')
    
    # Ensure there are inputs
    if not os.path.isdir(INPUT_DIR):
        print("Input directory not found. Please place files in /inputs")
    
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        print("Output directory did not exist and was created")
    
    if args.lisa:
        API_URL = config['lisa_api']
        API_KEY = config['lisa_api_key']
        lisa()
    elif args.mobius:
        API_URL = config['mobius_api']
        API_KEY = config['mobius_api_key']         # this doesnt work yet
        mobius()
    
    # Remove frame directory
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    print("\nDone")

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
    main(args)
    
    
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
    
    
    
    