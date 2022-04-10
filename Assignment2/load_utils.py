import cv2
from colors_labels import *
import os
import json
from collections import defaultdict


def load_image(images_path, image_name):
    path = os.path.join(images_path, f'{image_name}.jpg')
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_segmentation(segmentations_paths, image_name):
    path = os.path.join(segmentations_paths, f'{image_name}_seg.png')
    segmentation = cv2.imread(path)
    segmentation = cv2.cvtColor(segmentation, cv2.COLOR_RGB2GRAY)
    return segmentation


def load_json_log(json_log_path):
    log_dict = dict()
    
    with open(json_log_path, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            
            if 'epoch' not in log:
                continue
            
            epoch = log.pop('epoch')
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            
            for k, v in log.items():
                log_dict[epoch][k].append(v)
    
    return log_dict