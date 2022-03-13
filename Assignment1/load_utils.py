import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from skimage import img_as_float
import pickle


def read_annotation_file(annotations_path, image_name):
    file_path = os.path.join(annotations_path, f'{image_name}.xml')
    tree = ET.parse(file_path)
    root = tree.getroot()

    classes = []
    bounding_boxes = []

    dimensions = root.find('size')
    width = int(dimensions.find('width').text)
    height = int(dimensions.find('height').text)

    for object in root.iter('object'):
        class_name = object.find('name').text
        bb_xmin = int(object.find("bndbox/xmin").text)
        bb_ymin = int(object.find("bndbox/ymin").text)
        bb_xmax = int(object.find("bndbox/xmax").text)
        bb_ymax = int(object.find("bndbox/ymax").text)

        classes.append(class_name)
        bounding_box = [bb_xmin, bb_ymin, bb_xmax, bb_ymax]
        bounding_boxes.append(bounding_box)

    return classes, bounding_boxes, width, height


def read_image(images_path, image_name, image_size):
    path = os.path.join(images_path, f'{image_name}.jpg')
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_as_float(cv2.resize(image, (image_size, image_size)))
    return image


def exists_segmentation(segmentations_path, image_name):
    return os.path.exists(os.path.join(segmentations_path, f'{image_name}.png'))


def get_segmentation(segmentations_path, image_name):
    path = os.path.join(segmentations_path, f'{image_name}.png')
    segmentation = cv2.imread(path)
    segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB)
    return segmentation


def exists_path(path):
    return os.path.exists(path)


def load_pickle_dict(path):
    with open(path, "rb") as file:
        dictionary = pickle.load(file)
    return dictionary


def save_dict_to_pickle(path, dictionary):
    with open(path, "wb") as file:
        pickle.dump(dictionary, file)


def read_split_names(path):
    split_names = []
    with open(path, 'r') as file:
        for line in file.readlines():
            split_names.append(line.strip())
    return split_names


def read_train_val_split(train_split_path, val_split_path):
    train_names = read_split_names(train_split_path)
    val_names = read_split_names(val_split_path)
    return train_names, val_names


def create_results_folder(path):
    os.makedirs(path, exist_ok=True)


def extract_train_classes_counts(annotations_path, image_names):
    labels_counts = {}

    for image_name in image_names:
        classes, _, _, _ = read_annotation_file(annotations_path, image_name)

        for class_name in classes:
            labels_counts[class_name] = labels_counts.get(class_name, 0) + 1

    return labels_counts