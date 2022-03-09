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

    return classes, bounding_box, width, height


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
    segmentation = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return segmentation


def exists_segmentations_pickle(path):
    return os.path.exists(path)


def save_segmentations(path, segmentation_objects):
    with open(path, "wb") as file:
        pickle.dump(segmentation_objects, file)


def load_segmentations_pickle(path):
    with open(path, "rb") as file:
        segmentation_objects = pickle.load(file)
    return segmentation_objects


def create_train_val_split(num_images, val_percentage, seed):
    np.random.seed(seed)
    val_indices = np.random.randint(0, num_images, int(num_images*val_percentage))
    train_val_split = np.zeros(num_images)
    train_val_split[val_indices] = 1
    return train_val_split