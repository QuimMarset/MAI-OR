import os
import numpy as np
import time
import random
from data_augmentation_utils import *
from load_utils import *
from other_utils import to_one_hot, scale_bounding_box


class Dataset:

    def __init__(self, dataset_path, image_names, image_size, seed=0):
        self.segmentation_folder = "SegmentationObject"
        self.annotation_folder = "Annotations"
        self.images_folder = "JPEGImages"

        self.dataset_path = dataset_path
        self.images_names = image_names

        self.image_size = image_size
        self.num_classes = 20
        self.seed = seed

        self.classes = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 
            'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 
            'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}


    def get_batch(self, index, batch_size):
        batch_names = self.images_names[index : index+batch_size]
        batch_images, batch_classes = self.load_batch(batch_names)
        return batch_images, batch_classes
    

    def shuffle_paths(self):
        random.Random(self.seed).shuffle(self.images_names)


    def load_batch(self, batch_names):
        batch_size = len(batch_names)
        batch_images = np.zeros((batch_size, self.image_size, self.image_size, 3))
        batch_classes = np.zeros((batch_size, self.num_classes))

        image_path = os.path.join(self.dataset_path, self.images_folder)
        annot_path = os.path.join(self.dataset_path, self.annotation_folder)

        for (index, image_name) in enumerate(batch_names):
            image = read_image(image_path, image_name, self.image_size)
            classes, _, _, _ = read_annotation_file(annot_path, image_name)
            
            batch_images[index] = image
            batch_classes[index] = to_one_hot(classes)

        return batch_images, batch_classes


class TrainDataset(Dataset):

    def __init__(self, dataset_path, image_names, image_size, segmentation_objects, augmentation_mode=0, seed=0):
        super.__init__(dataset_path, image_names, image_size, seed)
        self.segmentation_objects = segmentation_objects


    def load_batch(self, batch_names):
        batch_size = len(batch_names)
        batch_images = np.zeros((batch_size, self.image_size, self.image_size, 3))
        batch_classes = np.zeros((batch_size, self.num_classes))

        image_path = os.path.join(self.dataset_path, self.images_folder)
        annot_path = os.path.join(self.dataset_path, self.annotation_folder)

        for (index, image_name) in enumerate(batch_names):
            image = read_image(image_path, image_name, self.image_size)
            classes, boxes, width, height = read_annotation_file(annot_path, image_name)
            scaled_boxes = [scale_bounding_box(bb, self.image_size, width, height) for bb in boxes] 

            if self.augmentation:
                image, classes = corrupt_image(image, classes, scaled_boxes, self.segmentation_objects)
            
            batch_images[index] = image
            batch_classes[index] = to_one_hot(classes)

        return batch_images, batch_classes