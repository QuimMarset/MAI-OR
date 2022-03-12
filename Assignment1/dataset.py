import os
import numpy as np
import time
import random
from data_augmentation_utils import *
from load_utils import *
from other_utils import *


class Dataset:

    def __init__(self, images_path, annotations_path, image_names, image_size, seed=0):
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.images_names = image_names
        self.num_images = len(image_names)

        self.image_size = image_size
        self.num_classes = 20
        self.seed = seed

        self.classes_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 
            'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 
            'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}


    def get_batch(self, index, batch_size):
        batch_names = self.images_names[index : index+batch_size]
        return self.load_batch(batch_names)
             

    def shuffle_paths(self):
        random.Random(self.seed).shuffle(self.images_names)

    
    def get_num_images(self):
        return self.num_images


    def load_batch(self, batch_names):
        batch_size = len(batch_names)
        batch_images = np.zeros((batch_size, self.image_size, self.image_size, 3))
        batch_classes = np.zeros((batch_size, self.num_classes))

        for (index, image_name) in enumerate(batch_names):
            image = read_image(self.images_path, image_name, self.image_size)
            classes, _, _, _ = read_annotation_file(self.annotations_path, image_name)
            
            batch_images[index] = image
            batch_classes[index] = to_one_hot(classes, self.num_classes, self.classes_dict)

        return batch_images, batch_classes


class TrainDataset(Dataset):

    def __init__(self, images_path, annotations_path, image_names, image_size, segmentation_objects, augmentation_mode, 
        overlap, num_to_place, possible_positions, prob_augment=0.5, seed=0):
        
        super().__init__(images_path, annotations_path, image_names, image_size, seed)
        self.segmentation_objects = segmentation_objects
        self.augmentation_mode = augmentation_mode
        self.overlap = overlap
        self.num_to_place = num_to_place
        self.prob_augment = prob_augment
        self.transform = augmentation_mode == AugmentationMode.AugmentationTransform
        self.possible_positions = possible_positions


    def load_batch(self, batch_names):
        batch_size = len(batch_names)
        batch_images = np.zeros((batch_size, self.image_size, self.image_size, 3))
        batch_classes = np.zeros((batch_size, self.num_classes))

        for (index, image_name) in enumerate(batch_names):
            image = read_image(self.images_path, image_name, self.image_size)
            classes, boxes, width, height = read_annotation_file(self.annotations_path, image_name)

            if self.augmentation_mode > AugmentationMode.NoAugmentation and random.random() < self.prob_augment:
                scaled_boxes = [scale_bounding_box(bb, self.image_size, width, height) for bb in boxes]

                image, classes = corrupt_image(image, classes, scaled_boxes, self.segmentation_objects, self.possible_positions[image_name],
                    self.num_to_place, self.overlap, self.image_size)
            
            batch_images[index] = image
            batch_classes[index] = to_one_hot(classes, self.num_classes, self.classes_dict)

        return batch_images, batch_classes