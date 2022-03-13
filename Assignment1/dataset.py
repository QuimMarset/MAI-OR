import os
import numpy as np
import time
import random
from data_augmentation_utils import *
from load_utils import *
from other_utils import *


class Dataset:

    def __init__(self, images_path, annotations_path, image_names, image_size, seed):
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

    def __init__(self, images_path, annotations_path, image_names, image_size, segmentation_objects, 
        augmentation_mode, num_to_place, prob_augment=0.5, seed=0):
        
        super().__init__(images_path, annotations_path, image_names, image_size, seed)
        self.segmentation_objects = segmentation_objects
        self.augmentation_mode = augmentation_mode
        self.num_to_place = num_to_place
        self.prob_augment = prob_augment
        self.overlap = augmentation_mode == AugmentationMode.AugmentationOverlap


    def load_batch(self, batch_names):
        batch_size = len(batch_names)
        batch_images = np.zeros((batch_size, self.image_size, self.image_size, 3))
        batch_classes = np.zeros((batch_size, self.num_classes))

        num_batch_placed = 0

        for (index, image_name) in enumerate(batch_names):
            image = read_image(self.images_path, image_name, self.image_size)
            classes, boxes, width, height = read_annotation_file(self.annotations_path, image_name)

            if self.augmentation_mode > AugmentationMode.NoAugmentation and random.random() < self.prob_augment:
                scaled_boxes = [scale_bounding_box(bb, self.image_size, width, height) for bb in boxes]

                num_placed = corrupt_image(image, classes, scaled_boxes, self.segmentation_objects, 
                    self.num_to_place, self.overlap, self.image_size)

                num_batch_placed += num_placed
            
            batch_images[index] = image
            batch_classes[index] = to_one_hot(classes, self.num_classes, self.classes_dict)

        return batch_images, batch_classes, num_batch_placed/batch_size


class TrainBalancedDataset(Dataset):

    def __init__(self, images_path, annotations_path, image_names, image_size, place_per_label, segmentation_objects, augmentation_mode, seed):
        super().__init__(images_path, annotations_path, image_names, image_size, seed)
        self.segmentation_objects = segmentation_objects
        self.place_per_label = place_per_label
        self.num_to_place = round(np.sum(list(place_per_label.values()))/self.num_images)
        
        self.overlap = augmentation_mode == AugmentationMode.AugmentationOverlap
        self.augmentation_mode = augmentation_mode


    def to_class_indices(self, classes_names):
        class_indices = []
        for class_name in classes_names:
            class_indices.append(self.classes_dict[class_name])
        return class_indices


    def load_batch(self, batch_names):
        batch_size = len(batch_names)
        batch_images = np.zeros((batch_size, self.image_size, self.image_size, 3))
        batch_classes = np.zeros((batch_size, self.num_classes))
        batch_classes_counts = []

        place_per_label = self.place_per_label.copy()

        num_batch_placed = 0

        for (index, image_name) in enumerate(batch_names):

            image = read_image(self.images_path, image_name, self.image_size)
            classes, boxes, width, height = read_annotation_file(self.annotations_path, image_name)

            scaled_boxes = [scale_bounding_box(bb, self.image_size, width, height) for bb in boxes]

            if np.sum(list(place_per_label.values())) > 0:
                num_placed = corrupt_image_same_proportion(image, classes, scaled_boxes, self.segmentation_objects,
                    self.num_to_place, place_per_label, self.overlap, self.image_size)

                num_batch_placed += num_placed
            
            batch_images[index] = image
            batch_classes[index] = to_one_hot(classes, self.num_classes, self.classes_dict)
            batch_classes_counts.append(self.to_class_indices(classes))

        return batch_images, batch_classes, num_batch_placed/batch_size, batch_classes_counts