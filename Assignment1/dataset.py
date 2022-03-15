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
        self.seed = seed
        self.num_classes = get_num_classes()


    def get_batch(self, index, batch_size):
        batch_names = self.images_names[index : index+batch_size]
        return self.load_batch(batch_names)
             

    def shuffle_paths(self):
        random.Random(self.seed).shuffle(self.images_names)


    def load_batch(self, batch_names):
        batch_size = len(batch_names)
        batch_images = np.zeros((batch_size, self.image_size, self.image_size, 3))
        batch_classes = np.zeros((batch_size, self.num_classes))

        for (index, image_name) in enumerate(batch_names):
            image = read_image(self.images_path, image_name, self.image_size)
            classes, _, _, _ = read_annotation_file(self.annotations_path, image_name)
            
            batch_images[index] = image
            batch_classes[index] = to_one_hot(classes)

        return batch_images, batch_classes


class TrainDataset(Dataset):

    def __init__(self, images_path, annotations_path, image_names, image_size, segmentation_objects, 
        augmentation_mode, num_to_place, prob_augment=0.5, seed=0):
        
        super().__init__(images_path, annotations_path, image_names, image_size, seed)
        self.segmentation_objects = segmentation_objects
        self.augmentation_mode = augmentation_mode
        self.num_to_place = num_to_place
        self.prob_augment = prob_augment


    def load_batch(self, batch_names):
        batch_size = len(batch_names)
        batch_images = np.zeros((batch_size, self.image_size, self.image_size, 3))
        batch_classes = np.zeros((batch_size, self.num_classes))

        num_batch_placed = 0
        num_batch_augmented = 0

        for (index, image_name) in enumerate(batch_names):
            image = read_image(self.images_path, image_name, self.image_size)
            classes, boxes, width, height = read_annotation_file(self.annotations_path, image_name)

            if self.augmentation_mode.permits_augmentation() and random.random() < self.prob_augment:

                scaled_boxes = [scale_bounding_box(bb, self.image_size, width, height) for bb in boxes]

                num_placed = corrupt_image(image, classes, scaled_boxes, self.segmentation_objects, 
                    self.num_to_place, self.augmentation_mode, self.image_size)

                num_batch_placed += num_placed
                num_batch_augmented += 1
            
            batch_images[index] = image
            batch_classes[index] = to_one_hot(classes)

        if num_batch_augmented == 0: num_batch_augmented = 1e-6

        return batch_images, batch_classes, num_batch_placed/num_batch_augmented


class TrainBalancedDataset(Dataset):

    def __init__(self, images_path, annotations_path, image_names, image_size, place_per_label, segmentation_objects, augmentation_mode, seed):
        super().__init__(images_path, annotations_path, image_names, image_size, seed)
        self.segmentation_objects = segmentation_objects
        self.place_per_label = place_per_label
        self.num_to_place = round(np.sum(list(place_per_label.values()))/self.num_images)
        self.augmentation_mode = augmentation_mode


    def load_batch(self, batch_names):
        batch_size = len(batch_names)
        batch_images = np.zeros((batch_size, self.image_size, self.image_size, 3))
        batch_classes = []

        num_batch_placed = 0
        num_batch_augmented = 0

        for (index, image_name) in enumerate(batch_names):

            image = read_image(self.images_path, image_name, self.image_size)
            classes, boxes, width, height = read_annotation_file(self.annotations_path, image_name)

            scaled_boxes = [scale_bounding_box(bb, self.image_size, width, height) for bb in boxes]

            if self.place_per_label:
                
                num_placed = corrupt_image_same_proportion(image, classes, scaled_boxes, self.segmentation_objects,
                    self.num_to_place, self.place_per_label, self.augmentation_mode, self.image_size)

                num_batch_placed += num_placed
                num_batch_augmented += 1
            
            batch_images[index] = image
            batch_classes.append(classes)

            if num_batch_augmented == 0: num_batch_augmented = 1e-6

        return batch_images, batch_classes, num_batch_placed/num_batch_augmented

    
    def reset_place_per_label(self, place_per_label):
        self.place_per_label = place_per_label