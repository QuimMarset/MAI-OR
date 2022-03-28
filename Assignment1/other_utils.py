import numpy as np
from enum import IntEnum


classes_dict = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 
    'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 
    'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}

def get_num_classes():
    return 20


def to_one_hot(classes):
    one_hot = np.zeros(get_num_classes())
    for class_name in classes:
        one_hot[classes_dict[class_name]] = 1
    return one_hot


def get_box_from_mask(mask):
    indices = np.where(mask == 1.0)
    return [min(indices[1]), indices[0][0], max(indices[1]), indices[0][-1]]


def scale_bounding_box(bounding_box, image_size, original_width, original_height):
    
    def _scale_coordinate(coordinate, size):
        return int(image_size*coordinate/size)

    bb_xmin = _scale_coordinate(bounding_box[0], original_width)
    bb_ymin = _scale_coordinate(bounding_box[1], original_height)
    bb_xmax = _scale_coordinate(bounding_box[2], original_width)
    bb_ymax = _scale_coordinate(bounding_box[3], original_height)

    return [bb_xmin, bb_ymin, bb_xmax, bb_ymax]


class AugmentationMode(IntEnum):
    # Without performing data augmentation
    NoAugmentation = 1
    
    # Data augmentation without overlap
    Augmentation = 2
    
    # Data augmentation without overlap, random position, rotation
    AugmentationRotate = 3
    
    # Data augmentation without overlap, random position, scaling
    AugmentationScale = 4
    
    # Data augmentation without overlap, random position, scaling and rotation
    AugmentationTransform = 5
    
    # Data augmentation without overlap and having same class proportion
    AugmentationSameProportion = 6

    # Data augmentation without overlap, random position, random rotation, and same proportion
    AugmentationRotateSameProportion = 7
    
    # Data augmentation without overlap, random position, random scaling, and same proportion
    AugmentationScaleSameProportion = 8
    
    # Data augmentation without overlap, random position, random scaling and rotation, and same proportion
    AugmentationTransformSameProportion = 9

    # Data augmentation permitting overlap
    AugmentationOverlap = 10

    # Data augmentation permitting overlap, random position, random rotation
    AugmentationOverlapRotate = 11
    
    # Data augmentation permitting overlap, random position, random scaling
    AugmentationOverlapScale = 12

    # Data augmentation permitting overlap and random position, scaling and rotation
    AugmentationOverlapTransform = 13

    # Data augmentation permitting overlap and same class proportion
    AugmentationOverlapSameProportion = 14

    # Data augmentation permitting overlap, random rotation, and same class proportion
    AugmentationOverlapRotateSameProportion = 15

    # Data augmentation permitting overlap, random scaling, and same class proportion
    AugmentationOverlapScaleSameProportion = 16

    # Data augmentation permitting overlap, random transform, and same class proportion
    AugmentationOverlapTransformSameProportion = 17

    def permits_augmentation(self):
        return self.value > 1

    def permits_overlap(self):
        return self.value >= 10

    def permits_transformations(self):
        return self.value == 5 or self.value == 9 or self.value == 13 or self.value == 17

    def permits_only_scaling(self):
        return self.value == 4 or self.value == 8 or self.value == 12 or self.value == 16

    def permits_only_rotation(self):
        return self.value == 3 or self.value == 7 or self.value == 11 or self.value == 15

    def maintains_proportion(self):
        return self.value >= 6 and self.value <= 9 or self.value >= 14