import numpy as np


def to_one_hot(classes, num_classes, classes_dict):
    one_hot = np.zeros((num_classes))
    for class_name in classes:
        one_hot[classes_dict[class_name]] = 1
    return one_hot


def get_box_from_mask(mask):
    indices = np.where(mask == 1.0)
    return [min(indices[1]), min(indices[0]), max(indices[1]), max(indices[0])]


def scale_bounding_box(bounding_box, image_size, original_width, original_height):
    
    def _scale_coordinate(coordinate, size):
        return int(image_size*coordinate/size)

    bb_xmin = _scale_coordinate(bounding_box[0], original_width)
    bb_ymin = _scale_coordinate(bounding_box[1], original_height)
    bb_xmax = _scale_coordinate(bounding_box[2], original_width)
    bb_ymax = _scale_coordinate(bounding_box[3], original_height)

    return [bb_xmin, bb_ymin, bb_xmax, bb_ymax]